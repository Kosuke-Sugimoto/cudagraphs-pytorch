import argparse
import nvtx
import time
import torch
import torch.nn as nn

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32
}

def hook(module, i, o):
    torch.cuda.nvtx.range_pop()

def pre_hook(module, i):
    torch.cuda.nvtx.range_push(f"{module}")

def pre_hook_simple(module, i):
    name = repr(module).split("(")[0]
    torch.cuda.nvtx.range_push(f"{name}")

def register_hooks(simple=True):
    if simple:
        nn.modules.module.register_module_forward_pre_hook(pre_hook_simple)
    else:
        nn.modules.module.register_module_forward_pre_hook(pre_hook)
    
    nn.modules.module.register_module_forward_hook(hook)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x_fp32 * rms
        return (x_norm.to(x.dtype)) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, attn_mask):
        b, s, d = x.shape
        h = self.num_heads
        hd = self.head_dim

        q = self.q_proj(x).view(b, s, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, s, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, s, h, hd).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
        attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, mlp_hidden)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyQwenLikeDecoder(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, mlp_hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            DecoderBlock(dim, num_heads, mlp_hidden) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, attn_mask):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.final_norm(x)
        return self.lm_head(x)


def make_causal_mask(seq_len, device, dtype):
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)
    return mask.to(dtype)


class CUDAGraphs:
    def __init__(self, model, example_input_ids, example_attn_mask, warmup=10):
        self._model = model
        self._graph = torch.cuda.CUDAGraph()

        self._static_input_ids = example_input_ids.clone()
        self._static_attn_mask = example_attn_mask.clone()
        self._static_output = None
        self._captured = False
        self._warmup = warmup

    def capture(self):
        if self._captured:
            raise RuntimeError("Graph has already been captured.")

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream()) # current stream の実行完了まで待つ

        with torch.cuda.stream(warmup_stream), nvtx.annotate("warmup", color="blue"):
            for _ in range(self._warmup):
                self._static_output = self._model(
                    self._static_input_ids,
                    self._static_attn_mask,
                )

        torch.cuda.current_stream().wait_stream(warmup_stream) # warmup stream の実行完了を待つ

        with torch.cuda.graph(self._graph), nvtx.annotate("capture", color="blue"):
            self._static_output = self._model(
                self._static_input_ids,
                self._static_attn_mask,
            )

        self._captured = True

    def replay(self):
        with nvtx.annotate("replay", color="blue"):
            if not self._captured:
                raise RuntimeError("Graph has not been captured yet.")
            self._graph.replay()

    def output_reset(self):
        if self._static_output is None:
            raise RuntimeError("Output buffer has not been created yet.")
        self._static_output.zero_()

    def get_input(self):
        return {
            "input_ids": self._static_input_ids,
            "attn_mask": self._static_attn_mask,
        }

    def get_output(self):
        if self._static_output is None:
            raise RuntimeError("Output buffer has not been created yet.")
        return self._static_output

    def run(self, input_ids, attn_mask):
        if not self._captured:
            raise RuntimeError("Graph has not been captured yet.")

        if input_ids.shape != self._static_input_ids.shape:
            raise ValueError(
                f"input_ids shape mismatch: got {tuple(input_ids.shape)}, "
                f"expected {tuple(self._static_input_ids.shape)}"
            )
        if attn_mask.shape != self._static_attn_mask.shape:
            raise ValueError(
                f"attn_mask shape mismatch: got {tuple(attn_mask.shape)}, "
                f"expected {tuple(self._static_attn_mask.shape)}"
            )

        self._static_input_ids.copy_(input_ids)
        self._static_attn_mask.copy_(attn_mask)
        self.replay()
        return self.get_output()


@torch.inference_mode()
def run_eager(model, input_ids, attn_mask, warmup, iters):
    with nvtx.annotate("warmup", color="blue"):
        for _ in range(warmup):
            _ = model(input_ids, attn_mask)
        torch.cuda.synchronize()

    with nvtx.annotate("run", color="blue"):
        for _ in range(iters):
            _ = model(input_ids, attn_mask)
        torch.cuda.synchronize()


@torch.inference_mode()
def run_graph(model, input_ids, attn_mask, warmup, iters):
    graphs = CUDAGraphs(
        model=model,
        example_input_ids=input_ids,
        example_attn_mask=attn_mask,
        warmup=warmup,
    )
    graphs.capture()

    with nvtx.annotate("run", color="blue"):
        for _ in range(iters):
            _ = graphs.run(input_ids, attn_mask)
        torch.cuda.synchronize()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_hidden", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    dtype = DTYPE_MAP[args.dtype]

    model = TinyQwenLikeDecoder(
        vocab_size=args.vocab,
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers,
        mlp_hidden=args.mlp_hidden,
    ).to(device, dtype=dtype).eval()

    input_ids = torch.randint(
        0, args.vocab, (args.batch, args.seq),
        device=device, dtype=torch.long
    )
    attn_mask = make_causal_mask(args.seq, device, dtype)

    print(f"params: {count_parameters(model)/1e6:.2f}M")
    print(f"config: {args}")

    # register_hooks()

    torch.cuda.profiler.start()

    if args.eager:
        run_eager(model, input_ids, attn_mask, args.warmup, args.iters)
    else:
        run_graph(model, input_ids.clone(), attn_mask, args.warmup, args.iters)

    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
