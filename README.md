# cudagraphs-pytorch

PyTorch で CUDA Graphs を使ってニューラルネットワークの推論を高速化するサンプルコードです。

詳細な解説は Zenn の記事をご覧ください:
https://zenn.dev/kosuke_sugimoto/articles/1614e65b27f6b4

## 動作環境

- Python 3.12
- CUDA 対応 NVIDIA GPU
- [uv](https://github.com/astral-sh/uv) (パッケージマネージャ)

## 環境構築

```bash
git clone https://github.com/Kosuke-Sugimoto/cudagraphs-pytorch
cd cudagraphs-pytorch
uv sync
```

## 実行

### ベンチマーク (CUDA Graphs あり)

```bash
uv run cudagraphs_pytorch.py
```

### ベンチマーク (CUDA Graphs なし / eager モード)

```bash
uv run cudagraphs_pytorch.py --eager
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--batch` | 1 | バッチサイズ |
| `--seq` | 128 | シーケンス長 |
| `--dim` | 256 | モデルの次元数 |
| `--layers` | 8 | デコーダ層数 |
| `--heads` | 8 | アテンションヘッド数 |
| `--mlp_hidden` | 512 | MLP の中間次元数 |
| `--vocab` | 32000 | 語彙サイズ |
| `--warmup` | 20 | ウォームアップ回数 |
| `--iters` | 200 | ベンチマーク反復回数 |
| `--dtype` | fp16 | データ型 (`bf16` / `fp16` / `fp32`) |
| `--eager` | - | CUDA Graphs を無効化して eager モードで実行 |

## プロファイリング (Nsight Systems)

```bash
# eager モード
nsys profile -o eager.nsys-rep -f true -c cudaProfilerApi --trace=cuda,nvtx \
  uv run cudagraphs_pytorch.py --eager

# CUDA Graphs あり
nsys profile -o graphs.nsys-rep -f true -c cudaProfilerApi \
  --trace=cuda,nvtx --cuda-graph-trace=graph \
  uv run cudagraphs_pytorch.py
```
