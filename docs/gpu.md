# GPU runbook

mini-vLLM is CPU-first, and every feature works on a laptop. This runbook
exists for the day you want CUDA numbers: rent a small GPU for under a
dollar, rerun the benchmark suite with `--device cuda`, and commit the
results next to the CPU ones. Nothing in the code needs to change; the
device flag is wired through every path already.

## Renting a box

Either provider works; both bill by the minute.

**vast.ai**: filter for RTX 3050/3060 (4 to 12 GB VRAM is plenty for the
models here), pick an image with CUDA 12.x and Python 3.11+, for example
`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`. Typical price for this
class is $0.05 to $0.12 per hour.

**RunPod**: Community Cloud, same GPU class, the official PyTorch pod
template. Comparable pricing.

Add your SSH public key in the provider's console before launching.

## Setup on the box (about 5 minutes)

```bash
ssh -p <port> root@<host>

# uv + clone
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
git clone https://github.com/Mihawii/mini-vllm.git
cd mini-vllm
uv sync

# sanity: CUDA visible?
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest -q          # suite stays CPU-based; confirms the install
```

If `uv sync` resolves a CPU-only torch on the pod, install the CUDA build
explicitly first: `uv pip install torch --index-url
https://download.pytorch.org/whl/cu124`.

## The runs that matter

```bash
# headline: all baselines on GPU, fp16
uv run mini-vllm bench --model distilbert/distilgpt2 --device cuda --dtype float16 \
    --requests 16 --max-new-tokens 64 --concurrency 8 \
    --compare-kv-cache --baselines --prompt-lengths 16,64,256

# speculative decoding: the draft/target cost ratio is far better on GPU
uv run mini-vllm bench --model gpt2 --device cuda --dtype float16 \
    --requests 8 --max-new-tokens 64 --no-compare-kv-cache --batch-sizes "1" \
    --speculative-draft distilbert/distilgpt2 --gamma 4

# a bigger target becomes practical with VRAM (gpt2-medium, 355M)
uv run mini-vllm bench --model gpt2-medium --device cuda --dtype float16 \
    --requests 8 --max-new-tokens 64 --baselines \
    --speculative-draft distilbert/distilgpt2 --gamma 4
```

Copy the artifacts back before destroying the pod:

```bash
scp -P <port> root@<host>:mini-vllm/benchmarks/results/bench-*.json benchmarks/results/
scp -P <port> root@<host>:mini-vllm/benchmarks/results/bench-*.csv benchmarks/results/
```

Then locally: `uv run mini-vllm bench-report`, update the README table with
the GPU rows, commit, and destroy the pod. Expected cost for the whole
session: well under one dollar.

## What to expect

- float16 on CUDA is the default the loader picks for `--dtype auto`.
- Batched throughput scales much further than on CPU; raise `--batch-sizes
  1,2,4,8,16,32` and `--concurrency` until the curve flattens.
- Speculative decoding should cross 1.0x here: a GPU forward pass has high
  fixed cost, so verifying four tokens in one target pass beats four
  passes by a wide margin once acceptance is decent.
- The dashboard and simulate work unchanged; `--cache-backend paged
  --prefix-caching` behaves identically on GPU.

## Troubleshooting

- `CUDA was requested but torch.cuda.is_available() is False`: the pod
  image shipped CPU torch; use the cu124 index install above.
- OOM on gpt2-medium at high concurrency: lower `--pool-blocks` or
  concurrency; the paged backend's preemption will also shed load by
  design.
- Driver mismatch errors on vast.ai: pick an image whose CUDA minor
  version matches the host driver shown on the listing card.

## The kernel experiment that belongs here

The roadmap's Triton paged-attention kernel needs this box too: a fused
kernel that reads K/V straight from the block pool through the block
table, compared against our gather-then-attend path. That experiment turns
the one honest "not vLLM-grade" caveat in this repo into measured kernel
work. Sketch lives in the roadmap until the GPU session happens.
