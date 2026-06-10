# Benchmarks

Run the suite:

```
uv run mini-vllm bench --model distilbert/distilgpt2 --requests 8 \
    --max-new-tokens 64 --concurrency 4 --compare-kv-cache --prompt-lengths 16,64,256
```

Results land in `results/` as a timestamped JSON (full detail) and CSV (flat
rows). `uv run mini-vllm bench-report` renders the newest run as Markdown
and writes `results/report.md`.

## Scenarios

| Scenario | Question it answers |
|---|---|
| latency | What does one request cost end to end, sequentially? |
| kv_cache | How much does the cache save on identical greedy workloads? |
| batch | How does static batch size trade latency for throughput? |
| concurrency | What does iteration-level scheduling gain over one-at-a-time serving? |
| prompt_sweep | How does prefill cost scale with prompt length while decode speed stays flat? |

## Methodology notes

- All scenarios decode greedily. Tests prove greedy output is identical with
  and without the KV cache and identical between batched and single-request
  paths, so timing comparisons measure the same work.
- A warmup generation runs first; the first forward pass pays one-off
  allocation and autotuning costs that would otherwise pollute run one.
- The kv_cache scenario caps at 5 requests because the uncached path is
  quadratic by design; the point is the ratio, never the absolute count.
- Machine info (CPU, platform, torch version) is embedded in every result
  file, because tok/s numbers without hardware context are noise.

The committed results in `results/` are from the machine described inside
them. Treat them as one honest data point, and rerun on your own hardware.
