# Demo video

`docs/assets/demo.mp4` is a 2-minute silent walkthrough, built entirely by
script from a live server: real browser sessions recorded headlessly, real
terminal captures, caption cards in the project's visual style, stitched
with ffmpeg. Rebuild it any time:

```bash
# prereqs: a server with saved bench results and a simulation, plus
# playwright-core (npm i playwright-core) and any ffmpeg on PATH
uv run mini-vllm serve --model distilbert/distilgpt2 --port 8742 &
./scripts/make_video.sh http://127.0.0.1:8742
```

## Timeline

| t | segment | source |
|---|---|---|
| 0:00 | title card | scripts/video/cards/title.html |
| 0:05 | playground: prompt typed, two completions stream live with metrics readouts | recorded browser session |
| 0:30 | terminal: `mini-vllm generate` capture | docs/assets/generate.svg |
| 0:35 | tokenizer inspector: text typed, segments and table appear | recorded browser session |
| 0:50 | caption: continuous batching + preemption | cards/batching.html |
| 0:55 | terminal: `mini-vllm simulate` with the paged pool column | docs/assets/simulate.svg |
| 1:00 | scheduler tab scroll: occupancy chart, request gantt, per-request table, pool chart with preemptions | recorded browser session |
| 1:25 | caption: measured numbers | cards/bench.html |
| 1:30 | terminal: benchmark report | docs/assets/bench-report.svg |
| 1:35 | benchmarks tab scroll: baselines table, KV cache bars, batching, quantization | recorded browser session |
| 1:50 | API request + metrics endpoint | cards/api.html |
| 1:55 | closing card with the repo URL | cards/closing.html |

## Re-recording with voice

The silent cut works as a README artifact. For a voiced version, screen
record yourself following the same timeline and read one sentence per
segment; docs/demo_script.md has the speaking lines.
