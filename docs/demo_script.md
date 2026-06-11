# Demo script

Exact commands for a live demo, the screenshot session, or the 60-second
video. Run them from the repo root in this order; earlier steps create the
data later steps display.

## Setup (once, before recording)

```bash
uv sync
uv run mini-vllm inspect --model distilbert/distilgpt2   # warms the model cache
```

## Scene 1: what model are we serving (5 s)

```bash
uv run mini-vllm inspect --model distilbert/distilgpt2
```

Say: 82M parameter GPT-2 variant, CPU, float32, 1024-token context.

## Scene 2: tokens are the unit of work (5 s)

```bash
uv run mini-vllm tokenize "The greenhouse robot inspected the tomatoes."
```

Say: the engine never sees words, it sees these ids.

## Scene 3: streaming generation (10 s)

```bash
uv run mini-vllm generate "An inference engine is" --stream --max-new-tokens 60 --seed 7
```

Say: our own decode loop, one forward pass per token, printed as it lands.

## Scene 4: the KV cache is the whole trick (10 s)

```bash
uv run mini-vllm generate "Write a haiku about GPUs" -n 80 --no-kv-cache
uv run mini-vllm generate "Write a haiku about GPUs" -n 80 --kv-cache
```

Say: same output, several times faster, because cached decoding feeds one
token instead of re-running the whole sequence.

## Scene 5: continuous batching (15 s)

```bash
uv run mini-vllm simulate examples/traffic.json --model distilbert/distilgpt2
```

Say: requests arrive over time, the batch grows and shrinks per tick, a
queue forms when slots run out, short requests leave early.

## Scene 6: it is a real server (15 s)

Terminal A:

```bash
uv run mini-vllm serve --model distilbert/distilgpt2
```

Terminal B:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Mini-vLLM is", "max_tokens": 50, "temperature": 0.7}' | python3 -m json.tool

curl -N http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain KV cache", "max_tokens": 40, "stream": true}'

uv run mini-vllm stats
```

Then open http://127.0.0.1:8000/dashboard and click through the four tabs.

Say: OpenAI-style API, SSE streaming, live metrics, dashboard.

## Scene 6b: paged cache and preemption (10 s, optional)

```bash
uv run mini-vllm simulate examples/traffic.json --model distilbert/distilgpt2 \
    --cache-backend paged --block-size 16 --pool-blocks 24 --prefill-chunk-size 32
```

Say: cache memory is a pool of blocks; squeeze the pool and watch requests
get preempted and recover with identical output.

## Scene 7: receipts (10 s)

```bash
uv run mini-vllm bench --model distilbert/distilgpt2 --requests 8 --compare-kv-cache
uv run pytest
```

Say: measured on this laptop, and the test suite proves the cached loop,
the batch path, and the scheduler all produce identical greedy output.

## 60-second video cut

Scenes 3, 4, 5, then the dashboard playground streaming a completion, then
the pytest summary line. Skip scenes 1 and 2 if tight; the inspect table can
be a still frame under the title card.
