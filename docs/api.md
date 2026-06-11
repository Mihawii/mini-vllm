# HTTP API reference

Start the server, then open http://127.0.0.1:8000/docs for the interactive
OpenAPI page.

```
mini-vllm serve --model distilbert/distilgpt2 --port 8000
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, model name, device, uptime |
| GET | `/models`, `/v1/models` | Loaded model card with metadata |
| GET | `/metrics` | JSON metrics (requests, tokens, latency percentiles, scheduler state) |
| GET | `/metrics/prometheus` | Same data in Prometheus text format |
| POST | `/tokenize` | Token breakdown for a text |
| POST | `/v1/completions` | Text completion, optional SSE streaming |
| POST | `/v1/chat/completions` | Chat completion, optional SSE streaming |
| POST | `/benchmark` | Small capped load test through the scheduler |
| GET | `/benchmark/results` | Saved benchmark runs for the dashboard |
| GET | `/simulations/latest` | Latest scheduler simulation for the dashboard |
| GET | `/dashboard` | The dashboard UI |

## Completions

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Mini-vLLM is",
    "max_tokens": 80,
    "temperature": 0.7,
    "top_p": 0.9,
    "seed": 42
  }'
```

Response:

```json
{
  "id": "cmpl-9b3f0a2e41c7",
  "object": "text_completion",
  "created": 1781117751,
  "model": "distilbert/distilgpt2",
  "choices": [{ "index": 0, "text": " a small inference engine...", "finish_reason": "length" }],
  "usage": { "prompt_tokens": 4, "completion_tokens": 80, "total_tokens": 84 },
  "mini_vllm": {
    "latency_ms": 1182.4,
    "ttft_ms": 95.4,
    "queue_ms": 3.1,
    "tokens_per_second": 67.7
  }
}
```

The `mini_vllm` block is an extension: total latency, time to first token,
time spent queued before a batch slot opened, and decode throughput.

### Request fields

| Field | Default | Notes |
|---|---|---|
| `prompt` / `messages` | required | string, or chat messages with `role` and `content` |
| `max_tokens` | 64 | 1 to 2048, also bounded by the model context window |
| `temperature` | 0.8 | 0 means greedy decoding |
| `top_p` | 0.95 | nucleus sampling mass, 1.0 disables |
| `top_k` | 0 | extension; 0 disables |
| `repetition_penalty` | 1.1 | extension; 1.0 disables |
| `stop` | none | string or list of strings; output is cut before the match |
| `seed` | none | reproducible sampling per request |
| `stream` | false | switch to Server-Sent Events |

## Streaming

Set `"stream": true`. The response is `text/event-stream` with OpenAI-shaped
chunks and a final `[DONE]`:

```
data: {"id":"cmpl-...","object":"text_completion","choices":[{"index":0,"text":" small","finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion","choices":[{"index":0,"text":"","finish_reason":"length"}]}

data: [DONE]
```

Chat streams use `object: "chat.completion.chunk"` with a `delta` field; the
first chunk carries `{"role": "assistant"}`, later ones carry `content`.

Try it with curl (the `-N` matters):

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain KV cache", "max_tokens": 60, "stream": true}'
```

## Chat template handling

If the tokenizer ships a chat template, it is applied. GPT-2 family models
have none, so their messages render as a plain transcript:

```
System: You are concise.
User: What is a KV cache?
Assistant:
```

Base GPT-2 models are not instruction-tuned, so for a chat endpoint that
actually answers questions, serve a small instruct model:

```
mini-vllm serve --model Qwen/Qwen2.5-0.5B-Instruct
```

The engine handles its ChatML template, its GQA cache layout (2 KV heads
against 14 attention heads), and EOS stopping without any model-specific
code. Verified output on the development machine, greedy:

> User: In one sentence, what does a KV cache do in an LLM server?
> Assistant: A KV cache in an LLM server stores frequently accessed data in
> memory to improve response times.

Expect about 2 to 3 tok/s for this 0.5B model on an M1 CPU in float32 (the
download is roughly 1 GB). MPS measured slower than CPU here, so the CPU
default stands.

## Errors

| Status | Meaning | Example trigger |
|---|---|---|
| 422 | Schema validation failed | missing `prompt`, `temperature: -1`, malformed JSON |
| 400 | Engine rejected the request | prompt + max_tokens exceeds the context window |
| 404 | No data yet | `/simulations/latest` before any simulate run |
| 500 | Scheduler failure mid-generation | a step raised; the message is forwarded |
| 504 | Result wait timed out | scheduler wedged longer than 300 s |

Error bodies are `{"detail": "human-readable message"}`.

## What is and is not OpenAI compatible

Supported and shape-compatible: completions, chat completions, streaming
chunks, `usage`, `finish_reason` values `stop` and `length`, `/v1/models`.

Not implemented: `n > 1`, `logprobs`, `logit_bias`, `echo`, `best_of`,
function/tool calls, vision content parts, and `model` switching per request
(one model is loaded per server; the field is informational). The official
OpenAI Python SDK works for basic calls if you point `base_url` at this
server and ignore the unsupported parameters.
