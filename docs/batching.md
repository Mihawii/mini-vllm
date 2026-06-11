# How batching works

## Why batch at all

A forward pass over one token and a forward pass over eight tokens cost
nearly the same wall time on most hardware, because matrix multiply
throughput, memory bandwidth, and kernel launch overhead dominate. Serving
requests one at a time leaves that capacity idle. Batching fills it: the
benchmark on this repo's development machine shows distilgpt2 going from 69
tok/s at batch size 1 to 301 tok/s at batch size 8 on a plain M1 CPU.

## Left padding, and why position ids need care

Prompts in a batch have different lengths, and tensors are rectangular, so
shorter prompts get padded. For decoder-only models the padding must go on
the LEFT: the next token is predicted from the last sequence position, so
every row's real text has to end at the right edge.

```
pad pad The cat sat
Once upon a   time ,
```

Two consequences follow. The attention mask marks pad positions so attention
ignores them. And position ids must be computed from that mask (a cumulative
sum over real tokens), because row one's first real token is at tensor index
2 but must still get position 0. Both helpers live in
`src/mini_vllm/engine/batching.py` and are shared by every batched code path.

## Static batching

`GenerationEngine.generate_batch` implements the classic version: collect N
prompts, pad them, prefill them as one batch, then decode all rows in
lockstep. Rows finish independently (EOS, stop string, or their token
budget); finished rows ride along with pad tokens while their outputs stay
frozen, until every row is done.

A correctness test pins the semantics: each row of a greedy batch must equal
the same prompt generated alone (`test_batch_matches_single_greedy`).

The weakness shows up with mixed lengths. A request that needs 5 tokens
keeps its slot while a neighbor generates 200, and nothing new can start
until the whole batch drains. Throughput is good; latency under real,
staggered traffic is poor.

## Continuous batching

Real inference servers (vLLM, TGI, and the Orca paper that named the idea)
schedule at iteration level instead. Every model step, the batch is rebuilt
from whatever requests are alive:

1. Admit waiting requests while slots are free.
2. Prefill each newcomer individually and merge its KV cache into the live
   batch.
3. Run ONE batched decode step; every active request gains one token.
4. Retire finished requests immediately; their slots are free for the next
   tick.

`src/mini_vllm/engine/scheduler.py` implements this on top of the same
engine primitives. The interesting mechanics:

**Per-request cache storage.** Each request's keys and values live in a
cache backend (contiguous tensors, or the paged block pool described in
[paged_kv_cache.md](paged_kv_cache.md)). Every tick the scheduler asks the
backend to assemble the live batch: per-request caches are left-padded with
zero K/V columns to a common length and stacked. The zero columns are safe
because the attention mask excludes them from the softmax; a masked
position is never read.

**Retirement.** When a request finishes, its storage is freed immediately
(with the paged backend, its blocks go straight back on the free list).
Because the batch is reassembled from scratch each tick, there is no
long-lived batched tensor to keep in sync and no stale padding to carry.

**Threading.** `submit()` is safe from any thread; `step()` runs only on the
model thread. The HTTP server wraps the scheduler in a daemon thread
(`SchedulerLoop`) and bridges results back through per-request queues, so
two curl requests arriving together genuinely share batches.

**Chunked prefill.** With `--prefill-chunk-size N`, a long prompt is
processed N tokens per tick instead of in one inline forward pass, so
decoding requests stall for at most one chunk. Details and tests are in
[paged_kv_cache.md](paged_kv_cache.md).

**Preemption.** When the paged pool runs out of blocks, the most recently
started request is evicted and requeued; its prompt plus the tokens it
already generated are prefilled again on re-admission, so no output is
lost. The simulate summary and `/metrics` count preemptions.

## Seeing it

`mini-vllm simulate examples/traffic.json` replays a traffic file with
wall-clock arrivals and prints the tick timeline: the batch grows as
requests arrive, a queue forms past `--max-batch-size`, short requests
retire while long ones keep decoding, and the cache length shrinks at
retirement. The same data lands in `logs/simulations/` as JSONL and feeds
the dashboard's Scheduler tab.

The benchmark quantifies the gain: the same eight requests pushed through
the scheduler with one slot versus four slots ran 2.3x faster end to end on
the development machine (`mini-vllm bench`, "concurrency" scenario).

## What real engines still add beyond this

The paged backend manages memory the way vLLM does, but vLLM's attention
kernel reads blocks in place; we gather them into contiguous tensors each
tick because stock Hugging Face models require that layout. There are also
no priority classes, no prefill/decode disaggregation across processes, and
no custom kernels anywhere. These are deliberate simplifications; the point
here is to make the core ideas readable and measurable.
