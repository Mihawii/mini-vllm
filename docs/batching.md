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

**Cache merge.** The newcomer's cache covers its prompt; the live batch's
cache covers longer histories. The shorter side is left-padded with zero
keys and values along the sequence dimension, then the two stack along the
batch dimension. Zero K/V columns are safe because the attention mask
excludes them from the softmax; a masked position is never read.

**Retirement.** When a request finishes, its row is dropped from the cache
(`select_rows`) and columns that became all-padding are trimmed
(`trim_left_padding`), so memory tracks the longest live request instead of
the longest request ever seen.

**Threading.** `submit()` is safe from any thread; `step()` runs only on the
model thread. The HTTP server wraps the scheduler in a daemon thread
(`SchedulerLoop`) and bridges results back through per-request queues, so
two curl requests arriving together genuinely share batches.

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

## What real engines add beyond this

This scheduler keeps one contiguous cache tensor per batch and re-pads when
shapes change, which costs copies that vLLM avoids with paged attention and
block tables. There is also no preemption, no priority, and no
prefill/decode disaggregation: prefills run inline on the model thread, so
one giant prompt briefly stalls everyone (the simulate timeline makes this
visible). These are deliberate simplifications; the point here is to make
the core idea readable.
