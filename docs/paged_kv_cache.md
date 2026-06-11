# Paged KV cache, preemption, and chunked prefill

These three features turn the v1 scheduler into something much closer to how
production engines manage memory and latency. They are also exactly the
parts of vLLM that are hardest to see from the outside, which is why each
one here is observable: in tick stats, in `/metrics`, in the simulate
report, and on the dashboard.

## Why contiguous caches waste memory

The v1 backend keeps each request's keys and values in one tensor that
grows by a column per token. Batch assembly left-pads everything to the
longest live request, so memory scales with `batch_size x max_len` even
when most rows are short. Worse, growth by `torch.cat` copies the whole
tensor every step, and a long request can fail to find space even when
plenty of total memory exists in awkward pieces. This is the fragmentation
problem the vLLM paper opens with.

## The block pool

The paged backend (`engine/cache_backend.py`) allocates one fixed pool per
layer up front:

```
k_pool[layer] : [num_blocks, kv_heads, block_size, head_dim]
v_pool[layer] : [num_blocks, kv_heads, block_size, head_dim]
```

Each request owns a block table, a list of block indices, plus a token
count. Appending a token writes one column into the tail block, O(1) with
no copying. When the tail block fills, the request grabs any free block;
blocks never need to be adjacent, so fragmentation cannot happen by
construction. When a request finishes, its blocks go straight back on the
free list.

Pool shapes are discovered lazily from the first prefill, so the backend
works with any causal LM, including GQA models whose KV head count is
smaller than their attention head count.

### What is honest and what is simplified

vLLM's attention kernel reads blocks in place through the block table; the
cache is never materialized contiguously. We drive stock Hugging Face
models, which require contiguous `[batch, heads, seq, head_dim]` tensors,
so `gather()` copies each request's blocks into that shape every tick. The
memory accounting, allocation, and reclamation are real; the zero-copy
attention read is not ours without custom kernels. The gather cost is the
price of keeping the engine model-agnostic, and it is measured, not hidden:
tick stats carry pool utilization, and the parity tests prove outputs are
identical to the contiguous backend.

## Preemption

A bounded pool can fill. When an append finds no free block, the scheduler
evicts the most recently started request (the one with the least work to
lose): its blocks are freed instantly and it goes back to the FRONT of the
waiting queue. Nothing generated is lost, because on re-admission the
prompt plus all tokens generated so far are prefilled again as one
sequence. vLLM calls this recompute-mode preemption; the alternative,
swapping blocks to host memory, needs a second memory tier we do not have.

A test pins the property that matters: with a pool deliberately too small
for the workload, requests get preempted (the counter is asserted nonzero)
and every request still finishes with exactly the tokens an unpreempted
greedy run produces.

If a single request cannot fit in the entire pool even alone, it fails with
a message naming `--pool-blocks` and `--block-size` instead of looping.

## Chunked prefill

Prefill is one big forward pass over the whole prompt, and in v1 it ran
inline: a 500-token prompt froze every decoding request for the full
prefill. With `--prefill-chunk-size N`, the prompt is processed N tokens
per tick, each chunk appending its keys and values to the backend, and the
decode batch advances between chunks. A request joins decoding only when
its last chunk lands. Decode stall is bounded by the chunk, never by the
prompt.

Two tests cover it: chunked greedy output equals unchunked output, and a
short request keeps gaining tokens on every tick while a 64-token prompt
prefills in 8-token chunks beside it.

## Seeing all of it

```
uv run mini-vllm simulate examples/traffic.json \
    --cache-backend paged --block-size 16 --pool-blocks 48 \
    --prefill-chunk-size 32
```

The tick table and the dashboard's Scheduler tab show pool blocks in use
over time; squeeze `--pool-blocks` and preemptions appear in the summary.
The same flags exist on `mini-vllm serve`, and `/metrics` reports the
backend, utilization, and preemption count live.
