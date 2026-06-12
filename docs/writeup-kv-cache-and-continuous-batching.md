# How I implemented KV cache and continuous batching from scratch

I built mini-vLLM because I kept reading about how inference servers work
and realized I could not explain, at the tensor level, why generating the
100th token of a completion should cost the same as generating the 10th.
Reading vLLM's source did not fix that; it is production code, full of
concerns I did not have yet. So I wrote my own engine around stock Hugging
Face models, with the rule that the decoding loop, the cache, the batching,
and the scheduling all had to be mine. This post walks through what I
learned building it, including the two bugs that taught me the most.

The numbers below are from an Apple M1 CPU running distilgpt2 in float32.
Small hardware, but every effect that matters shows up clearly, and the
benchmark suite in the repo reproduces all of it.

## Two phases that look nothing alike

Autoregressive generation has two phases with completely different shapes.

Prefill processes the whole prompt in one forward pass. Every position is
computed in parallel, so it is one big matrix multiplication per layer and
it scales with prompt length. On my machine, prefill takes about 24 ms for
a 16-token prompt and about 100 ms for a 256-token one.

Decode then produces one token per forward pass. The model predicts token
t+1 from the last position's logits, you sample, append, and run again.
This is the loop users actually wait inside, and it is where the first big
inefficiency hides.

A transformer layer needs a key and a value vector for every context
position to compute attention. The naive loop re-feeds the entire growing
sequence each step, which means step t recomputes keys and values for all
t-1 earlier tokens and then throws them away. Generating T tokens costs
O(T^2) forward work. The fix is embarrassingly simple in concept: keep the
keys and values from earlier steps in memory, and each decode step feeds
only the newest token. Attention still sees the full history because it
reads everything else from the cache. That is the entire KV cache idea.

```
keys   : [batch, num_heads, seq_len, head_dim]   per layer
values : [batch, num_heads, seq_len, head_dim]   per layer
```

For distilgpt2 (6 layers, 12 heads, 64 head dim, fp32) a 512-token context
costs about 4.7 MB. Trivial here; for a 7B model with long contexts this
becomes gigabytes per request, which is why cache memory management is
where serving systems earn their keep.

Measured on identical greedy workloads, the cached path runs 69.5 tok/s
against 26.7 tok/s without it, a 2.6x gap at just 64 generated tokens. The
gap keeps widening with length, because one side is linear and the other is
quadratic.

## The parts that bite: positions and masks

The cached decode loop fails in quiet ways if you get two details wrong,
and nothing crashes; the model just generates subtly worse text.

First, position ids. A model cannot infer absolute positions from a
one-token input. If your cache holds 17 tokens, the new token must be told
it sits at position 17. Hugging Face models will guess `past_length` for
you in the single-sequence case, but the guess breaks the moment batching
introduces padding.

Second, padding sides. Prompts in a batch have different lengths and
tensors are rectangular. For decoder-only models the padding must go on the
left, because the next token is predicted from the last position, and all
the real text has to end at the right edge. Left padding then forces you to
derive positions from the attention mask (a cumulative sum over real
tokens), so a row whose text starts at tensor index 2 still gets position
0. I wrote both helpers once and made every path in the engine share them,
which is the only reason the later features stayed sane.

The test that kept me honest the whole way: my loop's greedy output must
match Hugging Face's `model.generate()` token for token, with the cache on,
with it off, batched and unbatched. Any clever change that broke parity was
wrong by definition, no debate. I recommend this anchor to anyone building
an engine; it converts vague worry into a binary signal.

## From static batches to a scheduler

Batching exists because a forward pass over eight sequences costs barely
more than one over a single sequence; throughput on my CPU goes from 69
tok/s at batch 1 to 298 tok/s at batch 8. The classic version is static:
collect N prompts, pad, run all of them to completion together.

Static batching has an obvious waste built in. A request that finishes at
20 tokens keeps its slot while its neighbor generates 200, and nothing new
can start until the whole batch drains. Real servers (vLLM, TGI, the Orca
paper that named the idea) schedule at iteration level instead: every model
step, the batch is rebuilt from whatever requests are alive right now. New
arrivals get prefilled and join mid-flight; finished requests leave
immediately and free their slot.

My scheduler does exactly this, one token per active request per tick. The
design decision I am happiest with came out of a refactor: requests stopped
sharing one long-lived batched tensor. Each request's cache lives on its
own in a storage backend, and every tick the scheduler asks the backend to
assemble the current batch with left padding. Assembly costs a copy per
tick, which I accepted, and in exchange requests can join and leave freely
with no tensor surgery and no stale padding. Measured against one-at-a-time
serving with the same model and the same requests, the scheduler reaches
2.1x throughput at four slots.

## Paged memory, and the bug that taught me scheduling policy

The contiguous backend grows each request's cache with `torch.cat`, which
copies the whole tensor every step and fails awkwardly when memory runs
out. The paged backend replaces that with a fixed pool of 16-token blocks
and a block table per request. Appending a token writes one column into the
tail block, O(1), and fragmentation cannot happen by construction, because
blocks never need to be adjacent. When the pool runs dry, the scheduler
preempts the most recently started request: its blocks go back to the free
list instantly, and on re-admission its prompt plus everything it already
generated is prefilled again. No output is lost, and a test asserts the
preempted run produces byte-identical results to an unpressured one.

I will be honest about the part that is not vLLM-grade: stock Hugging Face
models need contiguous tensors, so I gather blocks into one each tick
instead of reading them in place inside a custom attention kernel. The
memory management is real; the kernel is not mine. The gap is measured, not
hidden.

The instructive bug lived in the preemption policy. My first version let
an arriving request evict a running one when the pool was full. Two
requests then evicted each other forever inside a single scheduler tick:
admit A by evicting B, re-admit B by evicting A, no decode step ever ran,
and the whole engine hung. The test suite caught it as a wall-clock hang,
and the crash-course lesson is one real systems have learned before me:
admissions must back off under memory pressure and wait for capacity;
only the growth of already-running work gets to preempt. One sentence of
policy, and the difference between a scheduler and a livelock generator.

Two follow-ups landed on top of the pool. Chunked prefill bounds how long a
big prompt can stall everyone: instead of one inline forward over 500
tokens, the prompt is processed a chunk per tick while the decode batch
keeps advancing beside it. And prefix caching makes the pool
content-addressed: every full block gets a hash chained over its prefix
(`sha256(parent_hash, block_tokens)`), blocks carry reference counts, and a
new request whose prompt prefix matches cached blocks skips prefill for
them entirely. Shared blocks stay immutable because writes only ever land
in a request's private tail block, so there is no copy-on-write to get
wrong.

## Speculative decoding, where correctness is provable

The latest addition pairs two models. A small draft proposes a few tokens
cheaply; the target verifies all of them in a single forward pass and
keeps a prefix of them according to a rejection rule (accept x with
probability min(1, p(x)/q(x)); on the first rejection, resample from the
positive part of p minus q). The remarkable property, from the Leviathan
and Chen papers, is that the committed stream follows exactly the target's
distribution. Under greedy decoding that guarantee stops being
philosophical and becomes a unit test: my speculative output must equal
plain target output token for token, and it does, while the target runs
several times fewer forward passes than tokens produced.

Whether that converts to wall-clock speedup depends on the cost ratio
between draft and target, and on CPU with an 82M draft against a 124M
target the margin is thin. The benchmark reports whatever it measures,
including when speculation loses; the acceptance rate and the parity proof
are the part I actually wanted.

## The crash that came from a memory map

One war story from outside the algorithms. distilgpt2 kept killing the
process with SIGBUS while the tiny test model ran fine. The macOS crash
report pointed inside Apple's Accelerate BLAS, at `cblas_sgemv`, with the
code `EXC_ARM_DA_ALIGN`: an alignment fault. The cause took a while to
believe. safetensors loads weights zero-copy by memory-mapping the
checkpoint, so a tensor's data pointer is whatever byte offset it happens
to have inside the file, and this macOS release's vectorized sgemv kernel
faults on certain unaligned float32 reads. The tiny model dodged it only
because its shapes never took that kernel.

The fix is one pass over the weights at load time, cloning each parameter
into a fresh, properly aligned allocation. Cost: milliseconds. Lesson:
zero-copy is a contract with everyone downstream of the pointer, and BLAS
libraries did not sign it.

## Receipts

All measured on the M1, committed under `benchmarks/results/`, reproducible
with `mini-vllm bench`:

| Effect | Measurement |
|---|---|
| KV cache on vs off | 69.5 vs 26.7 tok/s (2.6x) |
| Static batch 1 to 8 | 69 to 298 tok/s |
| Continuous batching, 4 slots vs 1 | 68 to 140 tok/s (2.1x) |
| Prefill scaling, 16 to 256 prompt tokens | 24 to 100 ms, decode speed flat |
| Preemption under a 10-block pool | 14 preemptions, outputs identical, makespan +17% |
| Dynamic int8 vs fp32 | 72 vs 64 tok/s, checkpoint 328 to 239 MB, 100% token agreement |

## What real engines still do better

Paged attention kernels that read block tables in place. Mixed
prefill-and-decode batches in one forward (needs variable-length
attention). Priority classes, multi-GPU sharding, watchdogs, and the thousand
operational concerns I deliberately left out. The point of this project was
never to compete; it was to make every one of those sentences mean
something specific to me, with a test or a measurement attached.

The code is at [github.com/Mihawii/mini-vllm](https://github.com/Mihawii/mini-vllm).
The engine is about 2,500 lines of Python, the test suite runs in under
twenty seconds on a laptop, and every number in this post comes out of one
command.
