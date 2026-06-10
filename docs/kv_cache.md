# How the KV cache works

## The problem it solves

A decoder-only transformer predicts the next token by attending over every
earlier position. Attention at each layer needs a key vector and a value
vector for every token in the context. If you generate token 100 of a
completion and recompute keys and values for the 99 tokens before it, you
have done almost the same work 99 times already.

Without a cache, generating T tokens costs roughly O(T^2) forward work,
because step t re-processes all t-1 earlier tokens. With a cache it costs
O(T): each step processes exactly one new token and reads everything else
from memory.

## What gets cached

For every layer, the cache stores two tensors:

```
keys   : [batch, num_heads, seq_len, head_dim]
values : [batch, num_heads, seq_len, head_dim]
```

For distilgpt2 that is 6 layers x 12 heads x 64 head_dim, float32. A single
512-token context costs about 4.7 MB total. Small here, but cache size is
the real capacity limit in production serving: a 7B model with long contexts
spends gigabytes per request, which is why vLLM invented paged attention to
manage this memory in fixed-size blocks.

## Prefill and decode

Generation has two phases with very different shapes:

**Prefill** runs the whole prompt through the model in one forward pass.
All prompt positions are processed in parallel (one big matrix multiply per
layer), and each layer's keys and values are written to the cache. The
logits at the last position give the distribution for the first new token.

**Decode** runs one forward pass per generated token. The model input is a
single token id. Inside each layer, the new token's query attends over the
cached keys and values plus its own. The new key and value are appended to
the cache, the loop samples from the output logits, and the next step
repeats with that token.

This is why decode feeds only the last token: all the history the attention
needs is already in the cache. Feeding the full sequence again would compute
identical keys and values and throw them away.

The code for both phases is in `src/mini_vllm/engine/generation.py`, and the
cache plumbing (layout, batching, merging) is in
`src/mini_vllm/engine/kv_cache.py`.

## Positions and masks still matter

The model cannot infer positions from a one-token input. We pass explicit
`position_ids` derived from the attention mask, so a token after a 17-token
history gets position 17 even when the batch is left-padded. The attention
mask covers the cache length plus the new token and is how padded cache
columns (from batching requests of different lengths) stay invisible.

## Proving the cache changes nothing

The cache must never change what the model generates. Two tests in
`tests/test_generation.py` enforce that:

- `test_kv_cache_on_off_parity`: greedy decoding with and without the cache
  produces identical token ids.
- `test_native_loop_matches_hf_generate`: our cached loop matches Hugging
  Face's `model.generate()` reference output token for token.

## Measuring the win

`mini-vllm bench --compare-kv-cache` runs identical greedy workloads with the
cache on and off. On the development machine (Apple M1, CPU, float32,
distilgpt2, 64 new tokens) the cached path is several times faster, and the
gap widens with longer completions because the uncached cost grows
quadratically. Run it yourself; the suite saves JSON and CSV under
`benchmarks/results/` and `mini-vllm bench-report` renders the comparison.

You can also feel the difference interactively:

```
mini-vllm generate "Write a haiku about GPUs" --max-new-tokens 80 --kv-cache
mini-vllm generate "Write a haiku about GPUs" --max-new-tokens 80 --no-kv-cache
```

## What this implementation does not do

The cache here is one contiguous tensor per layer that grows by one column
per step. Real engines allocate cache memory in fixed-size pages so that
requests can grow without copies and freed pages can be reused immediately
(vLLM's paged attention). We trim columns that became all-padding when
requests retire (`trim_left_padding`), which is a small gesture in that
direction, but there is no page table and no block reuse. That lives in the
roadmap.
