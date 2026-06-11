"""Tensor parallelism, demonstrated numerically on one CPU.

Real tensor parallelism splits a model's weight matrices across GPUs so each
device holds and multiplies a shard, with collective ops (all-reduce) gluing
the math back together. One laptop has no second device, so this script does
the only honest thing available: it runs the sharded math SEQUENTIALLY on
two simulated ranks and proves the result is identical to the unsharded
forward pass. No speedup is claimed; the point is the arithmetic.

The sharding follows Megatron-LM:

  Attention   QKV projection is COLUMN-parallel (each rank owns a subset of
              heads end to end), the output projection is ROW-parallel
              (each rank multiplies its slice, partial results are summed).
  MLP         The up projection is COLUMN-parallel (each rank owns a slice
              of the hidden dimension), the down projection is ROW-parallel.

The "all-reduce" here is a literal torch sum over the per-rank partials.
Per transformer block, that is exactly two all-reduces, same as Megatron.

Run it:
    uv run python experiments/tensor_parallel.py
"""

from __future__ import annotations

import math

import torch

torch.manual_seed(0)

RANKS = 2
D_MODEL = 64
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS
D_FF = 4 * D_MODEL
SEQ = 10


def attention(x, w_qkv, w_out, n_heads):
    """Plain causal self-attention, written out so the sharded version below
    can be compared line by line."""
    seq, d_model = x.shape
    qkv = x @ w_qkv  # [seq, 3*d]
    q, k, v = qkv.split(d_model, dim=-1)
    d_head = d_model // n_heads
    q = q.view(seq, n_heads, d_head).transpose(0, 1)  # [heads, seq, d_head]
    k = k.view(seq, n_heads, d_head).transpose(0, 1)
    v = v.view(seq, n_heads, d_head).transpose(0, 1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(d_head)
    causal = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
    attn = torch.softmax(scores + causal, dim=-1)
    out = (attn @ v).transpose(0, 1).reshape(seq, d_model)
    return out @ w_out


def mlp(x, w_up, w_down):
    return torch.nn.functional.gelu(x @ w_up) @ w_down


def reference_block(x, weights):
    return attention(x, weights["w_qkv"], weights["w_out"], N_HEADS) + mlp(
        x, weights["w_up"], weights["w_down"]
    )


def shard_weights(weights, ranks: int):
    """Cut the matrices the Megatron way. Column-parallel weights are split
    on their OUTPUT dim, row-parallel weights on their INPUT dim."""
    heads_per_rank = N_HEADS // ranks
    shards = []
    for r in range(ranks):
        # QKV column shard: rank r owns heads [r*hpr, (r+1)*hpr) of Q, K, V.
        cols = []
        for block_idx in range(3):  # q, k, v live side by side in w_qkv
            start = block_idx * D_MODEL + r * heads_per_rank * D_HEAD
            cols.append(weights["w_qkv"][:, start : start + heads_per_rank * D_HEAD])
        w_qkv_r = torch.cat(cols, dim=1)  # [d_model, 3 * d_model/ranks]
        # Output row shard: rank r's attention output covers those head dims.
        row0 = r * heads_per_rank * D_HEAD
        w_out_r = weights["w_out"][row0 : row0 + heads_per_rank * D_HEAD, :]
        # MLP: up is column-parallel, down is row-parallel.
        ff0 = r * (D_FF // ranks)
        w_up_r = weights["w_up"][:, ff0 : ff0 + D_FF // ranks]
        w_down_r = weights["w_down"][ff0 : ff0 + D_FF // ranks, :]
        shards.append({"w_qkv": w_qkv_r, "w_out": w_out_r, "w_up": w_up_r, "w_down": w_down_r})
    return shards


def rank_forward(x, shard, ranks: int):
    """What ONE rank computes locally: its heads' attention and its MLP
    slice, each ending in a PARTIAL result that needs an all-reduce."""
    seq = x.shape[0]
    heads_per_rank = N_HEADS // ranks
    local_d = heads_per_rank * D_HEAD

    qkv = x @ shard["w_qkv"]  # [seq, 3 * local_d]
    q, k, v = qkv.split(local_d, dim=-1)
    q = q.view(seq, heads_per_rank, D_HEAD).transpose(0, 1)
    k = k.view(seq, heads_per_rank, D_HEAD).transpose(0, 1)
    v = v.view(seq, heads_per_rank, D_HEAD).transpose(0, 1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(D_HEAD)
    causal = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
    attn = torch.softmax(scores + causal, dim=-1)
    local_attn = (attn @ v).transpose(0, 1).reshape(seq, local_d)
    attn_partial = local_attn @ shard["w_out"]  # needs all-reduce

    mlp_partial = torch.nn.functional.gelu(x @ shard["w_up"]) @ shard["w_down"]  # needs all-reduce
    return attn_partial, mlp_partial


def tensor_parallel_block(x, weights, ranks: int):
    shards = shard_weights(weights, ranks)
    attn_partials, mlp_partials = [], []
    for shard in shards:  # sequential stand-in for "each device computes"
        attn_p, mlp_p = rank_forward(x, shard, ranks)
        attn_partials.append(attn_p)
        mlp_partials.append(mlp_p)
    # The two all-reduces. On real hardware this is NCCL; here it is a sum.
    return torch.stack(attn_partials).sum(0) + torch.stack(mlp_partials).sum(0)


def main() -> bool:
    weights = {
        "w_qkv": torch.randn(D_MODEL, 3 * D_MODEL) / math.sqrt(D_MODEL),
        "w_out": torch.randn(D_MODEL, D_MODEL) / math.sqrt(D_MODEL),
        "w_up": torch.randn(D_MODEL, D_FF) / math.sqrt(D_MODEL),
        "w_down": torch.randn(D_FF, D_MODEL) / math.sqrt(D_FF),
    }
    x = torch.randn(SEQ, D_MODEL)

    ref = reference_block(x, weights)
    tp = tensor_parallel_block(x, weights, RANKS)
    ok = torch.allclose(ref, tp, atol=1e-5)

    total = sum(w.numel() for w in weights.values())
    per_rank = sum(w.numel() for w in shard_weights(weights, RANKS)[0].values())
    print(f"d_model={D_MODEL} heads={N_HEADS} d_ff={D_FF} seq={SEQ} ranks={RANKS}")
    print(f"parameters: {total:,} total, {per_rank:,} per rank ({per_rank / total:.0%})")
    print("all-reduces per block: 2 (attention out, MLP out)")
    print(f"max |reference - tensor_parallel| = {(ref - tp).abs().max().item():.2e}")
    print("MATCH" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
