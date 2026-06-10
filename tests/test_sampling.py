import math

import pytest
import torch

from mini_vllm.engine.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    filter_top_k,
    filter_top_p,
    make_generator,
    sample_token,
)


def test_top_k_keeps_exactly_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    out = filter_top_k(logits, 2)
    kept = torch.isfinite(out[0])
    assert kept.sum() == 2
    assert kept[1] and kept[4]  # the two largest logits survive


def test_top_k_zero_disables():
    logits = torch.randn(2, 10)
    assert torch.equal(filter_top_k(logits, 0), logits)


def test_top_p_keeps_minimal_nucleus():
    # probs ~ [0.643, 0.237, 0.087, 0.032] for logits [4,3,2,1]
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    out = filter_top_p(logits, 0.7)
    kept = torch.isfinite(out[0])
    # 0.643 < 0.7, so the second token is needed to reach the mass; rest drop.
    assert kept.tolist() == [True, True, False, False]


def test_top_p_always_keeps_best_token():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    out = filter_top_p(logits, 0.01)
    assert torch.isfinite(out[0][0])
    assert torch.isinf(out[0][1:]).all()


def test_top_p_one_disables():
    logits = torch.randn(3, 7)
    assert torch.equal(filter_top_p(logits, 1.0), logits)


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    seen = [torch.tensor([0, 1])]
    out = apply_repetition_penalty(logits, seen, penalty=2.0)
    assert math.isclose(out[0, 0].item(), 1.0)  # positive logit divided
    assert math.isclose(out[0, 1].item(), -2.0)  # negative logit multiplied
    assert math.isclose(out[0, 2].item(), 0.5)  # unseen token untouched


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 9.0, 0.2], [3.0, 0.0, 0.1]])
    params = SamplingParams(temperature=0.0)
    tokens = sample_token(logits, params)
    assert tokens.tolist() == [1, 0]


def test_seeded_sampling_is_deterministic():
    logits = torch.randn(1, 100)
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=42)
    draws = [
        int(sample_token(logits.clone(), params, generator=make_generator(42, torch.device("cpu"))))
        for _ in range(3)
    ]
    assert len(set(draws)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"temperature": -0.5},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"repetition_penalty": 0.0},
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
