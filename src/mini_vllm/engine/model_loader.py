"""Model loading.

Loads a Hugging Face causal LM plus tokenizer and records the metadata the
rest of the system needs (parameter count, context window, KV cache support).
The model is only ever used through plain forward passes; generation logic
lives in mini_vllm.engine.generation.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM

from mini_vllm.engine.tokenizer import TokenizerWrapper
from mini_vllm.utils.device import device_summary, resolve_device, resolve_dtype


class ModelLoadError(RuntimeError):
    """Raised when a model or tokenizer cannot be loaded."""


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: TokenizerWrapper
    name: str
    device: torch.device
    dtype: torch.dtype
    num_parameters: int
    context_length: int | None
    architecture: str
    supports_kv_cache: bool

    @property
    def device_name(self) -> str:
        return device_summary(self.device)

    def metadata(self) -> dict:
        return {
            "model": self.name,
            "architecture": self.architecture,
            "parameters": self.num_parameters,
            "device": self.device_name,
            "dtype": str(self.dtype).removeprefix("torch."),
            "context_length": self.context_length,
            "vocab_size": self.tokenizer.vocab_size,
            "kv_cache": self.supports_kv_cache,
        }


def _context_length(config) -> int | None:
    for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def load_model(name: str, device: str = "auto", dtype: str = "auto") -> LoadedModel:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    try:
        tokenizer = TokenizerWrapper(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=resolved_dtype)
    except Exception as exc:  # surface a readable error instead of a traceback wall
        raise ModelLoadError(
            f"Could not load model '{name}': {exc}\n"
            "Check the model id on huggingface.co, your network connection, and free disk space."
        ) from exc

    model.to(resolved_device)
    model.eval()

    config = model.config
    architectures = getattr(config, "architectures", None) or [type(model).__name__]
    # A model supports KV caching when its forward pass accepts past_key_values.
    forward_params = inspect.signature(model.forward).parameters
    supports_kv = "past_key_values" in forward_params

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        name=name,
        device=resolved_device,
        dtype=resolved_dtype,
        num_parameters=sum(p.numel() for p in model.parameters()),
        context_length=_context_length(config),
        architecture=architectures[0],
        supports_kv_cache=supports_kv,
    )


def format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)
