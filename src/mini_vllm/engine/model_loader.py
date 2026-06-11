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
    quantization: str | None = None

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
            "quantization": self.quantization,
        }


def _realign_parameters(model: torch.nn.Module) -> None:
    """Copy weights out of the safetensors memory map.

    safetensors loads weights zero-copy via mmap, so a tensor's data pointer
    is whatever byte offset it has inside the checkpoint file. On macOS 26,
    Apple's Accelerate BLAS crashes with EXC_ARM_DA_ALIGN (SIGBUS) when
    cblas_sgemv reads float32 data from such unaligned addresses. Cloning
    every parameter forces fresh, properly aligned allocations. Costs one
    pass over the weights at load time; harmless everywhere else.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data.clone()
        for buffer in model.buffers():
            buffer.data = buffer.data.clone()


def _context_length(config) -> int | None:
    for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def load_model(
    name: str, device: str = "auto", dtype: str = "auto", quantize: str | None = None
) -> LoadedModel:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    if quantize not in (None, "int8"):
        raise ModelLoadError(f"Unknown quantization '{quantize}'. Supported: int8 (CPU).")
    if quantize == "int8" and resolved_device.type != "cpu":
        raise ModelLoadError("Dynamic int8 quantization is CPU only; use --device cpu.")

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
    if resolved_device.type == "cpu":
        _realign_parameters(model)
    # Count parameters before quantization: packed int8 weights are not
    # nn.Parameters, so counting afterwards would under-report.
    num_parameters = sum(p.numel() for p in model.parameters())
    if quantize == "int8":
        from mini_vllm.engine.quantize import quantize_int8

        model, _ = quantize_int8(model)

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
        num_parameters=num_parameters,
        context_length=_context_length(config),
        architecture=architectures[0],
        supports_kv_cache=supports_kv,
        quantization=quantize,
    )


def format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)
