"""Device and dtype resolution.

mini-vLLM is CPU-first. CUDA is selected automatically when present because it
is strictly faster for these models. MPS (Apple Silicon) is opt-in only: for
the small models this project targets, kernel dispatch overhead usually makes
MPS slower than plain CPU, so it is never auto-selected.
"""

from __future__ import annotations

import torch


class DeviceError(ValueError):
    """Raised when a device or dtype request cannot be satisfied."""


def resolve_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Run with --device cpu, or install a CUDA build of PyTorch."
            )
        return torch.device("cuda")
    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceError("MPS was requested but is not available on this machine.")
        return torch.device("mps")
    raise DeviceError(f"Unknown device '{spec}'. Expected one of: auto, cpu, cuda, mps.")


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(spec: str = "auto", device: torch.device | None = None) -> torch.dtype:
    spec = (spec or "auto").lower()
    if spec == "auto":
        # float16 halves memory and roughly doubles throughput on CUDA.
        # On CPU, float32 is both faster and numerically safer.
        if device is not None and device.type == "cuda":
            return torch.float16
        return torch.float32
    if spec not in _DTYPES:
        raise DeviceError(f"Unknown dtype '{spec}'. Expected one of: auto, float32, float16, bfloat16.")
    dtype = _DTYPES[spec]
    if dtype is torch.float16 and (device is None or device.type == "cpu"):
        raise DeviceError(
            "float16 on CPU is slow and numerically fragile. Use float32 (default) or bfloat16."
        )
    return dtype


def device_summary(device: torch.device) -> str:
    """Human-readable device line for the inspect command and /metrics."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type
