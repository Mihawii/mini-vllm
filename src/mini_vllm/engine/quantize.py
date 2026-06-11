"""Dynamic int8 quantization experiment (CPU only).

Dynamic quantization stores weights as int8 (4x smaller than float32) and
quantizes activations on the fly inside each matmul. PyTorch ships this for
nn.Linear, but GPT-2 family models hide their projections inside
transformers' Conv1D, a Linear with a transposed weight that the quantizer
does not recognize. So the conversion runs in two steps:

1. Replace every Conv1D with a numerically identical nn.Linear
   (weight transposed, bias copied). A test asserts greedy output is
   unchanged after this step.
2. Apply torch.ao.quantization.quantize_dynamic to all Linear layers.

Step 2 is lossy by design; the benchmark reports how much (token agreement
against fp32 greedy output) next to the memory and speed numbers, so the
tradeoff is visible instead of implied.

Implementation notes: recent torch builds ship with no quantized engine
selected (`torch.backends.quantized.engine == "none"`), so we pick the
first supported one explicitly. torch.ao.quantization is deprecated in
favor of the separate torchao package; it still works across torch 2.x and
keeps this experiment dependency-free, so we use it knowingly.
"""

from __future__ import annotations

import io
import warnings

import torch
from torch import nn


class QuantizationUnsupported(RuntimeError):
    """Raised when this torch build ships no quantized engine."""


def _select_engine() -> str:
    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not engines:
        raise QuantizationUnsupported(
            "This torch build has no quantized engine (qnnpack/fbgemm); "
            "dynamic int8 is unavailable on this platform."
        )
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = engines[0]
    return torch.backends.quantized.engine


def convert_conv1d_to_linear(model: nn.Module) -> int:
    """Swap transformers Conv1D modules for equivalent nn.Linear. Returns
    the number of modules converted."""
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:  # transformers moved it; nothing to convert
        return 0

    converted = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, Conv1D):
                # Conv1D stores weight as [in_features, out_features].
                in_features, out_features = child.weight.shape
                linear = nn.Linear(in_features, out_features, bias=child.bias is not None)
                with torch.no_grad():
                    linear.weight.copy_(child.weight.t())
                    if child.bias is not None:
                        linear.bias.copy_(child.bias)
                setattr(parent, name, linear)
                converted += 1
    return converted


def quantize_int8(model: nn.Module) -> tuple[nn.Module, dict]:
    """Convert Conv1D to Linear, then dynamically quantize every Linear.

    Returns the quantized model and an info dict (modules converted and
    quantized). CPU only; the caller enforces the device.
    """
    engine = _select_engine()
    converted = convert_conv1d_to_linear(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    n_quantized = sum(
        1 for m in quantized.modules()
        if type(m).__module__.startswith("torch.ao.nn.quantized")
    )
    return quantized, {
        "engine": engine,
        "conv1d_converted": converted,
        "linear_quantized": n_quantized,
    }


def state_dict_bytes(model: nn.Module) -> int:
    """Serialized checkpoint size. Works for quantized models too, where
    packed int8 weights are not ordinary parameters."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes
