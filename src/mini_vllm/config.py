"""Engine configuration with environment variable overrides.

Every value can be overridden by a MINI_VLLM_* environment variable so the
Docker image and the CLI share one configuration path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "distilbert/distilgpt2"
TEST_MODEL = "sshleifer/tiny-gpt2"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"MINI_VLLM_{name}", default)


@dataclass
class EngineConfig:
    model_name: str = DEFAULT_MODEL
    device: str = "auto"
    dtype: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    max_batch_size: int = 8
    log_dir: str = "logs"

    @classmethod
    def from_env(cls) -> "EngineConfig":
        return cls(
            model_name=_env("MODEL", DEFAULT_MODEL),
            device=_env("DEVICE", "auto"),
            dtype=_env("DTYPE", "auto"),
            host=_env("HOST", "127.0.0.1"),
            port=int(_env("PORT", "8000")),
            max_batch_size=int(_env("MAX_BATCH_SIZE", "8")),
            log_dir=_env("LOG_DIR", "logs"),
        )
