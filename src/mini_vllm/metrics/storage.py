"""JSONL persistence for request logs, simulations, and benchmark runs."""

from __future__ import annotations

import json
import time
from pathlib import Path


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_file(directory: str | Path, pattern: str) -> Path | None:
    directory = Path(directory)
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern))
    return candidates[-1] if candidates else None


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
