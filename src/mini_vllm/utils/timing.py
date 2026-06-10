"""Small timing helpers used by the engine and the benchmark runner."""

from __future__ import annotations

import time


class Timer:
    """Context manager that measures wall time with perf_counter.

    Usage:
        with Timer() as t:
            work()
        print(t.elapsed)  # seconds
    """

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start


def now_ms() -> float:
    return time.perf_counter() * 1000.0
