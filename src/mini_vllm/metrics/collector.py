"""In-process metrics.

A single MetricsCollector instance lives next to the scheduler and is updated
through the scheduler's event callback. Everything is guarded by one lock;
contention is negligible at the request rates this engine handles.
"""

from __future__ import annotations

import threading
import time
from collections import deque


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile; good enough for serving dashboards."""
    if not sorted_values:
        return 0.0
    idx = min(int(round(q * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return sorted_values[idx]


class MetricsCollector:
    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self._started_wall = time.time()
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.active_requests = 0
        self.queued_requests = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        # rolling windows so percentiles reflect recent behavior
        self._latencies_s: deque[float] = deque(maxlen=window)
        self._ttfts_s: deque[float] = deque(maxlen=window)

    # ------------------------------------------------------------------
    # scheduler event hook: collector.on_event is passed to Scheduler
    # ------------------------------------------------------------------

    def on_event(self, event: str, request) -> None:
        with self._lock:
            if event == "submitted":
                self.total_requests += 1
                self.queued_requests += 1
            elif event == "started":
                self.queued_requests -= 1
                self.active_requests += 1
            elif event == "finished":
                self.active_requests -= 1
                self.completed_requests += 1
                self.prompt_tokens += len(request.prompt_ids)
                self.generated_tokens += len(request.generated)
                self._latencies_s.append(max(request.finished_at - request.submitted_at, 0.0))
                if request.ttft_s is not None:
                    self._ttfts_s.append(request.ttft_s)
            elif event == "failed":
                # a request can fail from WAITING (validation) or RUNNING
                if request.started_at:
                    self.active_requests = max(self.active_requests - 1, 0)
                else:
                    self.queued_requests = max(self.queued_requests - 1, 0)
                self.failed_requests += 1
                self.total_requests = max(self.total_requests, self.failed_requests)

    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            uptime = time.time() - self._started_wall
            latencies = sorted(self._latencies_s)
            ttfts = sorted(self._ttfts_s)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            return {
                "uptime_s": round(uptime, 1),
                "requests": {
                    "total": self.total_requests,
                    "queued": self.queued_requests,
                    "active": self.active_requests,
                    "completed": self.completed_requests,
                    "failed": self.failed_requests,
                },
                "tokens": {
                    "prompt": self.prompt_tokens,
                    "generated": self.generated_tokens,
                    "per_second_lifetime": round(self.generated_tokens / uptime, 2) if uptime > 0 else 0.0,
                },
                "latency_s": {
                    "avg": round(avg_latency, 4),
                    "p50": round(percentile(latencies, 0.50), 4),
                    "p95": round(percentile(latencies, 0.95), 4),
                },
                "ttft_s": {
                    "p50": round(percentile(ttfts, 0.50), 4),
                    "p95": round(percentile(ttfts, 0.95), 4),
                },
            }

    def prometheus(self, prefix: str = "mini_vllm") -> str:
        """Hand-rolled Prometheus text exposition (no client dependency)."""
        snap = self.snapshot()
        lines = []

        def gauge(name: str, value, help_text: str) -> None:
            lines.append(f"# HELP {prefix}_{name} {help_text}")
            lines.append(f"# TYPE {prefix}_{name} gauge")
            lines.append(f"{prefix}_{name} {value}")

        gauge("uptime_seconds", snap["uptime_s"], "Server uptime")
        gauge("requests_total", snap["requests"]["total"], "Requests received")
        gauge("requests_active", snap["requests"]["active"], "Requests decoding right now")
        gauge("requests_queued", snap["requests"]["queued"], "Requests waiting for a batch slot")
        gauge("requests_completed", snap["requests"]["completed"], "Requests finished")
        gauge("requests_failed", snap["requests"]["failed"], "Requests failed")
        gauge("prompt_tokens_total", snap["tokens"]["prompt"], "Prompt tokens processed")
        gauge("generated_tokens_total", snap["tokens"]["generated"], "Tokens generated")
        gauge("latency_seconds_p50", snap["latency_s"]["p50"], "Median request latency")
        gauge("latency_seconds_p95", snap["latency_s"]["p95"], "p95 request latency")
        gauge("ttft_seconds_p50", snap["ttft_s"]["p50"], "Median time to first token")
        return "\n".join(lines) + "\n"
