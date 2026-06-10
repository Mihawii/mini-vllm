"""Benchmark scenarios.

Every number this module reports is measured on the machine running it.
Scenarios use greedy decoding so repeated runs (and the KV cache comparison)
do identical work: we proved in tests that cached and uncached greedy
decoding produce the same tokens, which makes their timing directly
comparable.

Scenarios:
  latency      sequential requests through the engine; per-request stats
  kv_compare   same workload with the cache on and off; reports the speedup
  batch        static batch sizes vs throughput
  concurrency  same workload through the scheduler with 1 slot vs N slots,
               which isolates the continuous-batching gain
  prompt_sweep prefill cost as the prompt grows
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import time
from pathlib import Path

from mini_vllm.engine.generation import GenerationEngine
from mini_vllm.engine.sampling import SamplingParams
from mini_vllm.engine.scheduler import Request, Scheduler
from mini_vllm.metrics.collector import percentile
from mini_vllm.metrics.storage import timestamp_slug

DEFAULT_PROMPTS = [
    "Explain what an inference engine does.",
    "Write a short product pitch for a greenhouse robot.",
    "What is the purpose of KV cache?",
    "Describe how transformers generate text one token at a time.",
    "Summarize why batching improves GPU utilization.",
]


def machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    import torch
    import transformers

    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _greedy(max_new_tokens: int) -> SamplingParams:
    return SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)


def _stats(latencies_s: list[float], tokens: list[int]) -> dict:
    ordered = sorted(latencies_s)
    total_tokens = sum(tokens)
    total_time = sum(latencies_s)
    return {
        "runs": len(latencies_s),
        "total_tokens": total_tokens,
        "latency_s_avg": round(sum(ordered) / len(ordered), 4),
        "latency_s_p50": round(percentile(ordered, 0.50), 4),
        "latency_s_p95": round(percentile(ordered, 0.95), 4),
        "throughput_tok_s": round(total_tokens / total_time, 2) if total_time else 0.0,
    }


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def run_latency(engine: GenerationEngine, requests: int, max_new_tokens: int) -> dict:
    latencies, tokens, ttfts = [], [], []
    for i in range(requests):
        prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
        result = engine.generate(prompt, _greedy(max_new_tokens))
        latencies.append(result.latency_s)
        tokens.append(result.completion_tokens)
        ttfts.append(result.prefill_s)
    out = _stats(latencies, tokens)
    out["ttft_s_p50"] = round(percentile(sorted(ttfts), 0.50), 4)
    return out


def run_kv_compare(engine: GenerationEngine, requests: int, max_new_tokens: int) -> dict:
    """The flagship comparison: identical greedy workloads, cache on vs off."""

    def measure(use_kv: bool) -> dict:
        latencies, tokens = [], []
        for i in range(requests):
            prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            result = engine.generate(prompt, _greedy(max_new_tokens), use_kv_cache=use_kv)
            latencies.append(result.latency_s)
            tokens.append(result.completion_tokens)
        return _stats(latencies, tokens)

    with_cache = measure(True)
    without_cache = measure(False)
    speedup = (
        without_cache["latency_s_avg"] / with_cache["latency_s_avg"]
        if with_cache["latency_s_avg"] > 0
        else 0.0
    )
    return {
        "with_kv_cache": with_cache,
        "without_kv_cache": without_cache,
        "speedup": round(speedup, 2),
    }


def run_batch(engine: GenerationEngine, batch_sizes: list[int], max_new_tokens: int) -> list[dict]:
    rows = []
    for size in batch_sizes:
        prompts = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(size)]
        start = time.perf_counter()
        results = engine.generate_batch(prompts, _greedy(max_new_tokens))
        elapsed = time.perf_counter() - start
        total_tokens = sum(r.completion_tokens for r in results)
        rows.append(
            {
                "batch_size": size,
                "latency_s": round(elapsed, 4),
                "total_tokens": total_tokens,
                "throughput_tok_s": round(total_tokens / elapsed, 2) if elapsed else 0.0,
            }
        )
    return rows


def run_concurrency(
    engine: GenerationEngine, requests: int, max_new_tokens: int, concurrency: int
) -> dict:
    """Same request set, scheduler with 1 slot vs N slots."""

    def measure(slots: int) -> dict:
        scheduler = Scheduler(engine, max_batch_size=slots)
        reqs = [
            Request(
                prompt=DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)],
                params=_greedy(max_new_tokens),
            )
            for i in range(requests)
        ]
        start = time.perf_counter()
        for req in reqs:
            scheduler.submit(req)
        while scheduler.has_work():
            scheduler.step()
        elapsed = time.perf_counter() - start
        total_tokens = sum(len(r.generated) for r in reqs)
        latencies = sorted(r.finished_at - r.submitted_at for r in reqs)
        return {
            "slots": slots,
            "makespan_s": round(elapsed, 4),
            "total_tokens": total_tokens,
            "throughput_tok_s": round(total_tokens / elapsed, 2) if elapsed else 0.0,
            "latency_s_p50": round(percentile(latencies, 0.50), 4),
            "latency_s_p95": round(percentile(latencies, 0.95), 4),
        }

    sequential = measure(1)
    batched = measure(concurrency)
    speedup = (
        sequential["makespan_s"] / batched["makespan_s"] if batched["makespan_s"] > 0 else 0.0
    )
    return {"sequential": sequential, "batched": batched, "speedup": round(speedup, 2)}


def run_prompt_sweep(
    engine: GenerationEngine, prompt_lengths: list[int], max_new_tokens: int
) -> list[dict]:
    base_ids = engine.tokenizer.encode(" ".join(DEFAULT_PROMPTS) * 30)
    rows = []
    for length in prompt_lengths:
        prompt = engine.tokenizer.decode(base_ids[:length])
        result = engine.generate(prompt, _greedy(max_new_tokens))
        decode_tok_s = (
            result.completion_tokens / result.decode_s if result.decode_s > 0 else 0.0
        )
        rows.append(
            {
                "prompt_tokens": result.prompt_tokens,
                "prefill_ms": round(result.prefill_s * 1000, 1),
                "decode_tok_s": round(decode_tok_s, 2),
                "latency_s": round(result.latency_s, 4),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    engine: GenerationEngine,
    requests: int = 10,
    max_new_tokens: int = 64,
    concurrency: int = 4,
    compare_kv_cache: bool = True,
    batch_sizes: list[int] | None = None,
    prompt_lengths: list[int] | None = None,
    on_progress=None,
) -> dict:
    def progress(label: str) -> None:
        if on_progress:
            on_progress(label)

    report: dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": engine.lm.name,
        "device": engine.lm.device_name,
        "dtype": str(engine.lm.dtype).removeprefix("torch."),
        "settings": {
            "requests": requests,
            "max_new_tokens": max_new_tokens,
            "concurrency": concurrency,
        },
        "machine": machine_info(),
    }

    progress("warmup")
    engine.generate(DEFAULT_PROMPTS[0], _greedy(8))  # first call pays one-off costs

    progress("single-request latency")
    report["latency"] = run_latency(engine, requests, max_new_tokens)

    if compare_kv_cache:
        progress("KV cache on/off comparison")
        kv_requests = max(min(requests, 5), 2)  # the no-cache path is slow by design
        report["kv_cache"] = run_kv_compare(engine, kv_requests, max_new_tokens)

    progress("static batch throughput")
    report["batch"] = run_batch(engine, batch_sizes or [1, 2, 4, 8], max_new_tokens)

    if concurrency > 1:
        progress("continuous batching (scheduler) comparison")
        report["concurrency"] = run_concurrency(engine, requests, max_new_tokens, concurrency)

    if prompt_lengths:
        progress("prompt length sweep")
        report["prompt_sweep"] = run_prompt_sweep(engine, prompt_lengths, max_new_tokens)

    return report


def save_results(report: dict, out_dir: str | Path = "benchmarks/results") -> tuple[Path, Path]:
    """Write the full JSON plus a flat CSV of every measured row."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = timestamp_slug()
    json_path = out_dir / f"bench-{slug}.json"
    csv_path = out_dir / f"bench-{slug}.csv"

    import json as _json

    json_path.write_text(_json.dumps(report, indent=2))

    rows: list[dict] = []

    def add(scenario: str, label: str, data: dict) -> None:
        rows.append({"scenario": scenario, "label": label, **data})

    add("latency", "sequential", report["latency"])
    if "kv_cache" in report:
        add("kv_cache", "with_cache", report["kv_cache"]["with_kv_cache"])
        add("kv_cache", "without_cache", report["kv_cache"]["without_kv_cache"])
        add("kv_cache", "speedup", {"speedup": report["kv_cache"]["speedup"]})
    for row in report.get("batch", []):
        add("batch", f"batch_{row['batch_size']}", row)
    if "concurrency" in report:
        add("concurrency", "sequential", report["concurrency"]["sequential"])
        add("concurrency", "batched", report["concurrency"]["batched"])
        add("concurrency", "speedup", {"speedup": report["concurrency"]["speedup"]})
    for row in report.get("prompt_sweep", []):
        add("prompt_sweep", f"prompt_{row['prompt_tokens']}", row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path
