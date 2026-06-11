"""Render a saved benchmark JSON into a Markdown report."""

from __future__ import annotations

import json
from pathlib import Path

from mini_vllm.metrics.storage import latest_file


def _table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    machine = report.get("machine", {})
    settings = report.get("settings", {})
    parts = [
        "# mini-vLLM benchmark report",
        "",
        f"Generated: {report.get('timestamp', '?')}",
        "",
        "## Setup",
        "",
        _table(
            ["item", "value"],
            [
                ["model", report.get("model")],
                ["device", report.get("device")],
                ["dtype", report.get("dtype")],
                ["cpu", machine.get("cpu")],
                ["platform", machine.get("platform")],
                ["python / torch", f"{machine.get('python')} / {machine.get('torch')}"],
                ["requests", settings.get("requests")],
                ["max_new_tokens", settings.get("max_new_tokens")],
            ],
        ),
        "",
    ]

    latency = report.get("latency")
    if latency:
        parts += [
            "## Single-request latency (sequential, greedy)",
            "",
            _table(
                ["runs", "avg latency", "p50", "p95", "throughput"],
                [[
                    latency["runs"],
                    f"{latency['latency_s_avg']:.2f}s",
                    f"{latency['latency_s_p50']:.2f}s",
                    f"{latency['latency_s_p95']:.2f}s",
                    f"{latency['throughput_tok_s']:.1f} tok/s",
                ]],
            ),
            "",
        ]

    kv = report.get("kv_cache")
    if kv:
        parts += [
            "## KV cache: on vs off",
            "",
            "Identical greedy workloads; the only difference is whether each decode step",
            "feeds one token (cache on) or re-runs the whole sequence (cache off).",
            "",
            _table(
                ["mode", "avg latency", "throughput"],
                [
                    [
                        "with KV cache",
                        f"{kv['with_kv_cache']['latency_s_avg']:.2f}s",
                        f"{kv['with_kv_cache']['throughput_tok_s']:.1f} tok/s",
                    ],
                    [
                        "without KV cache",
                        f"{kv['without_kv_cache']['latency_s_avg']:.2f}s",
                        f"{kv['without_kv_cache']['throughput_tok_s']:.1f} tok/s",
                    ],
                ],
            ),
            "",
            f"**Speedup: {kv['speedup']:.2f}x**",
            "",
        ]

    batch = report.get("batch")
    if batch:
        parts += [
            "## Static batch throughput",
            "",
            _table(
                ["batch size", "latency", "total tokens", "throughput"],
                [
                    [
                        row["batch_size"],
                        f"{row['latency_s']:.2f}s",
                        row["total_tokens"],
                        f"{row['throughput_tok_s']:.1f} tok/s",
                    ]
                    for row in batch
                ],
            ),
            "",
        ]

    conc = report.get("concurrency")
    if conc:
        parts += [
            "## Continuous batching (scheduler)",
            "",
            "The same request set submitted to the scheduler with one slot",
            "(sequential) and with N slots (batched decode).",
            "",
            _table(
                ["mode", "makespan", "throughput", "p50 latency", "p95 latency"],
                [
                    [
                        "1 slot (sequential)",
                        f"{conc['sequential']['makespan_s']:.2f}s",
                        f"{conc['sequential']['throughput_tok_s']:.1f} tok/s",
                        f"{conc['sequential']['latency_s_p50']:.2f}s",
                        f"{conc['sequential']['latency_s_p95']:.2f}s",
                    ],
                    [
                        f"{conc['batched']['slots']} slots (batched)",
                        f"{conc['batched']['makespan_s']:.2f}s",
                        f"{conc['batched']['throughput_tok_s']:.1f} tok/s",
                        f"{conc['batched']['latency_s_p50']:.2f}s",
                        f"{conc['batched']['latency_s_p95']:.2f}s",
                    ],
                ],
            ),
            "",
            f"**Throughput gain: {conc['speedup']:.2f}x**",
            "",
        ]

    sweep = report.get("prompt_sweep")
    if sweep:
        parts += [
            "## Prompt length sweep",
            "",
            "Prefill cost grows with prompt size; decode speed stays roughly flat",
            "because each cached step only processes one token.",
            "",
            _table(
                ["prompt tokens", "prefill", "decode speed", "total latency"],
                [
                    [
                        row["prompt_tokens"],
                        f"{row['prefill_ms']:.0f} ms",
                        f"{row['decode_tok_s']:.1f} tok/s",
                        f"{row['latency_s']:.2f}s",
                    ]
                    for row in sweep
                ],
            ),
            "",
        ]

    quant = report.get("quantization")
    if quant:
        parts += [
            "## Dynamic int8 quantization (CPU)",
            "",
            "Same greedy workload, fp32 vs dynamically quantized int8 Linear layers.",
            "Token agreement is the fraction of positions where both variants chose",
            "the same token; quantization is lossy and this makes the loss visible.",
            "",
            _table(
                ["variant", "avg latency", "throughput", "checkpoint"],
                [
                    [
                        "float32",
                        f"{quant['float32']['latency_s_avg']:.2f}s",
                        f"{quant['float32']['throughput_tok_s']:.1f} tok/s",
                        f"{quant['float32']['checkpoint_mb']:.1f} MB",
                    ],
                    [
                        "int8 dynamic",
                        f"{quant['int8']['latency_s_avg']:.2f}s",
                        f"{quant['int8']['throughput_tok_s']:.1f} tok/s",
                        f"{quant['int8']['checkpoint_mb']:.1f} MB",
                    ],
                ],
            ),
            "",
            f"**Latency ratio: {quant['speedup']:.2f}x; greedy token agreement: {quant['token_agreement']:.1%}**",
            "",
        ]

    parts += [
        "## Conclusions",
        "",
    ]
    if kv:
        parts.append(
            f"- The KV cache delivered a {kv['speedup']:.2f}x latency improvement on this workload."
        )
    if conc:
        parts.append(
            f"- Continuous batching raised throughput {conc['speedup']:.2f}x over sequential serving."
        )
    if batch:
        best = max(batch, key=lambda r: r["throughput_tok_s"])
        parts.append(
            f"- Static batching peaked at {best['throughput_tok_s']:.1f} tok/s with batch size {best['batch_size']}."
        )
    if quant:
        parts.append(
            f"- int8 cut the checkpoint from {quant['float32']['checkpoint_mb']:.0f} MB to "
            f"{quant['int8']['checkpoint_mb']:.0f} MB with {quant['token_agreement']:.0%} greedy token agreement."
        )
    parts.append("- Numbers are from a real run on the machine listed above; rerun `mini-vllm bench` to reproduce on yours.")
    return "\n".join(parts) + "\n"


def latest_report(results_dir: str | Path = "benchmarks/results") -> dict | None:
    path = latest_file(results_dir, "bench-*.json")
    if path is None:
        return None
    return json.loads(path.read_text())
