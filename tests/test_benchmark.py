"""Benchmark plumbing: tiny settings, real measurements, files on disk."""

import csv
import json


def test_benchmark_creates_files(engine, tmp_path):
    from mini_vllm.benchmark.runner import run_benchmark, save_results

    report = run_benchmark(
        engine,
        requests=2,
        max_new_tokens=4,
        concurrency=2,
        compare_kv_cache=True,
        batch_sizes=[1, 2],
    )
    json_path, csv_path = save_results(report, tmp_path)

    assert json_path.exists() and csv_path.exists()
    saved = json.loads(json_path.read_text())
    assert saved["model"] == engine.lm.name
    assert saved["latency"]["runs"] == 2
    assert saved["latency"]["throughput_tok_s"] > 0
    assert saved["kv_cache"]["speedup"] > 0
    assert saved["concurrency"]["speedup"] > 0
    assert {row["batch_size"] for row in saved["batch"]} == {1, 2}
    assert saved["machine"]["cpu"]

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    scenarios = {row["scenario"] for row in rows}
    assert {"latency", "kv_cache", "batch", "concurrency"} <= scenarios


def test_report_renders_markdown(engine, tmp_path):
    from mini_vllm.benchmark.report import latest_report, render_markdown
    from mini_vllm.benchmark.runner import run_benchmark, save_results

    report = run_benchmark(
        engine, requests=2, max_new_tokens=4, concurrency=2,
        compare_kv_cache=False, batch_sizes=[1],
    )
    save_results(report, tmp_path)
    loaded = latest_report(tmp_path)
    assert loaded is not None
    markdown = render_markdown(loaded)
    assert "# mini-vLLM benchmark report" in markdown
    assert "Single-request latency" in markdown
    assert "tok/s" in markdown


def test_prompt_sweep_rows(engine):
    from mini_vllm.benchmark.runner import run_prompt_sweep

    rows = run_prompt_sweep(engine, [8, 16], max_new_tokens=4)
    assert [row["prompt_tokens"] for row in rows] == [8, 16]
    assert all(row["prefill_ms"] >= 0 for row in rows)
