/* mini-vLLM dashboard: vanilla JS, no build step.
 * Talks to the FastAPI server it is served from. */

"use strict";

const $ = (id) => document.getElementById(id);

/* ------------------------------ tabs ------------------------------ */

document.querySelectorAll("nav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    $(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "benchmarks") loadBenchmarks();
    if (btn.dataset.tab === "scheduler") loadScheduler();
  });
});

/* ------------------------------ header ------------------------------ */

async function refreshHeader() {
  try {
    const [health, metrics] = await Promise.all([
      fetch("/health").then((r) => r.json()),
      fetch("/metrics").then((r) => r.json()),
    ]);
    $("chip-model").innerHTML = `model: <b>${health.model}</b>`;
    $("chip-device").innerHTML = `device: <b>${health.device}</b>`;
    $("chip-uptime").innerHTML = `uptime: <b>${Math.round(health.uptime_s)}s</b>`;
    const sched = metrics.scheduler || {};
    $("chip-batch").innerHTML = `batch: <b>${sched.active ?? 0}/${sched.max_batch_size ?? "?"}</b>`;
  } catch {
    $("chip-model").innerHTML = `model: <b class="error-text">offline</b>`;
  }
}
refreshHeader();
setInterval(refreshHeader, 5000);

/* ------------------------------ playground ------------------------------ */

function metricChip(label, value, green = false) {
  return `<div class="metric${green ? " green" : ""}">${label}<span class="v">${value}</span></div>`;
}

function playgroundBody() {
  const seed = $("pg-seed").value;
  const stopVal = $("pg-stop").value;
  return {
    prompt: $("pg-prompt").value,
    max_tokens: parseInt($("pg-max").value, 10) || 80,
    temperature: parseFloat($("pg-temp").value),
    top_p: parseFloat($("pg-topp").value),
    top_k: parseInt($("pg-topk").value, 10) || 0,
    repetition_penalty: parseFloat($("pg-rep").value),
    ...(seed !== "" ? { seed: parseInt(seed, 10) } : {}),
    ...(stopVal ? { stop: stopVal.replaceAll("\\n", "\n") } : {}),
  };
}

async function runPlayground() {
  const btn = $("pg-run");
  const out = $("pg-output");
  const metricsEl = $("pg-metrics");
  btn.disabled = true;
  metricsEl.innerHTML = "";
  const body = playgroundBody();
  const streaming = $("pg-stream").checked;
  const started = performance.now();

  try {
    if (!streaming) {
      const res = await fetch("/v1/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      out.textContent = data.choices[0].text;
      const mv = data.mini_vllm;
      metricsEl.innerHTML =
        metricChip("latency", `${(mv.latency_ms / 1000).toFixed(2)}s`) +
        metricChip("tok/s", mv.tokens_per_second, true) +
        metricChip("tokens", data.usage.completion_tokens) +
        metricChip("ttft", mv.ttft_ms ? `${mv.ttft_ms}ms` : "n/a") +
        metricChip("finish", data.choices[0].finish_reason);
      return;
    }

    // Streaming: POST + parse Server-Sent Events off a ReadableStream.
    out.innerHTML = `<span class="cursor">▋</span>`;
    const res = await fetch("/v1/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, stream: true }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    let tokens = 0;
    let firstTokenMs = null;
    let finish = null;

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") continue;
        const chunk = JSON.parse(payload);
        if (chunk.error) throw new Error(chunk.error.message);
        const choice = chunk.choices[0];
        if (choice.text) {
          if (firstTokenMs === null) firstTokenMs = performance.now() - started;
          text += choice.text;
          tokens += 1;
          out.innerHTML = `${escapeHtml(text)}<span class="cursor">▋</span>`;
          out.scrollTop = out.scrollHeight;
        }
        if (choice.finish_reason) finish = choice.finish_reason;
      }
    }
    out.textContent = text;
    const totalS = (performance.now() - started) / 1000;
    metricsEl.innerHTML =
      metricChip("latency", `${totalS.toFixed(2)}s`) +
      metricChip("chunks/s", (tokens / totalS).toFixed(1), true) +
      metricChip("chunks", tokens) +
      metricChip("ttft", firstTokenMs ? `${firstTokenMs.toFixed(0)}ms` : "n/a") +
      metricChip("finish", finish ?? "?");
  } catch (err) {
    out.innerHTML = `<span class="error-text">${escapeHtml(err.message)}</span>`;
  } finally {
    btn.disabled = false;
  }
}
$("pg-run").addEventListener("click", runPlayground);

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* ------------------------------ tokenizer ------------------------------ */

const PALETTE = ["#22d3ee", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#fb923c"];

async function runTokenizer() {
  const res = await fetch("/tokenize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: $("tk-text").value }),
  });
  const data = await res.json();
  if (!res.ok) {
    $("tk-result").innerHTML = `<span class="error-text">${escapeHtml(JSON.stringify(data))}</span>`;
    return;
  }
  $("tk-count").innerHTML = `<b>${data.count}</b> tokens`;
  const chips = data.tokens
    .map((t, i) => {
      const c = PALETTE[i % PALETTE.length];
      return `<span class="token-chip" style="color:${c};border-color:${c}55;background:${c}14">${escapeHtml(t.text)}</span>`;
    })
    .join("");
  const rows = data.tokens
    .map(
      (t) =>
        `<tr><td class="num">${t.index}</td><td class="num">${t.token_id}</td>` +
        `<td>${escapeHtml(t.token)}</td><td>${escapeHtml(JSON.stringify(t.text))}</td></tr>`
    )
    .join("");
  $("tk-result").innerHTML = `
    <div class="tokens-flow">${chips}</div>
    <table>
      <thead><tr><th class="num">#</th><th class="num">token id</th><th>BPE piece</th><th>text</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}
$("tk-run").addEventListener("click", runTokenizer);

/* ------------------------------ SVG helpers ------------------------------ */

function barChart(items, { width = 560, barH = 26, gap = 10, color = "#22d3ee", unit = "" } = {}) {
  const max = Math.max(...items.map((d) => d.value)) || 1;
  const labelW = 150;
  const valueW = 110;
  const plotW = width - labelW - valueW;
  const height = items.length * (barH + gap);
  const bars = items
    .map((d, i) => {
      const y = i * (barH + gap);
      const w = Math.max((d.value / max) * plotW, 2);
      const c = d.color || color;
      return `
        <text x="${labelW - 8}" y="${y + barH / 2 + 4}" text-anchor="end" fill="#8b98a5" font-size="12" font-family="monospace">${d.label}</text>
        <rect x="${labelW}" y="${y}" width="${w}" height="${barH}" rx="4" fill="${c}" opacity="0.85"/>
        <text x="${labelW + w + 8}" y="${y + barH / 2 + 4}" fill="#e6edf3" font-size="12" font-weight="bold" font-family="monospace">${d.value.toFixed(1)}${unit}</text>`;
    })
    .join("");
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">${bars}</svg>`;
}

function stepArea(ticks, key, width, height, maxY, color) {
  const maxT = ticks[ticks.length - 1].t_ms || 1;
  const x = (t) => (t / maxT) * (width - 20) + 10;
  const y = (v) => height - 18 - (v / maxY) * (height - 30);
  let d = `M ${x(0)} ${y(0)}`;
  let prev = 0;
  for (const tk of ticks) {
    d += ` L ${x(tk.t_ms)} ${y(prev)} L ${x(tk.t_ms)} ${y(tk[key])}`;
    prev = tk[key];
  }
  d += ` L ${x(maxT)} ${y(prev)}`;
  const area = `${d} L ${x(maxT)} ${y(0)} L ${x(0)} ${y(0)} Z`;
  return `<path d="${area}" fill="${color}" opacity="0.18"/><path d="${d}" stroke="${color}" stroke-width="2" fill="none"/>`;
}

/* ------------------------------ benchmarks tab ------------------------------ */

async function loadBenchmarks() {
  const el = $("bm-content");
  const results = await fetch("/benchmark/results").then((r) => r.json()).catch(() => []);
  if (!results.length) {
    el.innerHTML = `<div class="card"><div class="hint">No saved benchmark results yet.<br/>Run <code>mini-vllm bench --model distilbert/distilgpt2 --compare-kv-cache</code> and reload.</div></div>`;
    return;
  }
  const r = results[0];
  const cards = [];

  cards.push(`<div class="card"><h3>Latest run</h3><table>
    <tr><td>model</td><td>${r.model}</td></tr>
    <tr><td>device / dtype</td><td>${r.device} / ${r.dtype}</td></tr>
    <tr><td>cpu</td><td>${r.machine?.cpu ?? "?"}</td></tr>
    <tr><td>timestamp</td><td>${r.timestamp}</td></tr>
    <tr><td>settings</td><td>requests=${r.settings.requests}, max_new_tokens=${r.settings.max_new_tokens}</td></tr>
  </table></div>`);

  if (r.kv_cache) {
    cards.push(`<div class="card"><h3>KV cache: throughput (tok/s)</h3>${barChart([
      { label: "with cache", value: r.kv_cache.with_kv_cache.throughput_tok_s, color: "#34d399" },
      { label: "without cache", value: r.kv_cache.without_kv_cache.throughput_tok_s, color: "#f87171" },
    ], { unit: " tok/s" })}
    <div class="legend"><span><b style="color:#34d399">${r.kv_cache.speedup}x speedup</b> from feeding one token per step instead of the whole sequence</span></div></div>`);
  }

  if (r.batch?.length) {
    cards.push(`<div class="card"><h3>Static batch throughput</h3>${barChart(
      r.batch.map((b) => ({ label: `batch ${b.batch_size}`, value: b.throughput_tok_s })),
      { unit: " tok/s" }
    )}</div>`);
  }

  if (r.concurrency) {
    cards.push(`<div class="card"><h3>Continuous batching (scheduler)</h3>${barChart([
      { label: "1 slot (sequential)", value: r.concurrency.sequential.throughput_tok_s, color: "#8b98a5" },
      { label: `${r.concurrency.batched.slots} slots (batched)`, value: r.concurrency.batched.throughput_tok_s, color: "#22d3ee" },
    ], { unit: " tok/s" })}
    <div class="legend"><span><b style="color:#22d3ee">${r.concurrency.speedup}x throughput</b> with the same model and the same requests</span></div></div>`);
  }

  if (r.latency) {
    cards.push(`<div class="card"><h3>Single-request latency</h3><table>
      <thead><tr><th>avg</th><th>p50</th><th>p95</th><th>throughput</th></tr></thead>
      <tbody><tr>
        <td>${r.latency.latency_s_avg.toFixed(2)}s</td>
        <td>${r.latency.latency_s_p50.toFixed(2)}s</td>
        <td>${r.latency.latency_s_p95.toFixed(2)}s</td>
        <td>${r.latency.throughput_tok_s.toFixed(1)} tok/s</td>
      </tr></tbody></table></div>`);
  }

  el.innerHTML = `<div class="grid">${cards.join("")}</div>`;
}

/* ------------------------------ scheduler tab ------------------------------ */

async function loadScheduler() {
  const el = $("sc-content");
  const res = await fetch("/simulations/latest");
  if (!res.ok) {
    el.innerHTML = `<div class="card"><div class="hint">No simulation recorded yet.<br/>Run <code>mini-vllm simulate examples/traffic.json</code> and reload.</div></div>`;
    return;
  }
  const sim = await res.json();
  const ticks = sim.ticks || [];
  const W = 860, H = 200;
  const maxY = Math.max(sim.max_batch_size, ...ticks.map((t) => Math.max(t.active, t.queue_depth))) + 1;

  const chart = `
    <svg class="chart" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
      ${[...Array(maxY + 1).keys()].map((v) => {
        const y = H - 18 - (v / maxY) * (H - 30);
        return `<line x1="10" x2="${W - 10}" y1="${y}" y2="${y}" stroke="#1f2937" stroke-dasharray="3 5"/>
                <text x="0" y="${y + 4}" fill="#8b98a5" font-size="10" font-family="monospace">${v}</text>`;
      }).join("")}
      ${stepArea(ticks, "active", W, H, maxY, "#22d3ee")}
      ${stepArea(ticks, "queue_depth", W, H, maxY, "#fbbf24")}
      <text x="${W - 10}" y="${H - 2}" text-anchor="end" fill="#8b98a5" font-size="11" font-family="monospace">t = ${Math.round(ticks[ticks.length - 1]?.t_ms ?? 0)} ms</text>
    </svg>
    <div class="legend">
      <span><i style="background:#22d3ee"></i>active batch size</span>
      <span><i style="background:#fbbf24"></i>queue depth</span>
    </div>`;

  const maxEnd = Math.max(...sim.requests.map((q) => q.arrive_ms + q.latency_ms)) || 1;
  const gantt = sim.requests
    .map((q, i) => {
      const y = i * 30;
      const x0 = (q.arrive_ms / maxEnd) * (W - 220) + 130;
      const wQueue = (q.queue_ms / maxEnd) * (W - 220);
      const wRun = ((q.latency_ms - q.queue_ms) / maxEnd) * (W - 220);
      return `
        <text x="122" y="${y + 16}" text-anchor="end" fill="#8b98a5" font-size="11" font-family="monospace">${q.id}</text>
        <rect x="${x0}" y="${y + 4}" width="${Math.max(wQueue, 1)}" height="16" rx="3" fill="#fbbf24" opacity="0.5"/>
        <rect x="${x0 + wQueue}" y="${y + 4}" width="${Math.max(wRun, 2)}" height="16" rx="3" fill="#22d3ee" opacity="0.85"/>
        <text x="${x0 + wQueue + wRun + 6}" y="${y + 16}" fill="#e6edf3" font-size="11" font-family="monospace">${q.tokens} tok · ${(q.latency_ms / 1000).toFixed(2)}s</text>`;
    })
    .join("");

  const rows = sim.requests
    .map(
      (q) =>
        `<tr><td>${q.id}</td><td class="num">${q.arrive_ms}</td><td class="num">${q.queue_ms}</td>` +
        `<td class="num">${q.ttft_ms}</td><td class="num">${q.latency_ms}</td><td class="num">${q.tokens}</td>` +
        `<td>${q.finish_reason}</td></tr>`
    )
    .join("");

  el.innerHTML = `
    <div class="grid">
      <div class="card"><h3>Batch occupancy over time (${sim.model}, max ${sim.max_batch_size} slots)</h3>${chart}</div>
      <div class="card"><h3>Request lifecycle (queue wait + decode)</h3>
        <svg class="chart" viewBox="0 0 ${W} ${sim.requests.length * 30}" xmlns="http://www.w3.org/2000/svg">${gantt}</svg>
        <div class="legend"><span><i style="background:#fbbf24;opacity:.5"></i>queued</span><span><i style="background:#22d3ee"></i>decoding</span></div>
      </div>
      <div class="card"><h3>Per-request metrics</h3>
        <table><thead><tr><th>id</th><th class="num">arrive ms</th><th class="num">queue ms</th><th class="num">ttft ms</th><th class="num">latency ms</th><th class="num">tokens</th><th>finish</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <div class="metrics mt">
          ${metricChip("makespan", `${(sim.makespan_ms / 1000).toFixed(2)}s`)}
          ${metricChip("throughput", `${sim.throughput_tok_s} tok/s`, true)}
          ${metricChip("avg active batch", sim.avg_active_batch)}
          ${metricChip("total tokens", sim.total_tokens)}
        </div>
      </div>
    </div>`;
}
