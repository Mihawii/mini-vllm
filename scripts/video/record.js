/* Records the demo video segments against a live mini-vLLM server.
 * Usage: node record.js <repo_root> <base_url> <out_dir>
 * Needs playwright-core installed next to it (npm i playwright-core) and a
 * Playwright Chromium build in ~/Library/Caches/ms-playwright. */

const { chromium } = require("playwright-core");
const path = require("path");
const fs = require("fs");

const [ROOT, BASE, OUT] = process.argv.slice(2);
const EXEC = `${process.env.HOME}/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`;
const SIZE = { width: 1280, height: 720 };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function still(browser, url, file, fullName) {
  const page = await browser.newPage({ viewport: SIZE });
  await page.goto(url, { waitUntil: "networkidle" });
  await sleep(500);
  await page.screenshot({ path: path.join(OUT, file) });
  await page.close();
  console.log("still:", file, fullName || "");
}

async function recorded(browser, name, fn) {
  const ctx = await browser.newContext({
    viewport: SIZE,
    recordVideo: { dir: OUT, size: SIZE },
  });
  const page = await ctx.newPage();
  try {
    await fn(page);
  } finally {
    const video = page.video();
    await ctx.close();
    const tmp = await video.path();
    fs.renameSync(tmp, path.join(OUT, `${name}.webm`));
    console.log("recorded:", `${name}.webm`);
  }
}

async function smoothScroll(page, toY, ms) {
  const steps = Math.max(1, Math.round(ms / 40));
  const from = await page.evaluate(() => window.scrollY);
  for (let i = 1; i <= steps; i++) {
    const y = from + ((toY - from) * i) / steps;
    await page.evaluate((v) => window.scrollTo(0, v), y);
    await sleep(40);
  }
}

(async () => {
  const browser = await chromium.launch({ executablePath: EXEC, headless: true });

  // ---- caption cards ----
  for (const card of ["title", "batching", "bench", "api", "closing"]) {
    await still(browser, `file://${ROOT}/scripts/video/cards/${card}.html`, `card-${card}.png`);
  }

  // ---- terminal stills from the real SVG captures ----
  await still(browser, `file://${ROOT}/docs/assets/simulate.svg`, "term-simulate.png");
  await still(browser, `file://${ROOT}/docs/assets/generate.svg`, "term-generate.png");
  await still(browser, `file://${ROOT}/docs/assets/bench-report.svg`, "term-bench.png");

  // ---- segment: playground streaming ----
  await recorded(browser, "seg-playground", async (page) => {
    await page.goto(`${BASE}/dashboard/`, { waitUntil: "networkidle" });
    await sleep(1200);
    await page.fill("#pg-prompt", "");
    await page.type("#pg-prompt", "The hardest part of building an inference engine is", { delay: 45 });
    await page.fill("#pg-max", "90");
    await page.fill("#pg-seed", "7");
    await sleep(900);
    await page.click("#pg-run");
    await page.waitForSelector("#pg-metrics .metric", { timeout: 120000 });
    await sleep(3500);
    await page.fill("#pg-prompt", "");
    await page.type("#pg-prompt", "A KV cache works by", { delay: 45 });
    await page.fill("#pg-seed", "3");
    await sleep(700);
    await page.click("#pg-run");
    await page.waitForSelector("#pg-metrics .metric", { timeout: 120000 });
    await sleep(3000);
  });

  // ---- segment: tokenizer ----
  await recorded(browser, "seg-tokenizer", async (page) => {
    await page.goto(`${BASE}/dashboard/`, { waitUntil: "networkidle" });
    await page.click('nav button[data-tab="tokenizer"]');
    await sleep(900);
    await page.fill("#tk-text", "");
    await page.type("#tk-text", "Speculative decoding verifies many draft tokens in one forward pass.", { delay: 32 });
    await sleep(600);
    await page.click("#tk-run");
    await page.waitForSelector("#tk-result table", { timeout: 30000 });
    await sleep(3800);
  });

  // ---- segment: scheduler timeline scroll ----
  await recorded(browser, "seg-scheduler", async (page) => {
    await page.goto(`${BASE}/dashboard/`, { waitUntil: "networkidle" });
    await page.click('nav button[data-tab="scheduler"]');
    await page.waitForSelector("#sc-content .card", { timeout: 30000 });
    await sleep(4200);
    await smoothScroll(page, 460, 3000);
    await sleep(3200);
    await smoothScroll(page, 900, 3000);
    await sleep(3200);
    const maxY = await page.evaluate(() => document.body.scrollHeight - window.innerHeight);
    await smoothScroll(page, maxY, 3200);
    await sleep(4000);
  });

  // ---- segment: benchmarks scroll ----
  await recorded(browser, "seg-benchmarks", async (page) => {
    await page.goto(`${BASE}/dashboard/`, { waitUntil: "networkidle" });
    await page.click('nav button[data-tab="benchmarks"]');
    await page.waitForSelector("#bm-content .card", { timeout: 30000 });
    await sleep(4200);
    await smoothScroll(page, 520, 3000);
    await sleep(3000);
    await smoothScroll(page, 1100, 3000);
    await sleep(3000);
    const maxY2 = await page.evaluate(() => document.body.scrollHeight - window.innerHeight);
    await smoothScroll(page, maxY2, 3000);
    await sleep(3200);
  });

  await browser.close();
  console.log("done");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
