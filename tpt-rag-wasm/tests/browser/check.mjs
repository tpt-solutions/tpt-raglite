// Drives examples/wasm_browser in a real browser (Chromium through Playwright): indexes a
// document, searches it by meaning and by keyword, and checks that the saved database is still
// there after a reload. The browser downloads the embedding model on first run (about 23 MB).
//
//   npm install --no-save playwright@1 && npx playwright install chromium   (in this folder)
//   wasm-pack build --target web ../..                                       (in the repo root)
//   node check.mjs
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import { chromium } from "playwright";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const types = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
};

const server = createServer(async (request, response) => {
  const path = normalize(decodeURIComponent(new URL(request.url, "http://x").pathname));
  const file = join(root, path === "/" ? "index.html" : path);
  try {
    if (!(await stat(file)).isFile()) throw new Error("not a file");
    response.writeHead(200, { "Content-Type": types[extname(file)] ?? "application/octet-stream" });
    response.end(await readFile(file));
  } catch {
    response.writeHead(404).end("not found");
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const url = `http://127.0.0.1:${server.address().port}/examples/wasm_browser/index.html`;

const document = [
  "# Operations",
  "",
  "# Backups",
  "",
  "Backups run every night at two in the morning and are kept for thirty days.",
  "",
  "# Garden",
  "",
  "Tomato plants need full sun and regular watering through the summer.",
].join("\n");

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  page.on("pageerror", (error) => console.log("page error:", error.message));

  await page.goto(url);
  await page.waitForFunction(() => document.querySelector("#status").textContent.length > 0, null, { timeout: 60_000 });

  await page.selectOption("#kind", "markdown");
  await page.fill("#content", document);
  await page.click("#index");
  await page.waitForFunction(() => document.querySelector("#status").textContent.includes("saved"), null, { timeout: 15 * 60_000 });
  console.log("ok  indexed and saved");

  await page.fill("#question", "when do the backups run");
  await page.click("#search");
  await page.waitForSelector("#results li", { timeout: 60_000 });
  const top = await page.textContent("#results li .meta");
  assert.ok(top.includes("Backups"), `top result was "${top}"`);
  console.log(`ok  meaning search -> ${top}`);

  await page.fill("#question", "tomato");
  await page.click("#keyword");
  const keywordTop = await page.textContent("#results li .meta");
  assert.ok(keywordTop.includes("Garden"), `keyword top result was "${keywordTop}"`);
  console.log(`ok  keyword search -> ${keywordTop}`);

  // Results are shown as text: an indexed <b> tag must not become an element on the page.
  await page.fill("#content", "<img src=x onerror=alert(1)> Injected words stay text.");
  await page.click("#index");
  await page.waitForFunction(() => document.querySelector("#status").textContent.includes("saved"), null, { timeout: 60_000 });
  await page.fill("#question", "injected words");
  await page.click("#search");
  await page.waitForFunction(() => document.querySelectorAll("#results li p").length > 0, null, { timeout: 60_000 });
  assert.equal(await page.locator("#results img").count(), 0, "indexed HTML must not become elements");
  console.log("ok  indexed markup is shown as text");

  // A reload brings the saved database back from IndexedDB.
  await page.reload();
  await page.waitForFunction(() => document.querySelector("#status").textContent.includes("saved notes"), null, { timeout: 60_000 });
  // The last save holds the injected text (it replaced the same source), so search for that.
  await page.fill("#question", "injected words");
  await page.click("#search");
  await page.waitForFunction(() => document.querySelectorAll("#results li").length > 0, null, { timeout: 120_000 });
  const saved = await page.textContent("#results li p");
  assert.ok(saved.includes("Injected words"), `after reload the top result was "${saved}"`);
  console.log("ok  reload keeps the saved notes");
} finally {
  await browser.close();
  server.close();
}
