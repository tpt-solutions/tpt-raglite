// End-to-end check of the browser path, run in Node: a JavaScript embedding model computes the
// vectors, the wasm engine indexes the README and answers questions. The model is downloaded on
// first run (about 23 MB), so this is a manual check, not part of CI.
//
//   npm install --no-save @huggingface/transformers@4.3.2   (in this folder)
//   wasm-pack build --target web ../..
//   node check.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import assert from "node:assert/strict";
import { pipeline } from "@huggingface/transformers";

const pkg = await import(
  new URL("../../pkg/tpt_rag_wasm.js", import.meta.url).href
);
pkg.initSync({
  module: readFileSync(fileURLToPath(new URL("../../pkg/tpt_rag_wasm_bg.wasm", import.meta.url))),
});
const { TptRag } = pkg;

const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "q8" });
const embed = async (texts) => (await extractor(texts, { pooling: "mean", normalize: true })).tolist();

const readme = readFileSync(new URL("../../../README.md", import.meta.url), "utf8");
const rag = new TptRag(384);
const chunks = JSON.parse(rag.plan("markdown", readme));
const vectors = [];
for (let i = 0; i < chunks.length; i += 32) {
  for (const v of await embed(chunks.slice(i, i + 32).map((c) => c.text))) vectors.push(...v);
}
assert.equal(rag.add("README.md", [], JSON.stringify(chunks), new Float32Array(vectors)), chunks.length);

// Each question should find the README section that answers it.
const expectations = [
  ["what is the license", "License"],
  ["how do I use it from Python", "Python"],
  ["which score ranges do the search methods return", "Searching"],
];
for (const [question, section] of expectations) {
  const [vector] = await embed([question]);
  const [top] = JSON.parse(rag.query(new Float32Array(vector), 3));
  assert.ok(top.section?.includes(section), `"${question}" found [${top.section}], expected ${section}`);
  console.log(`ok  ${question} -> ${top.section}`);
}

// The database survives an export and an import.
const reloaded = TptRag.import(rag.export());
assert.equal(JSON.parse(reloaded.stats()).chunks, chunks.length);
console.log(`ok  export and import keep ${chunks.length} chunks`);
void pathToFileURL;
