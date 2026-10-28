// Browser example for tpt-raglite. The browser computes the embeddings with transformers.js,
// and the wasm engine stores, searches and exports the database. The database is kept in
// IndexedDB between visits. Build the package first: `wasm-pack build --target web tpt-rag-wasm`.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";
import init, { TptRag } from "../../tpt-rag-wasm/pkg/tpt_rag_wasm.js";

// The same model as scripts/fetch_model (quantized all-MiniLM-L6-v2, 384 dimensions).
const MODEL = "Xenova/all-MiniLM-L6-v2";
const DIMENSIONS = 384;
const BATCH = 32;
const DB_NAME = "tpt-raglite-example";
const SOURCE = "my-notes";

const $ = (selector) => document.querySelector(selector);
const say = (message) => {
  $("#status").textContent = message;
};

// Persistence. Storage can be blocked (private windows, disabled site data), so every call here
// can fail, and the page falls back to an empty database that is not saved.
function openStore() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore("blobs");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveDatabase(bytes) {
  const db = await openStore();
  await new Promise((resolve, reject) => {
    const tx = db.transaction("blobs", "readwrite");
    tx.objectStore("blobs").put(bytes, "database");
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

async function loadDatabase() {
  const db = await openStore();
  const bytes = await new Promise((resolve, reject) => {
    const request = db.transaction("blobs").objectStore("blobs").get("database");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  db.close();
  return bytes;
}

// The embedding model loads on first use, so the page itself opens quickly.
let extractor = null;
async function embed(texts) {
  if (!extractor) {
    say("loading the embedding model (about 23 MB, cached after the first load)...");
    extractor = await pipeline("feature-extraction", MODEL, { dtype: "q8" });
  }
  const vectors = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const output = await extractor(texts.slice(i, i + BATCH), { pooling: "mean", normalize: true });
    vectors.push(...output.tolist());
  }
  return vectors;
}

// Shows results with textContent only. Indexed text is data, so it must never become HTML.
function showResults(hits) {
  const list = $("#results");
  list.replaceChildren();
  for (const hit of hits) {
    const item = document.createElement("li");
    const meta = document.createElement("div");
    meta.className = "meta";
    const where = hit.section ? ` | ${hit.section}` : "";
    meta.textContent = `${hit.score.toFixed(3)}  ${hit.source}${where}`;
    const text = document.createElement("p");
    text.textContent = hit.text;
    item.append(meta, text);
    list.append(item);
  }
  if (hits.length === 0) say("no results");
}

async function main() {
  await init();

  let rag;
  try {
    const saved = await loadDatabase();
    rag = saved ? TptRag.import(saved) : new TptRag(DIMENSIONS);
    say(saved ? "loaded your saved notes" : "no saved notes yet");
  } catch (error) {
    rag = new TptRag(DIMENSIONS);
    say(`starting empty (saved notes unavailable: ${error})`);
  }

  $("#index").addEventListener("click", async () => {
    try {
      const chunks = JSON.parse(rag.plan($("#kind").value, $("#content").value));
      const vectors = await embed(chunks.map((chunk) => chunk.text));
      rag.add(SOURCE, [], JSON.stringify(chunks), new Float32Array(vectors.flat()));
      say(`indexed ${chunks.length} chunks`);
      try {
        await saveDatabase(rag.export());
        say(`indexed ${chunks.length} chunks and saved`);
      } catch (error) {
        say(`indexed ${chunks.length} chunks (not saved: ${error})`);
      }
    } catch (error) {
      say(`could not index: ${error}`);
    }
  });

  $("#search").addEventListener("click", async () => {
    try {
      const question = $("#question").value;
      const [vector] = await embed([question]);
      showResults(JSON.parse(rag.query(new Float32Array(vector), 5)));
      say("done");
    } catch (error) {
      say(`could not search: ${error}`);
    }
  });

  $("#keyword").addEventListener("click", () => {
    showResults(JSON.parse(rag.keyword($("#question").value, 5)));
    say("keyword search done (no model needed)");
  });
}

main();
