# tpt-rag-wasm

A WebAssembly and JavaScript wrapper for tpt-raglite, built with [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/). It exposes a small `RAGLiteWasm` class to JavaScript.

> **Status: experimental.** The JavaScript surface is small, and the crate is not usable in a browser yet. Browser storage (OPFS or IndexedDB) is not written, and the constructor still opens a database folder on disk. Read [Current limitations](#current-limitations) before you use it. The rest of the engine is documented in [`tpt-rag-core`](../tpt-rag-core/README.md).

## Contents

- [What works today](#what-works-today)
- [Current limitations](#current-limitations)
- [Build](#build)
- [JavaScript API](#javascript-api)
- [Examples](#examples)
- [Errors](#errors)
- [Publishing](#publishing)

## What works today

- The crate compiles as a `cdylib` and as an `rlib`, so the Rust side can depend on it.
- `RAGLiteWasm` has a constructor, `add_text`, and `query`.
- The `release-wasm.yml` workflow runs `wasm-pack build` and publishes an npm package when a `v*` tag is pushed. The published package has the same limitations as this crate.

## Current limitations

These are the reasons the examples below are a preview of the API, not a working recipe:

1. **No browser storage.** `RAGLiteWasm::new` calls `RAGLite::open` with a path. That opens a folder on disk through the native file system code. A browser has no such folder, so the constructor is expected to fail there. This has not been run in a browser. OPFS and IndexedDB storage are the missing pieces.
2. **Unverified build for `wasm32-unknown-unknown`.** The core crate depends on ONNX Runtime and SQLite. Whether they build for that target with the current dependencies has not been verified.
3. **No embeddings in the browser yet.** The model is loaded from the same file lookup as the native build, so there is no way to pass model bytes in from JavaScript.
4. **Few options.** The constructor takes only a name. There is no way to set tags, sources, filters, or the configuration from JavaScript. The API has no delete or flush.
5. **`query` returns a JSON string.** It is not a JavaScript array. Parse it with `JSON.parse`.

If you need retrieval in a browser today, use the [Python package](../tpt-rag-py/README.md) or the [C library](../tpt-rag-ffi/README.md) on a server, and call it from the page over HTTP. `raglite serve` provides a JSON API for that.

## Build

Requirements: Rust with the `wasm32-unknown-unknown` target, and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/).

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build tpt-rag-wasm --target bundler --release --out-dir pkg
```

The output is in `tpt-rag-wasm/pkg/`: a JavaScript module, type definitions (`.d.ts`), and a `package.json`.

## JavaScript API

```ts
class RAGLiteWasm {
  constructor(db_name: string);             // opens the database folder (see limitations)
  add_text(text: string): void;             // stored with no source label and no tags
  query(query_text: string, top_k: number): string;   // a JSON array, as a string
}

// What each element of the parsed JSON array looks like:
interface JsSearchResult {
  score: number;
  text: string;
  source: string;
  tags: string[];
}
```

Failures are thrown as a JavaScript string, the error's message. There is no error object.

The wasm-bindgen type definitions in `pkg/` are the authoritative signatures for the build you have.

## Examples

These show how the API is meant to be used. They are the calls this crate exposes, and they will fail at the constructor until browser storage exists (see [Current limitations](#current-limitations)).

### 1. Open, add, and search

```js
import init, { RAGLiteWasm } from "./pkg/tpt_rag_wasm.js";

await init();                               // loads the wasm module once

const db = new RAGLiteWasm("notes");
db.add_text("The nightly backup runs at 02:00 and keeps seven days of history.");
db.add_text("Jupiter is the largest planet in the solar system.");

const json = db.query("when do backups run", 3);
const results = JSON.parse(json);

for (const r of results) {
  console.log(`[${r.score.toFixed(2)}] ${r.text}`);
}
```

### 2. Parse results safely

`query` can throw, and `JSON.parse` can throw on a malformed string. Handle both:

```js
function search(db, question, k = 3) {
  let raw;
  try {
    raw = db.query(question, k);
  } catch (err) {
    // Errors arrive as a plain string, the error's message.
    throw new Error(`search failed: ${err}`);
  }

  const results = JSON.parse(raw);
  return results.filter((r) => r.score > 0.2); // drop weak matches; tune the cutoff for your data
}
```

### 3. Keep the database in one place

Create the database once and reuse the instance, instead of constructing it on every search:

```js
// db.js
import init, { RAGLiteWasm } from "./pkg/tpt_rag_wasm.js";

let instance = null;

export async function getDb() {
  if (!instance) {
    await init();
    instance = new RAGLiteWasm("notes");
  }
  return instance;
}
```

```js
// search.js
import { getDb } from "./db.js";

export async function ask(question) {
  const db = await getDb();
  return JSON.parse(db.query(question, 5));
}
```

### 4. Use the results in a page

Show each hit with its source tag, escaping the text so a document cannot inject markup:

```js
import { ask } from "./search.js";

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

export async function renderResults(question, container) {
  const results = await ask(question);
  container.innerHTML = results
    .map((r) => `<li><strong>${escapeHtml(r.source)}</strong><p>${escapeHtml(r.text)}</p></li>`)
    .join("");
}
```

Text added with `add_text` has the source `inline`, because the JavaScript API cannot set a label yet.

## Errors

| Situation | What you get |
|---|---|
| The database folder cannot be opened (including in a browser) | A thrown string from the constructor |
| `add_text` fails, for example the model cannot run | A thrown string |
| `query` fails | A thrown string |
| The query result is not valid JSON | `JSON.parse` throws a `SyntaxError`. This should not happen; report it if it does |

A thrown value is a string, not an `Error`, so check `typeof err` if you need to distinguish.

## Publishing

On a `v*` tag, the `release-wasm.yml` workflow builds the package with `wasm-pack build --target bundler` and publishes it to npm as `tpt-raglite-wasm`. The version comes from the tag. Nothing is published from a normal push.

## License

MIT OR Apache-2.0. © TPT Solutions.
