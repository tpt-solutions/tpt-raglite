# tpt-raglite

**The SQLite for AI.** An embeddable, zero-configuration RAG engine in Rust. It parses your documents, splits them into chunks, embeds them with a local ONNX model, and stores them in a folder you can copy, back up, or delete.

Everything runs offline, in the same process as your application. There is no server, no API key, and no Docker.

> **Status: pre-release (0.1.0).** Not published to crates.io, PyPI, or npm yet. Build from source as described below.

## Features

- **Local embeddings**: `all-MiniLM-L6-v2` (384 dimensions) through ONNX Runtime, batched and L2-normalized.
- **Four ways to search**: vector similarity, hybrid vector + keyword (reciprocal rank fusion), vector search followed by cross-encoder reranking, and diverse results (maximal marginal relevance).
- **Persistent**: a disk-backed HNSW index plus a SQLite store for chunk text, tags, and sources. The index is rebuilt from SQLite if its file is lost or damaged.
- **Safe re-ingest**: adding a file again replaces its chunks. Deleting a document removes its chunks from search at once.
- **Bindings**: Rust, a C ABI with a generated header, Python (PyO3), and WebAssembly for the browser (chunking, vector and keyword search; the browser computes the embeddings, see [Browser](#browser-webassembly)).
- **Tools**: a `raglite` command line, a local HTTP API, and an MCP server for AI assistants. See [Command line](#command-line).

## Quick start

First, download the model files (about 46 MB, checksum-verified):

```bash
raglite models pull                 # any platform: downloads and checks the models into ./models
bash scripts/fetch_model.sh        # or: powershell -File scripts/fetch_model.ps1
```

This creates `models/all-MiniLM-L6-v2.onnx`, `models/vocab.txt`, and the optional reranker `models/ms-marco-MiniLM-L-6-v2.onnx`.

### Rust

```rust
use std::path::Path;
use tpt_rag_core::{Metadata, RAGLite};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rag = RAGLite::open(Path::new("./my_db"))?;

    rag.add_document(Path::new("manual.pdf"), &["docs"])?;
    rag.add_text(
        "Some important information.",
        &Metadata { source: Some("notes".into()), tags: vec![] },
    )?;

    for r in rag.query("How do I configure the engine?", 5)? {
        println!("[{:.2}] {}", r.score, r.text);
    }
    Ok(())
}
```

Runnable examples: [`tpt-rag-core/examples/rust_basic.rs`](tpt-rag-core/examples/rust_basic.rs) and [`tpt-rag-core/examples/rust_hybrid_rerank.rs`](tpt-rag-core/examples/rust_hybrid_rerank.rs).

### Python

Build the extension first (see [Install from source](#install-from-source)):

```python
from tpt_raglite import RAGLite

with RAGLite("./my_db") as db:          # flushes the index on exit
    db.add_files(["manual.pdf", "notes.md"], tags=["docs"])
    db.add_directory("./more_docs")      # recursive; returns {"added": [...], "failed": [...]}
    for r in db.query("How do I configure the engine?", top_k=5):
        print(f"[{r.score:.2f}] {r.source}: {r.text}")
```

Examples: [`examples/python_quickstart.py`](examples/python_quickstart.py), [`examples/python_pdf_qa.py`](examples/python_pdf_qa.py).

### C

This is [`examples/c_basic.c`](examples/c_basic.c), which CI compiles and runs:

```c
#include <stdio.h>
#include "tpt_raglite.h"

int main(void) {
    TptRagHandle *rag = tpt_rag_create("./my_c_db");
    if (!rag) {
        fprintf(stderr, "open failed: %s\n", tpt_rag_last_error());
        return 1;
    }

    if (tpt_rag_add_text(rag, "Jupiter is the largest planet in the solar system.") != TptRagOk) {
        fprintf(stderr, "ingest failed: %s\n", tpt_rag_last_error());
    }

    TptRagResult *results = NULL;
    size_t count = 0;
    if (tpt_rag_query(rag, "How big is Jupiter?", 5, &results, &count) == TptRagOk) {
        for (size_t i = 0; i < count; i++) {
            printf("[%.2f] %s\n", results[i].score, results[i].text);
        }
    }

    tpt_rag_free_results(results, count);
    tpt_rag_flush(rag);
    tpt_rag_destroy(rag);
    return 0;
}
```

Functions that add, query, or flush return a `TptRagError` (`TptRagOk` is 0). `tpt_rag_create` returns null on failure instead. After a failure, `tpt_rag_last_error()` describes it; the pointer stays valid until the next call on the same thread. Result strings belong to the library until `tpt_rag_free_results`. For non-default settings, start from `tpt_rag_default_config()` and pass the struct to `tpt_rag_create_with_config`.

## Searching

| Method (Rust / Python) | What it does | Score range |
|---|---|---|
| `query` / `query(top_k=)` | Nearest chunks by cosine similarity | -1 to 1 (higher is better) |
| `query_hybrid(text, top_k, alpha)` / `query_hybrid` | Vector and BM25 keyword rankings fused with reciprocal rank fusion. `alpha` is the vector weight, 0 to 1 | 0 to 1 |
| `query_reranked(text, top_k, rerank_top_n)` / `query_reranked` | Takes the best `max(rerank_top_n, top_k)` vector candidates and rescores them with a cross-encoder. Needs the reranker model | 0 to 1 |
| `neighbors(chunk_id, radius)` / `neighbors(chunk_id, radius=)` | The chunks just before and after a hit in the same document, in reading order, not including the hit. For expanding a hit into context. Score is 0 | 0 |
| `query_diverse(text, top_k, relevance_weight)` / `query_diverse(top_k=, relevance_weight=)` | Maximal marginal relevance: picks results that are relevant but not near-duplicates of each other. `relevance_weight` 1 is plain relevance order, 0 is the most varied. Score is the query similarity | -1 to 1 |

Scores are not comparable across methods. Compare results within one method.

Each result also carries `chunk_id` (the row id in the metadata store, stable within one database, so it can be cited), `created_at` (ingestion time in Unix seconds), and `start_offset` / `end_offset`: the byte range of the chunk in the text it was cut from. For plain text and `add_text` that is the text itself, so `text[start:end]` gives back the chunk. (The offsets count bytes. Python slices strings by character, so for non-ASCII text slice the UTF-8 bytes instead.) For PDFs, Markdown and HTML it is the extracted text, not the original file. Chunks stored before schema version 2 have no offsets (`None`). `section` is the heading path the chunk starts under, such as `Setup > Backups` for Markdown, or `page 3` for PDFs. It is `None` for other formats, for documents without headings, and for chunks stored before schema version 3. `query_filtered` (Rust) and `query(..., tags=, source_prefix=, since=)` (Python) keep only chunks with every given tag, a source starting with a prefix, or ingested since a Unix time (`created_since` in Rust). The filter is applied to the best `10 × top_k` vector matches, so a very selective filter can return fewer than `top_k` results.

## Command line

The `raglite` binary (`cargo run -p tpt-rag-core --bin raglite -- <command>`) covers the common tasks:

```bash
raglite ingest ./kb ./notes/*.md --tag kb     # folders (recursive), files, or glob patterns
raglite query ./kb "how do I rotate the key" -k 3 --tag kb
raglite query ./kb "API key" --hybrid 0.3     # or --rerank 20 with the reranker installed
raglite query ./kb "API key" --diverse 0.3     # fewer near-duplicate chunks in the results
raglite sync ./kb ./notes                   # index new and changed files, drop deleted ones
raglite watch ./kb ./notes --interval 30     # sync every 30 seconds until stopped
raglite stats ./kb
raglite doctor ./kb                           # health checks; exits 1 if anything fails
raglite eval ./kb cases.jsonl -k 5            # recall@k and MRR, see below
raglite delete ./kb ./notes/old.md
raglite rebuild ./kb                          # re-embed everything
```

`raglite synth ./kb --per-source 5 > cases.jsonl` writes cases made from the chunks themselves, so `eval` can run without a question set. Each case is a few words of one chunk, and its expected source is that chunk's source. These check that a chunk can be found from its own words, which is easier than real questions, so use them to compare settings rather than as a quality score.

`eval` reads one JSON object per line, `{"query": "...", "expected_source": "part of the source path"}`, and reports recall@k, mean reciprocal rank, nDCG@k, and the first misses. Use it to check a change to chunking or the model against queries you know the answers to.

### HTTP API

`raglite serve ./kb` starts a JSON API, by default on `127.0.0.1:8080`:

```bash
curl -s localhost:8080/health
curl -s localhost:8080/stats
curl -s localhost:8080/query -d '{"text": "API key", "top_k": 3, "mode": "hybrid", "alpha": 0.3}'
curl -s localhost:8080/text -d '{"text": "Rotate keys monthly.", "source": "policy", "tags": ["security"]}'
curl -s localhost:8080/delete -d '{"source": "policy"}'
```

`/query` takes the same options as the CLI: `mode` is `vector`, `hybrid`, `rerank`, or `diverse`, and `tags`, `source_prefix` and `since` filter in `vector` mode. Results include the section and byte offsets. Requests are handled one at a time. Bodies are limited to 1 MB.

The API has **no authentication**. The server refuses any address other than loopback unless you pass `--allow-remote`, and it never reads files from the server's disk. Anyone who can reach the port can read and change the database, so do not expose it without a proxy that adds authentication.

### MCP

`raglite serve ./kb --mcp` speaks the [Model Context Protocol](https://modelcontextprotocol.io) over stdin and stdout, so an MCP client can search and update the database as tools. It has four tools: `raglite_query`, `raglite_add_text`, `raglite_delete` and `raglite_stats`. The tools accept the same options as the HTTP API. Nothing listens on a network port. Add it to a client's configuration like this:

```json
{
  "mcpServers": {
    "raglite": {
      "command": "raglite",
      "args": ["serve", "/absolute/path/to/kb", "--mcp"]
    }
  }
}
```

Logs go to stderr, so stdout carries only protocol messages.

## Browser (WebAssembly)

`tpt-rag-wasm` runs the engine in the browser. The browser cannot run the native ONNX model or SQLite, so the host computes the embeddings and the wasm module does the rest: parsing and chunking with the same boundaries as the native library, the HNSW vector index, BM25 keyword search, and export and import of the whole database as bytes.

```js
import init, { TptRag } from "./tpt_rag_wasm.js";
await init();

const rag = new TptRag(384);                                 // dimensions of your embedding model
const chunks = JSON.parse(rag.plan("markdown", text));       // [{text, start_offset, end_offset, section}]
const vectors = await embedTexts(chunks.map((c) => c.text)); // your model: one Float32Array with every vector
rag.add("my-notes", ["tag"], JSON.stringify(chunks), vectors);

const hits = JSON.parse(rag.query(await embedOne(question), 5)); // by meaning
const words = JSON.parse(rag.keyword(question, 5));              // by keywords, no model needed

const bytes = rag.export();          // keep it in IndexedDB or OPFS
const again = TptRag.import(bytes);  // load it on the next visit
```

The example in [`examples/wasm_browser`](examples/wasm_browser) does this with transformers.js and the same quantized all-MiniLM-L6-v2 model, and keeps the database in IndexedDB. Build the package with `wasm-pack build --target web tpt-rag-wasm`, then serve the folder over HTTP.

Limits compared with the native library:

- **Formats:** text, Markdown and HTML. PDF, DOCX, CSV, JSON and source code are native-only, because they need file access.
- **Search:** vector and keyword only. Hybrid fusion, reranking, diverse results and metadata filters are not in the browser yet.
- **Scores:** keyword scores are BM25, not the fused scores of native hybrid search, so compare results within one method.
- **Embeddings:** whatever model the host uses. Index and question embeddings must come from the same model.
- **Size:** about 0.9 MB of wasm before compression.

Run the browser tests with `wasm-pack test --node tpt-rag-wasm`. The end-to-end check in `tpt-rag-wasm/tests/e2e` also downloads the model; its header comment explains how to run it.

## Templates

Two small Python programs in [`templates/`](templates) show the engine in use. They import the `tpt_raglite` package, so build the Python extension first.

- **[`local-notes-search`](templates/local-notes-search)**: indexes a folder of notes and finds them by meaning from the command line. Each run ingests new and changed notes and removes deleted ones, using the state in `<db>/sync.json`. Nothing leaves the machine.
- **[`docs-chatbot`](templates/docs-chatbot)**: retrieval for a documentation chatbot. It builds a prompt that answers only from the retrieved passages and cites them as `[1]`, `[2]`. You supply the model call; the template does not pick a model.

Run their tests with `python -m pytest templates/local-notes-search` (or `templates/docs-chatbot`).

## Configuration

`RagConfig::default()` (Rust) and the keyword arguments of `RAGLite(...)` (Python) use these defaults:

| Setting | Default | Meaning |
|---|---|---|
| `m` | 16 | HNSW links per node (layer 0 uses `2 * m`). Changing it rebuilds the index on open |
| `ef_construction` | 200 | Index build effort. Higher is slower and more accurate |
| `ef_search` | 100 | Query effort. Raised to `top_k` when smaller |
| `chunk.window_size` | 512 | Words per chunk. Chunks never cross a Markdown heading, a PDF page, or the edge of a code block, so each section is chunked on its own |
| `chunk.overlap` | 50 | Words shared by neighboring chunks. Must be below `window_size` |
| `flush_every` | 1000 | Write the index after this many changes. `0` turns it off |
| `reembed_on_model_change` | false | Re-embed all chunks instead of refusing to open when the model differs |
| `model_path` / `reranker_path` | none | Explicit model files (see the lookup order below) |

### Model lookup order

1. The `model_path` / `reranker_path` setting, if given.
2. The `RAGLITE_MODEL` / `RAGLITE_RERANKER` environment variable.
3. In the database folder: `model.onnx` or `models/all-MiniLM-L6-v2.onnx`, and `reranker.onnx` or `models/ms-marco-MiniLM-L-6-v2.onnx`.
4. For the embedding model only: the model compiled into the binary, if the `embedded-model` Cargo feature is on. This feature needs the model files in `models/` at build time.

The current working directory is never searched.

## Persistence

A database is a folder with two files:

- `metadata.sqlite3`: chunk text, source, tags, timestamps, and the identity of the embedding model. Each operation is written in one SQLite transaction.
- `hnsw.bin`: the vector index. Written atomically (temp file, `fsync`, rename) on `flush()`, every `flush_every` changes, and when the handle is dropped.

Drop cannot report errors, so call `flush()` explicitly when you need to know that the index is on disk. A crash can lose changes made since the last flush, but the metadata and index are never left pointing at each other's wrong ids:

- If `hnsw.bin` is missing, damaged, or from a different rebuild (checked by an epoch id stored in both files), the index is rebuilt from the chunk text. This re-embeds every chunk, which takes time on a large database.
- If you open a database with a different model file, opening fails, so vectors from two models never mix. Set `reembed_on_model_change` to switch.

Schema changes are migrations. Opening an older database runs the missing steps, each in its own transaction, so a step that fails leaves the database as it was. A database written by a newer version of the library is refused, without being changed. Upgrading is one way: an upgraded database cannot be opened by an older release.

## Supported formats

| Format | Extensions | Notes |
|---|---|---|
| Plain text | `.txt` | BOM removed; invalid UTF-8 replaced |
| Markdown | `.md`, `.markdown`, `.mdown` | Headings, paragraphs, lists, and code are kept; markup is removed |
| HTML | `.html`, `.htm` | `<script>`, `<style>`, and `<noscript>` are dropped |
| PDF | `.pdf` | Text layer only. A PDF with no text (for example, a scan) is reported as `EmptyDocument`; run OCR first |
| CSV | `.csv` | Each row becomes `header: value; header: value` |
| JSON | `.json` | Each leaf becomes `path: value`, such as `user.tags[0]: math` |
| Word | `.docx` | Body text in document order, one line per paragraph. Headers, footers, comments and embedded objects are not read |
| Source and config | `.rs`, `.py`, `.js`, `.ts`, `.go`, `.java`, `.c`, `.cpp`, `.toml`, `.yaml`, and others | Indexed as text |

Files over 64 MB are rejected. Other formats can be added with `RAGLite::register_parser` (Rust), which takes precedence over a built-in parser for the same extension. To keep a folder indexed, use `sync_directory` (Rust and Python), `raglite sync`, or `raglite watch`. A sync compares a hash of each file with the one it stored, so unchanged files are skipped, changed files are ingested again, and files that are gone from the folder are removed from the index. Polling is used rather than file system events, so it works the same on every platform. Only files that a sync has ingested are tracked for removal. Ingestion takes a folder (recursive), a list of files, or a glob pattern (`add_glob` in Rust and Python, `notes/**/*.md` on the command line). HTML repairs are logged at debug level through `tracing`.

## Model requirements

Any model that matches these rules works. The default is `all-MiniLM-L6-v2`.

- **Embedding model**: an ONNX file with inputs `input_ids` and `attention_mask`, and optionally `token_type_ids` (passed as zeros when present). Its output is either token-level `[batch, seq, hidden]` (mean-pooled over real tokens) or sentence-level `[batch, hidden]`. The vector size is read from the model when it loads.
- **Vocabulary**: a WordPiece `vocab.txt` (one token per line, with `[UNK]`, `[CLS]`, `[SEP]`, and `[PAD]`) in the same folder as the model. BERT-uncased vocabularies match the tokenizer; other tokenizers do not.
- **Reranker**: a BERT-style cross-encoder with a single logit output and the same `vocab.txt`. Scores are `sigmoid(logit)`.

## Install from source

Requirements: Rust (stable), and a C toolchain. On Windows, the prebuilt ONNX Runtime binaries need a newer MSVC standard library than Visual Studio 2022 17.8 (toolset 14.38) ships. Use a current Visual Studio 2022 install.

```bash
bash scripts/fetch_model.sh        # or on Windows: powershell -File scripts/fetch_model.ps1
cargo build --release

# Python extension
pip install maturin
cd tpt-rag-py && maturin develop --release && cd ..
```

The C library is `target/release/tpt_rag_ffi.dll` (Windows), `libtpt_rag_ffi.so` (Linux), or `libtpt_rag_ffi.dylib` (macOS). The header is `tpt_raglite.h`.

## Testing

```bash
cargo test --workspace                       # unit and integration tests
bash tpt-rag-ffi/tests/run_c_integration.sh  # C ABI tests (Linux)
pytest tpt-rag-py/tests                      # Python tests (after maturin develop)
python -m pytest templates/local-notes-search templates/docs-chatbot
wasm-pack test --node tpt-rag-wasm           # browser engine tests
```

Integration tests that need the model print a notice and pass when it is missing, so run `bash scripts/fetch_model.sh` first to exercise them for real.

On Windows with the prebuilt ONNX Runtime, the test binaries may not link (see Troubleshooting). Running the suite under WSL or on Linux avoids this.

## Troubleshooting

- **`ONNX model not found` / `no embedding model configured`**: run `raglite models pull` (or `scripts/fetch_model`), or set `RAGLITE_MODEL` to the `.onnx` file. The vocabulary must be next to it.
- **`database was built with a different embedding model`**: open it with the original model, or set `reembed_on_model_change` (Rust: in `RagConfig`; Python: `reembed_on_model_change=True`).
- **`reranker model not found`**: `query_reranked` needs `RAGLITE_RERANKER` or `reranker.onnx` in the database folder. Vector and hybrid search do not need it.
- **`EmptyDocument` for a PDF**: the file has no text layer. Run OCR on it, then add the result.
- **Linker errors mentioning `__std_max_element_d` or similar on Windows**: the ONNX Runtime binaries were built with a newer MSVC standard library than your toolset. Update Visual Studio 2022.
- **Slow first `open`**: an index was rebuilt from the chunk text. This happens after the index file was lost or damaged, or when the model changed.

## Architecture

```
tpt-rag-core/     Rust library: parsing, chunking, ONNX embedding and reranking, HNSW index, SQLite metadata;
                  also the raglite CLI (src/bin/raglite: ingest, query, serve, mcp, models)
tpt-rag-ffi/      C ABI with opaque handles and a cbindgen header (tpt_raglite.h)
tpt-rag-py/       PyO3 Python package (maturin)
tpt-rag-wasm/     Browser engine: chunking, vector and BM25 search, export and import, for WebAssembly
examples/         C, Python and browser examples
templates/        Python programs built on the engine (notes search, docs chatbot)
scripts/          Model download (fetch_model.sh / .ps1)
```

## License

MIT OR Apache-2.0. © TPT Solutions.
