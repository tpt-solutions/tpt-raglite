# Changelog

All notable changes are recorded here. The project follows [Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [Unreleased]

Nothing is published yet. The first release will be 0.1.0.

### Added
- Browser build: hybrid search (`hybrid`), reciprocal rank fusion with the native formula and scale.
- Browser build: `tpt-rag-wasm` runs the engine in WebAssembly. The host computes embeddings with the same model family (for example transformers.js with quantized all-MiniLM-L6-v2), and the wasm module provides plan, add, query, keyword, export and import, with the native chunk boundaries and an HNSW index. The example in `examples/wasm_browser` stores the database in IndexedDB. The wasm binary is about 0.9 MB. Native code is behind a `native` feature in `tpt-rag-core` (on by default), so the pure parts build for `wasm32-unknown-unknown`.

- `RagConfig` with HNSW, chunking, flush, and model settings. `RAGLite::open_with_config` takes it. Available in Rust, the C ABI (`tpt_rag_default_config`, `tpt_rag_create_with_config`), and Python (keyword arguments).
- Cross-encoder reranking with a real model. `query_reranked` scores candidates as query-passage pairs, loads the model once, and returns an error when no reranker is installed.
- Batched embedding: texts are grouped by length and run through ONNX in batches with attention masking.
- Parsers for CSV (each row as `header: value` pairs) and JSON (each leaf as `path: value`), and for source code and configuration files.
- `ParserRegistry` and `RAGLite::register_parser`, so applications can add or override parsers.
- `RAGLite::add_directory`: recursive ingestion that reports added and failed files and calls back with progress.
- `RAGLite::rebuild_index`: re-embeds every chunk with the current model.
- `delete_document` in Python. `query_hybrid`, `query_reranked`, `flush`, `stats`, and `add_text` with source and tags in Python.
- Typed Python exceptions: `RAGLiteError`, `ModelError`, `UnsupportedFormatError`, `EmptyDocumentError`. Heavy calls release the GIL.
- Context manager support in Python (`with RAGLite(...) as db:` flushes on exit).
- Optional `embedded-model` Cargo feature: compiles the model into the binary.
- Model download scripts `scripts/fetch_model.sh` and `scripts/fetch_model.ps1`, with pinned SHA-256 hashes.
- Integration tests against the real model (`tpt-rag-core/tests/integration.rs`), C ABI tests (`tpt-rag-ffi/tests/`), and Python tests (`tpt-rag-py/tests/`).
- Examples: Rust (`tpt-rag-core/examples/`), Python (`examples/`), and C (`examples/c_basic.c`).
- Templates: `templates/docs-chatbot` (cited answers from a documentation folder) and `templates/local-notes-search` (incremental sync of a notes folder by modification time, with search).
- `raglite eval` reports nDCG@k alongside recall@k and MRR.
- `raglite serve <db> --mcp`: an MCP server over stdio with four tools (`raglite_query`, `raglite_add_text`, `raglite_delete`, `raglite_stats`). It runs the same routes as the HTTP API and needs no network port.
- `raglite serve <db>`: a JSON API over HTTP. Routes: `GET /health`, `GET /stats`, `POST /query` (vector, hybrid, rerank, and diverse modes, with filters in vector mode), `POST /text`, and `POST /delete`. It binds to loopback by default and refuses other addresses unless `--allow-remote` is given, because it has no authentication. It cannot read files from the server's disk. Requests are handled one at a time, and bodies are limited to 1 MB. Adds the `tiny_http` dependency.
- Schema migrations are an ordered list in `metadata.rs`. Each step runs in its own transaction together with its version record, so a failed step leaves the database as it was. Adding a schema version means adding one step. A database from a newer library is still refused, before anything is changed.
- Structure-aware chunking: a chunk never crosses a Markdown heading, a PDF page, or the edge of a Markdown code block (fenced or indented). Code blocks therefore never share a chunk with the prose around them. Each section is chunked on its own, so a chunk's section is the one it sits in. Documents without headings, and other formats, are chunked as one section, as before.
- `raglite models pull [folder]` downloads the embedding model, its vocabulary and the reranker into `./models` (or the folder given), with the same pinned SHA-256 checks as the download scripts. A file that already matches is kept, and a download that does not match is discarded. It works on every platform, without `curl` or `bash`. Adds the `ureq` and `sha2` dependencies.
- `raglite synth <db>` prints eval cases made from the chunks, as JSONL. The cases are a few words of one chunk with that chunk's source as the expected result, so `raglite eval` can run without a question set. They are easier than real questions, so they compare settings rather than measure quality. `RAGLite::eval_cases` does the same from Rust.
- Folder sync: `RAGLite::sync_directory` (Rust), `sync_directory` (Python), and `raglite sync` / `raglite watch`. A sync ingests new and changed files, skips unchanged ones by content hash, and removes files that are gone from the folder. `watch` repeats the sync every `--interval` seconds (default 30). Schema version 4 adds a `documents` table of content hashes. Opening an older database upgrades it.
- Sections: Markdown chunks record the heading path they start under (`Setup > Backups`), and PDF chunks record their page (`page 3`), as `section` in `SearchResult` (Rust and Python). PDF text is now extracted page by page, and pages are joined with a blank line. `raglite query` prints it after the source. Schema version 3; older databases are upgraded when opened, and their chunks have no section. The migration now adds each column with its own type.
- Citations: every chunk stores its byte offsets in the text it was cut from. `SearchResult` has `start_offset` and `end_offset` (Rust and Python), and `raglite query` prints them as `(bytes start-end)`. Schema version 2. Databases from version 1 are upgraded in place when opened. Note that a database upgraded to version 2 cannot be opened by an older release, which refuses newer schema versions.
- Neighbour expansion: `RAGLite::neighbors(chunk_id, radius)` and Python `neighbors` return the chunks before and after a hit in the same document, in reading order. The hit itself is not included, so its score is kept. Neighbours are ordered by their byte offsets, so chunks stored before schema version 2 have none.
- Diverse search: `RAGLite::query_diverse` (Rust), `query_diverse` (Python) and `raglite query --diverse` apply maximal marginal relevance, so results are relevant but not near-duplicates. `relevance_weight` 1 is plain relevance order and 0 the most varied.
- Python `query` takes `since` (Unix seconds) to keep only chunks ingested at or after that time, matching the Rust `created_since` filter and the CLI `--since` option.
- Every public item in `tpt-rag-core` is documented.

### Changed

- **Index persistence**: the index file carries an epoch that matches the metadata. A rebuild that was interrupted is detected on open and redone, instead of silently pointing chunks at the wrong vectors.
- **Index recovery**: a missing, corrupt, or out-of-step index is rebuilt from SQLite instead of refusing to open.
- **Model identity**: the database stores the model file name and vector size. Opening it with a different model fails unless `reembed_on_model_change` is set.
- **Ingest**: adding or replacing a document is one SQLite transaction. If the metadata write fails, the new vectors are hidden again so search and metadata stay consistent.
- **Deletes**: removing a document hides its vectors from search immediately.
- **Hybrid search** scores are scaled to 0 to 1. The best possible score is 1.
- **Reranked search** takes the best `max(rerank_top_n, top_k)` vector candidates, not only `top_k`.
- **Model lookup** no longer searches the current working directory. Use `RAGLITE_MODEL`, `RAGLITE_RERANKER`, or the database folder (see README).
- **Vector size** is read from the model. It is no longer hard-coded to 384.
- **Python packaging**: the license is declared as the expression `MIT OR Apache-2.0`. The license classifiers that listed both licenses separately were removed.
- The `tests/` folder at the repository root was replaced by `tpt-rag-core/tests/`.
- `Cargo.lock` is no longer ignored, so it can be committed. Committing it makes CI build the exact dependency versions.

### Fixed

- Tokenizer: real WordPiece tokenization. A test compares token ids with the bert-base-uncased reference.
- Chunker: no text is dropped at sentence boundaries; overlap must be smaller than the window; abbreviations such as "Dr." do not end sentences.
- HNSW: nodes are inserted into every layer they belong to, neighbor lists are pruned, and random levels use `1 / ln(M)`.
- Index files are written atomically with a checksum, and are validated on load.
- FFI: panics are caught at the boundary. Strings with NUL bytes keep their text. Results are freed with the matching allocator.
- PDF: a parser panic becomes an error. A PDF with no text layer is reported as `EmptyDocument`.
- Markdown is parsed with a real parser. HTML drops `<script>`, `<style>`, and `<noscript>` content.
- Text files: a BOM is removed, invalid UTF-8 is replaced, and files over 64 MB are rejected.
- The HTML parser compiled against the markup5ever_rcdom API that stores parse errors in a `RefCell`, so the crate did not build.
- The `raglite` help text had unescaped quotes, so the CLI binary did not compile.
- `Cargo.lock` was missing the `zip` and `glob` entries, so builds with `--locked` failed. It has been regenerated.

### Not yet done

- WebAssembly storage (OPFS and IndexedDB), the wasm build in CI, and the JavaScript package.
- The `raglite serve` command, `raglite models pull`, watch mode, and an MCP or HTTP server.
- Recall and latency benchmarks.
