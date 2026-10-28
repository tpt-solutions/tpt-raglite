# Changelog

All notable changes to `tpt-rag-core` are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0.

The repository-wide changelog is at [`../CHANGELOG.md`](../CHANGELOG.md). It also covers the bindings.

## [Unreleased]

Nothing is published yet. The first release will be 0.1.0.

### Added

- **Configuration.** `RagConfig` holds the HNSW, chunking, flush, and model settings. `RAGLite::open_with_config` takes it, and `validate()` rejects zero values before the database opens.
- **Ingestion.**
  - `add_document`, `add_text`, `add_directory` (recursive, with a progress callback), and `add_glob`.
  - `sync_directory` ingests new and changed files, skips unchanged ones by content hash, and removes documents whose files are gone. Schema version 4 adds the `documents` table of hashes.
  - Re-adding a file replaces its chunks. `delete_document` hides a document's vectors from search at once.
  - Adding a document is one SQLite transaction. If the metadata write fails, the new vectors are hidden again.
- **Parsers.** Plain text, Markdown, HTML, PDF (page by page), CSV (each row as `header: value` pairs), JSON (each leaf as `path: value`), Word (`.docx`, body text only), and source code and configuration files as text.
- **`Parser` trait and `ParserRegistry`.** `register_parser` adds a parser or overrides a built-in one for an extension.
- **Chunking.** Structure-aware. A chunk never crosses a Markdown heading, a PDF page, or the edge of a Markdown code block. Each section is chunked on its own. Documents without headings are one section. `SemanticChunker` never drops text at sentence boundaries, and it does not end a sentence at an abbreviation such as "Dr.".
- **Search.**
  - `query` (vector), `query_filtered` (tags, source prefix, and creation time), `query_hybrid` (vector and BM25 fused with reciprocal rank fusion), `query_reranked` (cross-encoder), and `query_diverse` (maximal marginal relevance).
  - `neighbors(chunk_id, radius)` returns the chunks before and after a hit, in reading order.
  - Batched embedding: texts are grouped by length and run through ONNX in batches, with attention masking.
- **Citations.** `SearchResult` carries `chunk_id`, `created_at`, `start_offset` and `end_offset` (byte range in the text the chunk was cut from), and `section` (heading path for Markdown, `page N` for PDF). Schema versions 2 and 3.
- **Index and storage.** A disk-backed HNSW index plus a SQLite store. The index is written atomically (temp file, `fsync`, rename) with a checksum and an epoch shared with the metadata. A missing or out-of-step index is rebuilt from SQLite instead of refusing to open.
- **Model identity.** The database records the model file name and vector size. A different model is refused unless `reembed_on_model_change` is set. Vector size is read from the model and is not fixed at 384.
- **Schema migrations.** An ordered list in `metadata.rs`. Each step runs in its own transaction with its version record. A database from a newer release is refused before anything changes.
- **Diagnostics.** `check()` runs the health checks behind `raglite doctor`. `stats()` reports chunk, vector, and index sizes.
- **Evaluation.** `eval_cases` and `raglite synth` generate queries from chunks. `raglite eval` reports recall@k, MRR, and nDCG@k.
- **`raglite` command.** `ingest`, `query`, `sync`, `watch`, `serve` (HTTP JSON API, or MCP over stdio with `--mcp`), `stats`, `doctor`, `eval`, `synth`, `delete`, and `rebuild`.
- **HTTP API.** `GET /health`, `GET /stats`, `POST /query` (vector, hybrid, rerank, and diverse modes, with filters in vector mode), `POST /text`, and `POST /delete`. It binds to loopback by default and refuses other addresses unless `--allow-remote` is given. It has no authentication, and it cannot read files from the server's disk. Bodies are limited to 1 MB.
- **MCP server.** `raglite serve <db> --mcp` exposes `raglite_query`, `raglite_add_text`, `raglite_delete`, and `raglite_stats` over stdio. It opens no network port.
- **Optional features.** `embedded-model` compiles the model into the binary. `reranking` (on by default) compiles the cross-encoder code.
- **Documentation.** Every public item is documented.

### Changed

- **Index recovery.** A missing, corrupt, or out-of-step index is rebuilt from SQLite instead of refusing to open.
- **Model lookup** no longer searches the current working directory. It uses `RagConfig::model_path`, `RAGLITE_MODEL` or `RAGLITE_RERANKER`, or the database folder.
- **Hybrid search** scores are scaled to 0 to 1. The best possible score is 1.
- **Reranked search** takes the best `max(rerank_top_n, top_k)` vector candidates, not only `top_k`.
- **Deletes** hide vectors from search immediately.
- **Cargo.lock** is committed, so builds use the exact dependency versions, including the `ort` release candidate.

### Fixed

- **Tokenizer.** Real WordPiece tokenization. A test compares token ids with the bert-base-uncased reference.
- **HNSW.** Nodes are inserted into every layer they belong to, neighbour lists are pruned, and random levels use `1 / ln(M)`.
- **PDF.** A parser panic becomes an error. A PDF with no text layer is reported as `EmptyDocument`.
- **Markdown** is parsed with a real parser. HTML drops `<script>`, `<style>`, and `<noscript>` content.
- **Text files.** A byte-order mark is removed, invalid UTF-8 is replaced, and files over 64 MB are rejected.
- **HTML parser** compiled against the markup5ever_rcdom API that stores parse errors in a `RefCell`. The crate did not build before this.
- **`raglite` help text** had unescaped quotes, so the binary did not compile.
- **`Cargo.lock`** was missing the `zip` and `glob` entries, so `--locked` builds failed.

### Not yet done

- WebAssembly storage (OPFS and IndexedDB). The `tpt-rag-wasm` crate still opens a native database folder.
- Recall and latency benchmarks beyond `examples/bench_hnsw.rs`.
- Publishing to crates.io. The FFI, Python and WASM crates depend on this crate by path, and crates.io needs a version on that dependency first.
