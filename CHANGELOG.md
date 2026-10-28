# Changelog

All notable changes are recorded here. The project follows [Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [Unreleased]

Nothing is published yet. The first release will be 0.1.0.

### Added

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

### Not yet done

- WebAssembly storage (OPFS and IndexedDB), the wasm build in CI, and the JavaScript package.
- DOCX parsing, glob patterns for ingestion, and the `raglite` command-line tool.
- Recall and latency benchmarks.
