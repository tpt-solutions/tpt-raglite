# tpt-raglite — Master Task Checklist

> **Project:** tpt-raglite ("The SQLite for AI")  
> **Org:** TPT Solutions  
> **License:** MIT OR Apache-2.0  
> Track progress by checking off items as you complete them.

---

## Phase 0 – Project Scaffolding

- [x] Create Cargo workspace (`Cargo.toml` with `[workspace]`)
- [x] Create `tpt-rag-core` crate stub
- [x] Create `tpt-rag-ffi` crate stub
- [x] Create `tpt-rag-py` crate stub + `pyproject.toml`
- [x] Create `tpt-rag-wasm` crate stub
- [x] Add `models/` directory with `.gitkeep`
- [x] Add `tests/` directory with integration test stub
- [x] Write `README.md` (project overview, install, quickstart)
- [x] Add `LICENSE-MIT` and `LICENSE-APACHE` (TPT Solutions)
- [x] Add `.gitignore` (Rust/Python/WASM artifacts)
- [x] Initialize git repository

---

## Phase 1 – Core Engine (Pure Rust)

### Parsing

- [x] Add `pdf-extract` dependency for PDF text extraction
- [x] Add `html5ever` for HTML parsing
- [x] Implement `Parser` trait with implementations: `PlainText`, `Markdown`, `Html`, `Pdf`
- [x] Unit test: parse each format, assert text extracted correctly

### Chunking

- [x] Implement sliding-window chunker (512-token window, 50-token overlap)
- [x] Implement sentence-boundary detection for semantic chunk splitting
- [x] Expose `ChunkConfig` struct (configurable window/overlap)
- [x] Unit tests: verify chunk sizes, overlap correctness, no mid-sentence splits

### Local ONNX Embedding

- [x] Add `ort` crate (ONNX Runtime bindings)
- [x] Download quantized INT8 `all-MiniLM-L6-v2` ONNX model (`scripts/fetch_model.sh` / `.ps1`, checksum-verified into `models/`)
- [x] Embed model into binary via `include_bytes!` (optional `embedded-model` feature; see 6G)
- [x] Implement `EmbeddingModel` struct wrapping `ort::session::Session`
- [x] Implement batch embedding (batched ONNX runs with attention masking; `rayon` was never used, so the earlier claim is dropped)
- [x] Unit test: embed a known sentence, assert 384-dim output, cosine similarity sanity check

### In-Memory HNSW Index (custom Rust)

- [x] Define `HnswIndex` struct with configurable `M` and `ef_construction` params
- [x] Implement node insertion with layered graph construction
- [x] Implement KNN search (`ef_search` parameter)
- [x] Implement cosine distance metric
- [x] Unit tests: insert N vectors, query nearest neighbor, assert recall > 0.95 on synthetic data

### SQLite Metadata Store

- [x] Add `rusqlite` dependency
- [x] Implement `MetadataStore` struct managing a local `.sqlite3` file
- [x] Schema: `chunks(id, vector_id, text, source_path, created_at, tags_json)`
- [x] Implement `insert_chunk`, `get_chunk_by_vector_id`, `delete_by_source`
- [x] Unit tests: insert/query/delete round-trip

### RAGLite Core API (`tpt-rag-core`)

- [x] Define `RAGLite` struct holding `HnswIndex` + `MetadataStore` + `EmbeddingModel`
- [x] Implement `RAGLite::open(path: &Path) -> Result<Self>`
- [x] Implement `add_document(path: &Path, tags: &[&str]) -> Result<()>`
- [x] Implement `add_text(text: &str, metadata: &Metadata) -> Result<()>`
- [x] Implement `query(text: &str, top_k: usize) -> Result<Vec<SearchResult>>`
- [x] Integration test: ingest 3 docs, query, assert correct chunk returned in top-3

---

## Phase 2 – Disk-Backed Index

- [x] Implement memory-mapped HNSW storage (write graph edges + vectors to `.bin` file)
- [x] Add `memmap2` crate for OS-managed paging
- [x] Implement `HnswIndex::flush()` and `HnswIndex::load_from_disk(path)`
- [x] Add `RwLock<HnswIndex>` for concurrent reads / exclusive writes
- [ ] Benchmark: query latency < 10 ms at 1M vectors on CI machine (not started; needs a dedicated machine, see 6E)
- [x] Implement semantic chunking: split on sentence boundaries, merge short chunks
- [x] Integration test: ingest 10k chunks, restart process, confirm persistence

---

## Phase 3 – FFI & Python Wrapper

### C ABI (`tpt-rag-ffi`)

- [x] Define opaque handle `TptRagHandle` type (named `TptRagHandle` as implemented, matching the README)
- [x] Implement `extern "C"` functions: `tpt_rag_create`, `tpt_rag_destroy`
- [x] Implement `tpt_rag_add_file`, `tpt_rag_add_text`
- [x] Implement `tpt_rag_query` returning `TptRagResult*` array + count
- [x] Implement `tpt_rag_free_results`
- [x] Define error code enum (`TptRagOk`, `TptRagErrInvalidArg`, `TptRagErr*`, as implemented)
- [x] Generate `tpt_raglite.h` C header via `cbindgen`
- [x] C integration test: compile and link against `.so`/`.dll`, run add+query (`tpt-rag-ffi/tests/c_integration.c`, run by `tpt-rag-ffi/tests/run_c_integration.sh`; passes on Linux, CI job added)

### Python Wrapper (`tpt-rag-py`)

- [x] Add `pyo3` dependency with `extension-module` feature
- [x] Configure `pyproject.toml` for `maturin` build backend
- [x] Implement `PyRAGLite` class exposing `__init__`, `add_files`, `add_text`, `query`
- [x] Implement `PySearchResult` with `score`, `text`, `source`, `tags` fields
- [x] Python integration test: `from tpt_raglite import RAGLite` → ingest PDF → query (`tpt-rag-py/tests/test_raglite.py`, 14 tests pass on Linux)
- [ ] Verify wheel builds on Windows, macOS, Linux (Linux built and tested locally; CI matrix added for all three OSes, first run pending)

### CI: PyPI Publishing

- [x] Create `.github/workflows/release-python.yml`
- [x] Matrix: `ubuntu-latest`, `windows-latest`, `macos-latest` x Python 3.9–3.12
- [x] Use `maturin publish` on tagged release (`v*`)
- [x] Add `--interpreter` flag for all target Python versions
- [x] Test: dry-run publish to TestPyPI on PR

---

## Phase 4 – WebAssembly / JS Wrapper

- [x] Add `wasm-bindgen` dependency to `tpt-rag-wasm`
- [x] Configure `Cargo.toml` with `crate-type = ["cdylib"]`
- [x] Implement `RAGLite` JS class with async `init(db_name: string)`
- [x] Implement `addText(text: string, metadata: object): Promise<void>`
- [x] Implement `query(text: string, top_k: number): Promise<SearchResult[]>`
- [x] Browser storage: the engine exports and imports its whole database as bytes (xport / TptRag.import), and the example keeps them in IndexedDB. OPFS is not written; IndexedDB is enough for this size
- [x] IndexedDB persistence in xamples/wasm_browser (guarded: when storage is blocked, the page works without saving)
- [x] Build: `wasm-pack build --target web` and `--target bundler` (the release workflow uses bundler)
- [x] Verify WASM binary < 15 MB: 937 KB after release build and wasm-opt
- [x] JS tests: `wasm-pack test --node` (3 tests on the real module, in CI) and `tpt-rag-wasm/tests/e2e/check.mjs` (downloads the model; run by hand)
- [x] `package.json`: generated by wasm-pack for the chosen target (ESM only; no CJS build)

### CI: npm Publishing

- [x] Create `.github/workflows/release-wasm.yml`
- [x] Build WASM on tagged release, publish to npm with `npm publish`
- [x] Add `--provenance` flag for npm audit trail

---

## Phase 5 – Advanced Features

### Cross-Encoder Reranking

- [x] Select small cross-encoder ONNX model: `ms-marco-MiniLM-L-6-v2` (quantized export, pinned in `scripts/fetch_model`)
- [x] Implement `Reranker` struct wrapping secondary `ort::Session` (cross-encoder: query and passage as one sequence)
- [x] Gate behind feature flag: `#[cfg(feature = "reranking")]`
- [x] Add `query_reranked(text, top_k, rerank_top_n)` to `RAGLite`
- [ ] Benchmark: measure latency budget for reranking step (not started; see 6E)

### Custom Embedding Models

- [x] Add `RAGLite::with_model(path: &Path)` constructor override (`open_with_model`, `open_with_config`)
- [x] Validate model outputs: the vector size is probed at load, outputs must be float32 with 2 or 3 dimensions and match the batch shape, and each vector must match the dimension stored in the database
- [x] Document model format requirements in README

### Hybrid Search (BM25 + Vector)

- [x] Enable SQLite FTS5 extension and add BM25 full-text index on `chunks.text`
- [x] Implement `bm25_search(query, top_k) -> Vec<ScoredChunk>`
- [x] Implement Reciprocal Rank Fusion (RRF) to merge BM25 and vector results
- [x] Expose `query_hybrid(text, top_k, alpha: f32)` on `RAGLite`
- [ ] Benchmark recall vs. pure-vector on a document benchmark dataset (not started; needs a labelled dataset)

### CI: Basic CI

- [x] Create `.github/workflows/ci.yml`
- [x] Run `cargo check`, `cargo test --lib`, `cargo clippy`, `cargo fmt --check`

---

## Phase 6 – Review Findings (2026-10-28)

> Source: static code review. Items marked [x] in earlier phases that don't match the code are re-opened here.
>
> Progress note (2026-10-28): the core crate is fully documented, `raglite eval` reports nDCG, Python `query` takes `since`, the notes template is added, and the Docker and HTML parse-error items are decided. Remaining open: WASM, benchmarks, and the 6H features (serve/MCP, watch, MMR, neighbour expansion, citations with offsets, structure-aware chunking).
>
> Earlier note: Phases 6A–6G are done except where marked open. Phase 4 (WebAssembly) is blocked on ONNX Runtime having no wasm32 build, and most of 6H is still open (the CLI has ingest, query, stats, doctor, delete, rebuild and eval, but not serve, watch or MCP). The benchmarks need a dedicated machine. On the Windows dev machine the test binaries cannot be linked (the prebuilt ONNX Runtime needs a newer MSVC standard library than the installed toolset provides), so the test suites were run on Linux (WSL). The Windows and macOS CI legs have not been run yet.

### 6A – Critical Correctness

- [x] **Tokenizer**: replace hash-based `tokenize` with real WordPiece (`tokenizers` crate or `vocab.txt`); `embedding.rs:124-182`
- [x] **Tokenizer**: add test comparing token IDs to known HuggingFace output for MiniLM (bert-base-uncased reference ids; skips when `models/vocab.txt` is absent)
- [x] **Chunker**: remove `break` that drops the rest of the document when a sentence boundary lands before window end; `chunker.rs:66-70`
- [x] **Chunker**: validate `overlap < window_size` (underflow panic); fix `word_offset` (quadratic, ignores whitespace-leading text)
- [x] **Chunker**: test that a 5k-word doc round-trips with no dropped text
- [x] **Chunker**: abbreviation-aware sentence detection ("Dr.", "U.S."); preserve newlines instead of `join(" ")`
- [x] **HNSW**: insert node into every layer `0..=level`, not just `layers[level]`; `hnsw.rs:87-91`
- [x] **HNSW**: fix results-heap polarity (max-heap for results, min-heap for candidates)
- [x] **HNSW**: prune neighbor lists to M / 2M; add select-neighbors heuristic
- [x] **HNSW**: O(1) node lookup (index by id) instead of linear scans; avoid cloning neighbor Vecs
- [x] **HNSW**: `random_level` use mL = 1/ln(M)
- [x] **HNSW**: raise recall test threshold to 0.95 with realistic M/ef (current test uses M=4, 0.8)
- [x] **HNSW**: replace `assert_eq!` dimension checks with `Result` errors (no panics through FFI)
- [ ] **Persistence**: auto-flush on `Drop` and after N writes; expose `flush` in FFI/Python/WASM (core, FFI and Python done: `flush_every`, drop flush, `tpt_rag_flush`, `RAGLite.flush`; WASM pending)
- [x] **Persistence**: atomic flush (write temp + fsync + rename); add magic number, version, checksum
- [x] **Persistence**: validate counts/IDs on load (no huge `with_capacity`, no out-of-range neighbor ids)
- [x] **Persistence**: fix `vector_id` collision after restart: ids come from the index, which is rebuilt from SQLite when it does not match; the index file carries an epoch that the metadata records, so an interrupted rebuild is detected
- [x] **Persistence**: store model id + dimension in SQLite (`meta` table); detect mismatch (refused unless `reembed_on_model_change`); rebuild HNSW from SQLite when index is missing/corrupt/out of step
- [x] **Ingest**: wrap SQLite writes in a transaction; roll back on failure so index and metadata stay consistent (replace is one transaction; a failed write hides the new vectors)
- [x] **BM25**: build tokenized OR query instead of one quoted phrase; `metadata.rs:105`
- [ ] **Delete**: fix `delete_by_source` (only remove matching FTS rows, use FTS5 delete protocol); tombstone HNSW vectors; expose `delete_document` in all APIs (core, FFI and Python done; WASM pending)
- [x] **Re-ingest**: content-hash / unique `source_path`; upsert instead of duplicating chunks

### 6B – Embedding & Reranking

- [x] L2-normalize embeddings; validate output dim against `EMBEDDING_DIM`
- [x] Dynamic padding (don't pad every input to 512); handle models without `token_type_ids`
- [x] True batched inference in `embed_batch` (batches of 32 with attention masking; rayon is not used)
- [x] Real cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`, pair encoding, sigmoid scores); load model once; load errors are returned, not swallowed by `.ok()`
- [x] Rerank a candidate pool larger than `top_k` (`max(rerank_top_n, top_k)`)
- [x] Normalize scores across `query` / `query_hybrid` / `query_reranked`: cosine, fused score scaled to 0–1, sigmoid; documented in `SearchResult`; `alpha` clamped and NaN defaults to 0.5; `top_k * 2` saturates

### 6C – Bindings

- [x] **FFI**: wrap entry points in `catch_unwind`
- [x] **FFI**: free result array via `into_boxed_slice` (no capacity assumption); null pointer on zero results
- [x] **FFI**: add `tpt_rag_last_error`, `tpt_rag_version`, flush/hybrid/rerank/delete functions; stop swallowing NUL-byte strings
- [x] **FFI**: generate C-compatible header (add `cbindgen.toml`, `language = "C"`); stop writing into source tree in `build.rs`
- [x] **FFI**: fix README C example (names match the header; the example is `examples/c_basic.c`, which CI compiles and runs); add C integration test
- [x] **Python**: fix `pyproject.toml` (`python-source` dir exists; `module-name` matches the package); `maturin develop` works on Linux
- [x] **Python**: typed exceptions (`RAGLiteError`, `ModelError`, `UnsupportedFormatError`, `EmptyDocumentError`) instead of `IOError`; GIL released during heavy calls; flush/hybrid/rerank/delete/stats added; license is the expression `MIT OR Apache-2.0`
- [x] **Python**: integration test (ingest PDF, query)
- [x] **WASM** (reversed 2026-10-29 at the user's request): built as a browser engine. The host computes embeddings, because ONNX Runtime has no wasm32 build; the browser engine has its own index instead of SQLite; and the database is exported as bytes instead of living on a filesystem. See the README's Browser section. The blockers that were recorded: (1) no wasm32 ONNX Runtime, (2) bundled SQLite needs a C standard library for wasm32-unknown-unknown, (3) storage assumed a filesystem
- [x] **WASM**: the core builds for wasm32 without its native features (`--no-default-features`, with `getrandom` using the browser's crypto). ONNX Runtime, SQLite, the filesystem database and PDF parsing sit behind the `native` feature
- [x] **WASM**: `wasm` CI job builds the core for wasm32, builds the package and runs the Node tests

### 6D – Parsers

- [x] HTML: strip `<script>`, `<style>`, `<noscript>`; parse errors are not surfaced (decided 2026-10-28: html5ever repairs malformed markup, so a bad page still yields its text. Repairs are logged at debug level)
- [x] Markdown: use a real parser (pulldown-cmark) instead of char stripping
- [x] PDF: `catch_unwind` around `pdf_extract`; PDF test fixtures (`tpt-rag-core/tests/fixtures/`, generated by `make_pdf_fixtures.py`); report empty (scanned) PDFs as `EmptyDocument`
- [x] Text: handle BOM / non-UTF-8 encodings; add max file size limit
- [x] Parser registry (custom parsers via `register_parser`; a later registration takes precedence for the same extension)
- [x] New parsers: DOCX, CSV, JSON, and source code
- [x] Directory ingestion with progress callback (recursive, in Rust; glob patterns not done)

### 6E – Tests & CI

- [x] Replace placeholder `tests/integration_test.rs` and move it into a real package (`tpt-rag-core/tests/`)
- [x] Integration: ingest 3 docs, query top-3 returns correct source
- [x] Integration: persistence across reopen; 10k chunks, restart, query
- [x] Fix metadata tests sharing a fixed temp dir (use `tempfile`)
- [x] CI: run full `cargo test` (not only `--lib`); matrix ubuntu/windows/macos; model is fetched first so the integration tests run
- [x] CI: ffi, python (maturin) and wasm32 build jobs
- [x] Commit `Cargo.lock`; decide on pinning `ort` RC (`Cargo.lock` is no longer ignored, so committing it pins the release candidate; `ort` stays on `2.0.0-rc.13`). The committed lock was stale (no `zip` or `glob` entries), so it has been regenerated. The commit itself is still to be made
- [ ] `criterion` benchmarks (insert, search, embed); publish numbers; remove unproven "<10ms @ 1M" and "<100MB" claims from README until verified (the README never contained those numbers; benchmarks not started)

### 6F – Config & Observability

- [ ] `RagConfig` struct/builder (M, ef_construction, ef_search, `ChunkConfig`, reranker options) exposed in all bindings (core, FFI and Python done; WASM pending)
- [x] Env override for model path (`RAGLITE_MODEL`, `RAGLITE_RERANKER`); the current working directory is no longer searched by `find_model`
- [x] Add `tracing` logging (model load, index load and rebuild, ingest, flush, errors); the library does not install a subscriber
- [x] `stats()` API (chunks, vectors, index size, model id)
- [x] `Drop`/`close` semantics documented (`RAGLite` docs: writes are flushed on drop; drop cannot report errors, so call `flush()`)

### 6G – Adoption: Setup, Examples, Templates

- [x] `scripts/fetch_model.sh` / `.ps1` (SHA-256 checked download into `models/`)
- [x] Optional `embedded-model` cargo feature (`include_bytes!`; needs the model files in `models/` at build time)
- [x] `examples/rust_basic`, `rust_hybrid_rerank` (in `tpt-rag-core/examples/`, run with cargo)
- [x] `examples/python_quickstart.py`, `python_pdf_qa.py`
- [x] `examples/c_basic.c`
- [x] `examples/wasm_browser/`: index, keyword and vector search in the browser, with transformers.js embeddings
- [ ] `templates/docs-chatbot` (folder in, Q&A out, citation-aware prompt builder)
- [x] `templates/local-notes-search` (folder sync by mtime, removes deleted notes, search CLI; tests use a fake DB)
- [x] `Dockerfile` + `docker-compose.yml`: dropped (decided 2026-10-28). The README says there is no Docker, and a container build cannot be checked on this machine.
- [x] `justfile` / Makefile targets: `setup`, `test`, `demo` (also `ffi-test`, `python-test`, `header`)
- [x] README rewrite: status, quickstart for Rust/Python/C, search methods and score ranges, configuration, persistence, formats, model requirements, install from source, testing, troubleshooting
- [x] CHANGELOG.md, CONTRIBUTING.md, and rustdoc on public items (every public item in `tpt-rag-core` now has docs; `cargo doc` with `-W missing_docs` is clean)
- [x] Only advertise `pip install` / `npm install` after first publish (README says the package is not published and gives build-from-source steps)

### 6H – Innovation (Post-Stabilization)

- [x] `raglite` CLI: `ingest`, `query`, `stats`, `eval`, `doctor`, `delete`, `rebuild`, `sync`, `watch`, `serve`, `synth` and `models pull` done
- [x] `raglite doctor`: check model, ONNX runtime, index/SQLite consistency (`RAGLite::check`)
- [x] `raglite eval`: JSONL of (query, expected source) -> recall@k, MRR, nDCG. `raglite synth` makes cases from the chunks themselves (a few words of a chunk, expected its source); LLM-written questions can be added in the same JSONL format
- [x] Watch mode: `raglite watch` and `sync_directory` poll the folder and re-index by content hash (polling, not OS file events)
- [x] MCP server (`raglite serve --mcp`, over stdio) and small HTTP API (`raglite serve`)
- [x] Metadata/tag/source/time filters in search (`query_filtered`, `QueryFilter`; CLI `--tag`, `--source-prefix`, `--since`)
- [x] MMR diversification (`query_diverse`, CLI `--diverse`, Python `query_diverse`)
- [x] Neighbor-chunk context expansion (`neighbors`, Python `neighbors`; ordered by byte offsets)
- [x] Citation-ready results (chunk id, ingestion time, byte offsets, Markdown heading path and PDF page numbers)
- [x] Structure-aware chunking: Markdown headings, Markdown code block edges and PDF pages bound chunks (decision 2026-10-29: SemanticChunker stays unwired. The heading, page and code-block boundaries cover what it was for; the type remains public and unused, and can be removed in a breaking release)
- [x] Schema versioning and migrations: an ordered list of migrations (`MIGRATIONS` in `metadata.rs`), each run in a transaction with its version bump; failed steps roll back; newer databases are refused with `SchemaTooNew` before anything changes
