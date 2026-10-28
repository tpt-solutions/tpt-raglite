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
- [ ] Download quantized INT8 `all-MiniLM-L6-v2` ONNX model (see `models/` directory)
- [ ] Embed model into binary via `include_bytes!` build script
- [x] Implement `EmbeddingModel` struct wrapping `ort::session::Session`
- [x] Implement batch embedding using `rayon` for parallelism
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
- [ ] Benchmark: query latency < 10 ms at 1M vectors on CI machine
- [x] Implement semantic chunking: split on sentence boundaries, merge short chunks
- [x] Integration test: ingest 10k chunks, restart process, confirm persistence

---

## Phase 3 – FFI & Python Wrapper

### C ABI (`tpt-rag-ffi`)

- [x] Define opaque handle `tpt_rag_handle` type
- [x] Implement `extern "C"` functions: `tpt_rag_create`, `tpt_rag_destroy`
- [x] Implement `tpt_rag_add_file`, `tpt_rag_add_text`
- [x] Implement `tpt_rag_query` returning `tpt_rag_result*` array + count
- [x] Implement `tpt_rag_free_results`
- [x] Define error code enum (`TPT_RAG_OK`, `TPT_RAG_ERR_*`)
- [x] Generate `tpt_raglite.h` C header via `cbindgen`
- [ ] C integration test: compile and link against `.so`/`.dll`, run add+query

### Python Wrapper (`tpt-rag-py`)

- [x] Add `pyo3` dependency with `extension-module` feature
- [x] Configure `pyproject.toml` for `maturin` build backend
- [x] Implement `PyRAGLite` class exposing `__init__`, `add_files`, `add_text`, `query`
- [x] Implement `PySearchResult` with `score`, `text`, `source`, `tags` fields
- [ ] Python integration test: `from tpt_raglite import RAGLite` → ingest PDF → query
- [ ] Verify wheel builds on Windows, macOS, Linux

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
- [ ] Implement browser storage backend: Origin Private File System (OPFS) via `web_sys`
- [ ] Implement IndexedDB fallback for browsers without OPFS
- [ ] Build with `wasm-pack build --target bundler`
- [ ] Verify WASM binary < 15 MB (strip debug symbols, `wasm-opt -Oz`)
- [ ] JS integration test (Playwright or Vitest): load WASM in browser, ingest text, query
- [ ] Write `package.json` with proper ESM/CJS exports

### CI: npm Publishing

- [x] Create `.github/workflows/release-wasm.yml`
- [x] Build WASM on tagged release, publish to npm with `npm publish`
- [x] Add `--provenance` flag for npm audit trail

---

## Phase 5 – Advanced Features

### Cross-Encoder Reranking

- [ ] Select small cross-encoder ONNX model (e.g., `ms-marco-MiniLM-L-6-v2`)
- [x] Implement `Reranker` struct wrapping secondary `ort::Session`
- [x] Gate behind feature flag: `#[cfg(feature = "reranking")]`
- [x] Add `query_reranked(text, top_k, rerank_top_n)` to `RAGLite`
- [ ] Benchmark: measure latency budget for reranking step

### Custom Embedding Models

- [x] Add `RAGLite::with_model(path: &Path)` constructor override
- [ ] Validate model outputs: assert output tensor is 1D float32 of expected dims
- [ ] Document model format requirements in README

### Hybrid Search (BM25 + Vector)

- [x] Enable SQLite FTS5 extension and add BM25 full-text index on `chunks.text`
- [x] Implement `bm25_search(query, top_k) -> Vec<ScoredChunk>`
- [x] Implement Reciprocal Rank Fusion (RRF) to merge BM25 and vector results
- [x] Expose `query_hybrid(text, top_k, alpha: f32)` on `RAGLite`
- [ ] Benchmark recall vs. pure-vector on a document benchmark dataset

### CI: Basic CI

- [x] Create `.github/workflows/ci.yml`
- [x] Run `cargo check`, `cargo test --lib`, `cargo clippy`, `cargo fmt --check`
