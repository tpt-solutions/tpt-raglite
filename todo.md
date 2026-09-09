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
- [ ] Initialize git repository

---

## Phase 1 – Core Engine (Pure Rust)

### Parsing

- [ ] Add `pdf-extract` dependency for PDF text extraction
- [ ] Add `markup5ever` / `html5ever` for HTML parsing
- [ ] Implement `Parser` trait with implementations: `PlainText`, `Markdown`, `Html`, `Pdf`
- [ ] Unit test: parse each format, assert text extracted correctly

### Chunking

- [ ] Implement sliding-window chunker (512-token window, 50-token overlap)
- [ ] Implement sentence-boundary detection for semantic chunk splitting
- [ ] Expose `ChunkConfig` struct (configurable window/overlap)
- [ ] Unit tests: verify chunk sizes, overlap correctness, no mid-sentence splits

### Local ONNX Embedding

- [ ] Add `ort` crate (ONNX Runtime bindings)
- [ ] Download quantized INT8 `all-MiniLM-L6-v2` ONNX model
- [ ] Embed model into binary via `include_bytes!` build script
- [ ] Implement `EmbeddingModel` struct wrapping `ort::Session`
- [ ] Implement batch embedding using `rayon` for parallelism
- [ ] Unit test: embed a known sentence, assert 384-dim output, cosine similarity sanity check

### In-Memory HNSW Index (custom Rust)

- [ ] Define `HnswIndex` struct with configurable `M` and `ef_construction` params
- [ ] Implement node insertion with layered graph construction
- [ ] Implement KNN search (`ef_search` parameter)
- [ ] Implement cosine distance metric
- [ ] Unit tests: insert N vectors, query nearest neighbor, assert recall > 0.95 on synthetic data

### SQLite Metadata Store

- [ ] Add `rusqlite` dependency
- [ ] Implement `MetadataStore` struct managing a local `.sqlite3` file
- [ ] Schema: `chunks(id, vector_id, text, source_path, created_at, tags_json)`
- [ ] Implement `insert_chunk`, `get_chunk_by_vector_id`, `delete_by_source`
- [ ] Unit tests: insert/query/delete round-trip

### RAGLite Core API (`tpt-rag-core`)

- [ ] Define `RAGLite` struct holding `HnswIndex` + `MetadataStore` + `EmbeddingModel`
- [ ] Implement `RAGLite::open(path: &Path) -> Result<Self>`
- [ ] Implement `add_document(path: &Path, tags: &[&str]) -> Result<()>`
- [ ] Implement `add_text(text: &str, metadata: &Metadata) -> Result<()>`
- [ ] Implement `query(text: &str, top_k: usize) -> Result<Vec<SearchResult>>`
- [ ] Integration test: ingest 3 docs, query, assert correct chunk returned in top-3

---

## Phase 2 – Disk-Backed Index

- [ ] Implement memory-mapped HNSW storage (write graph edges + vectors to `.bin` file)
- [ ] Add `memmap2` crate for OS-managed paging
- [ ] Implement `HnswIndex::flush()` and `HnswIndex::load_from_disk(path)`
- [ ] Add `RwLock<HnswIndex>` for concurrent reads / exclusive writes
- [ ] Benchmark: query latency < 10 ms at 1M vectors on CI machine
- [ ] Implement semantic chunking: split on sentence boundaries, merge short chunks
- [ ] Integration test: ingest 10k chunks, restart process, confirm persistence

---

## Phase 3 – FFI & Python Wrapper

### C ABI (`tpt-rag-ffi`)

- [ ] Define opaque handle `tpt_rag_handle` type
- [ ] Implement `extern "C"` functions: `tpt_rag_create`, `tpt_rag_destroy`
- [ ] Implement `tpt_rag_add_file`, `tpt_rag_add_text`
- [ ] Implement `tpt_rag_query` returning `tpt_rag_result*` array + count
- [ ] Implement `tpt_rag_free_results`
- [ ] Define error code enum (`TPT_RAG_OK`, `TPT_RAG_ERR_*`)
- [ ] Generate `tpt_raglite.h` C header via `cbindgen`
- [ ] C integration test: compile and link against `.so`/`.dll`, run add+query

### Python Wrapper (`tpt-rag-py`)

- [ ] Add `pyo3` dependency with `extension-module` feature
- [ ] Configure `pyproject.toml` for `maturin` build backend
- [ ] Implement `PyRAGLite` class exposing `__init__`, `add_files`, `add_text`, `query`
- [ ] Implement `PySearchResult` with `score`, `text`, `source`, `tags` fields
- [ ] Python integration test: `from tpt_raglite import RAGLite` → ingest PDF → query
- [ ] Verify wheel builds on Windows, macOS, Linux

### CI: PyPI Publishing

- [ ] Create `.github/workflows/release-python.yml`
- [ ] Matrix: `ubuntu-latest`, `windows-latest`, `macos-latest` × Python 3.9–3.12
- [ ] Use `maturin publish` on tagged release (`v*`)
- [ ] Add `--interpreter` flag for all target Python versions
- [ ] Test: dry-run publish to TestPyPI on PR

---

## Phase 4 – WebAssembly / JS Wrapper

- [ ] Add `wasm-bindgen` dependency to `tpt-rag-wasm`
- [ ] Configure `Cargo.toml` with `crate-type = ["cdylib"]`
- [ ] Implement `RAGLite` JS class with async `init(db_name: string)`
- [ ] Implement `addText(text: string, metadata: object): Promise<void>`
- [ ] Implement `query(text: string, top_k: number): Promise<SearchResult[]>`
- [ ] Implement browser storage backend: Origin Private File System (OPFS) via `web_sys`
- [ ] Implement IndexedDB fallback for browsers without OPFS
- [ ] Build with `wasm-pack build --target bundler`
- [ ] Verify WASM binary < 15 MB (strip debug symbols, `wasm-opt -Oz`)
- [ ] JS integration test (Playwright or Vitest): load WASM in browser, ingest text, query
- [ ] Write `package.json` with proper ESM/CJS exports

### CI: npm Publishing

- [ ] Create `.github/workflows/release-wasm.yml`
- [ ] Build WASM on tagged release, publish to npm with `npm publish`
- [ ] Add `--provenance` flag for npm audit trail

---

## Phase 5 – Advanced Features

### Cross-Encoder Reranking

- [ ] Select small cross-encoder ONNX model (e.g., `ms-marco-MiniLM-L-6-v2`)
- [ ] Implement `Reranker` struct wrapping secondary `ort::Session`
- [ ] Gate behind feature flag: `#[cfg(feature = "reranking")]`
- [ ] Add `query_reranked(text, top_k, rerank_top_n)` to `RAGLite`
- [ ] Benchmark: measure latency budget for reranking step

### Custom Embedding Models

- [ ] Add `RAGLite::with_model(path: &Path)` constructor override
- [ ] Validate model outputs: assert output tensor is 1D float32 of expected dims
- [ ] Document model format requirements in README

### Hybrid Search (BM25 + Vector)

- [ ] Enable SQLite FTS5 extension and add BM25 full-text index on `chunks.text`
- [ ] Implement `bm25_search(query, top_k) -> Vec<ScoredChunk>`
- [ ] Implement Reciprocal Rank Fusion (RRF) to merge BM25 and vector results
- [ ] Expose `query_hybrid(text, top_k, alpha: f32)` on `RAGLite`
- [ ] Benchmark recall vs. pure-vector on a document benchmark dataset
