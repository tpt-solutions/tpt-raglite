# Changelog

All notable changes to the `tpt-raglite` Python package (`tpt-rag-py`) are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0.

The repository-wide changelog is at [`../CHANGELOG.md`](../CHANGELOG.md). The engine's changes are listed in [`../tpt-rag-core/CHANGELOG.md`](../tpt-rag-core/CHANGELOG.md).

## [Unreleased]

Nothing is published yet, and the package is not on PyPI. The first release will be 0.1.0. Build from source with `maturin develop --release`.

### Added

- **`RAGLite` class.** Opens or creates a database. Keyword arguments set the model paths, `reembed_on_model_change`, and the index and chunking settings (`m`, `ef_construction`, `ef_search`, `window_size`, `overlap`, `flush_every`).
- **Context manager.** `with RAGLite(...) as db:` flushes the index on exit, and a flush failure is raised.
- **Ingestion.**
  - `add_files`, `add_directory` (returns the added and failed files), and `add_glob`.
  - `sync_directory` (returns the indexed, unchanged, removed, and failed files).
  - `add_text` with an optional source label and tags.
- **Search.**
  - `query` takes `tags`, `source_prefix`, and `since` filters.
  - `query_hybrid` takes `alpha`.
  - `query_diverse` takes `relevance_weight`.
  - `query_reranked` takes `rerank_top_n`.
  - `neighbors` returns the chunks around a hit.
- **`SearchResult`.** Exposes `score`, `text`, `source`, `tags`, `chunk_id`, `created_at`, `start_offset`, `end_offset`, and `section`.
- **Maintenance.** `delete_document`, `flush`, `rebuild_index`, and `stats` (a dict with `chunks`, `vectors`, `index_bytes`, and `model`).
- **Exceptions.** `RAGLiteError` is the base class. `ModelError`, `UnsupportedFormatError`, and `EmptyDocumentError` are subclasses. File errors raise `OSError`, and invalid settings raise `ValueError`.
- **Threads.** Heavy calls release the GIL. A `RAGLite` object can be shared between threads, with calls serialized by an internal lock.
- **Tests.** `tpt-rag-py/tests/test_raglite.py`. Tests that need the model are skipped when it is missing.
- **Examples.** `examples/python_quickstart.py` and `examples/python_pdf_qa.py`.

### Changed

- **License packaging.** The license is declared as the expression `MIT OR Apache-2.0`. The separate license classifiers were removed, because listing both licenses would contradict the expression.
- **Metadata.** The package has a `readme` and `keywords`, so PyPI shows this README.

### Not yet done

- Wheels for PyPI, and a CI build for each platform.
- An async API.
- A typed stub file (`.pyi`) for editor completion and type checkers.
