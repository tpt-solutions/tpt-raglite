# Changelog

All notable changes to `tpt-rag-ffi` are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0.

The repository-wide changelog is at [`../CHANGELOG.md`](../CHANGELOG.md). The C header is generated from this crate, so ABI changes are listed here.

## [Unreleased]

Nothing is published yet. The first release will be 0.1.0.

### Added

- **C ABI** built as a shared library (`cdylib`) and a static library (`staticlib`).
- **Generated header.** `tpt_raglite.h` is produced by cbindgen from `build.rs` and `cbindgen.toml`, and a copy is committed at the repository root.
- **Handles and creation.** `tpt_rag_create`, `tpt_rag_create_with_config`, and `tpt_rag_destroy`. `tpt_rag_default_config` returns the settings as a `TptRagConfig` struct to start from.
- **Ingestion.** `tpt_rag_add_file` (with tags) and `tpt_rag_add_text`.
- **Search.** `tpt_rag_query` (vector), `tpt_rag_query_hybrid` (with `alpha`), and `tpt_rag_query_reranked` (with `rerank_top_n`).
- **Maintenance.** `tpt_rag_delete_document` and `tpt_rag_flush`.
- **Errors.** `TptRagError` codes (`TptRagOk` is 0), and `tpt_rag_last_error` for a per-thread message.
- **Memory.** `tpt_rag_free_results` releases a result array, and `tpt_rag_version` reports the library version.
- **Tests.** `tpt-rag-ffi/tests/c_integration.c`, run by `tpt-rag-ffi/tests/run_c_integration.sh` with `-Werror`. CI compiles and runs `examples/c_basic.c`.

### Fixed

- **Panics are caught** at every entry point and returned as `TptRagErrInternal`. A Rust panic never unwinds into C.
- **NUL bytes** in a string keep their text instead of truncating it.
- **Results are freed** with the allocator that created them.

### Not yet done

- No FFI for `neighbors`, `query_diverse`, `query_filtered`, folder sync (`sync_directory`), or `stats`.
- No `add_directory` in C; ingest files one at a time.
- The C tests run on Linux only.
