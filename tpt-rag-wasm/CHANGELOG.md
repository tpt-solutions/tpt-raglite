# Changelog

All notable changes to `tpt-rag-wasm` are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0.

The repository-wide changelog is at [`../CHANGELOG.md`](../CHANGELOG.md). The engine's changes are listed in [`../tpt-rag-core/CHANGELOG.md`](../tpt-rag-core/CHANGELOG.md).

## [Unreleased]

Nothing is published yet. The first release will be 0.1.0. The crate is experimental.

### Added

- **`RAGLiteWasm` class** for JavaScript, built with wasm-bindgen. It has a constructor (`new RAGLiteWasm(name)`), `add_text(text)`, and `query(text, top_k)`.
- **`JsSearchResult`** with `score`, `text`, `source`, and `tags`. `query` returns these as a JSON array string.
- **Release workflow.** `.github/workflows/release-wasm.yml` builds the package with `wasm-pack` and publishes `tpt-raglite-wasm` to npm when a `v*` tag is pushed.

### Changed

- **Crate types** are `cdylib` and `rlib`, so the Rust side can depend on the crate as well as the JavaScript build.

### Not yet done

- **Browser storage.** The constructor still opens a database folder through the native file system. OPFS and IndexedDB storage are needed before the crate can run in a browser.
- **Build check.** The crate has not been verified to build for `wasm32-unknown-unknown` with the current dependencies.
- **JavaScript API.** Tags, source labels, filters, configuration, delete, and flush are not exposed. `query` returns a JSON string rather than an array.
- **Model input.** There is no way to pass model bytes from JavaScript.
