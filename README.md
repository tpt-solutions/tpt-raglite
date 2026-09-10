# tpt-raglite

**The SQLite for AI** — Blazing-fast, embeddable, zero-configuration RAG engine in pure Rust.

## Overview

`tpt-raglite` replaces the fragmented vector-DB + embedding-server + chunking-framework stack with a single, statically compiled library. It handles document parsing, local ONNX-based embeddings, smart chunking, and disk-backed vector indexing — entirely offline, with zero external infrastructure.

## Features

- **Zero-config & offline-first** — No Docker, no API keys, no cloud dependencies
- **Embeddable, not a server** — Runs in the same process as your application
- **Multi-language** — Rust core with C ABI for Python (`pip install`), JavaScript (`npm install`), and C++
- **Memory-efficient** — Memory-mapped HNSW index handles millions of vectors
- **Fast** — Query latency < 10ms at 1M vectors on modern CPUs

## Quick Start

### Rust

```rust
use tpt_rag_core::RAGLite;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rag = RAGLite::open(Path::new("./my_db"))?;

    // Add a document
    rag.add_document(Path::new("manual.pdf"), &["docs"])?;

    // Add text directly
    rag.add_text("Some important information.", &Default::default())?;

    // Query
    let results = rag.query("How do I configure the engine?", 5)?;
    for r in &results {
        println!("[{:.2}] {}", r.score, r.text);
    }
    Ok(())
}
```

### Python

```python
from tpt_raglite import RAGLite

rag = RAGLite("./my_db")
rag.add_files(["manual.pdf", "notes.md"], tags=["docs"])
results = rag.query("How do I configure the engine?")
for r in results:
    print(f"[{r.score:.2f}] {r.text}")
```

### C

```c
#include "tpt_raglite.h"

int main() {
    tpt_rag_handle* rag = tpt_rag_create("./my_c_db");
    tpt_rag_add_file(rag, "manual.pdf");

    tpt_rag_result* results = NULL;
    int count = tpt_rag_query(rag, "How to configure?", 5, &results);

    // Process results...

    tpt_rag_free_results(results, count);
    tpt_rag_destroy(rag);
    return 0;
}
```

## Installation

### From source (Rust)

```bash
cargo build --release
```

### Python

```bash
pip install tpt-raglite
```

### JavaScript / WASM

```bash
npm install tpt-raglite-wasm
```

## Supported Formats

| Format | Extension |
|--------|-----------|
| Plain text | `.txt` |
| Markdown | `.md`, `.markdown`, `.mdown` |
| HTML | `.html`, `.htm` |
| PDF | `.pdf` |

## Architecture

```
tpt-rag-core/     # Pure Rust: parsing, chunking, ONNX embedding, HNSW index, SQLite metadata
tpt-rag-ffi/      # C ABI boundary with opaque handles
tpt-rag-py/       # PyO3 Python wrapper
tpt-rag-wasm/     # wasm-bindgen JS/Browser wrapper
```

## Requirements

- **Models:** Place a quantized `all-MiniLM-L6-v2.onnx` model in the `models/` directory or alongside your database path.
- The model provides 384-dimensional embeddings with < 100MB RAM usage.

## License

MIT OR Apache-2.0 — (c) TPT Solutions
