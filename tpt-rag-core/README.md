# tpt-rag-core

The Rust engine behind tpt-raglite: parsing, chunking, local ONNX embeddings, an HNSW vector index, and SQLite metadata, in one embeddable library. It also ships the `raglite` command-line tool.

A database is a folder. You can copy it, back it up, or delete it. Nothing leaves the process: no server, no API key, no network calls once the model files are on disk.

> **Status: pre-release (0.1.0).** Not yet published to crates.io. Depend on it by path or git (see [Install](#install)).

## Contents

- [What it does](#what-it-does)
- [Install](#install)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Search methods](#search-methods)
- [Results](#results)
- [Configuration](#configuration)
- [Persistence and safety](#persistence-and-safety)
- [Supported formats](#supported-formats)
- [Custom parsers](#custom-parsers)
- [Errors](#errors)
- [Feature flags](#feature-flags)
- [The `raglite` command](#the-raglite-command)
- [Testing](#testing)
- [Limitations](#limitations)

## What it does

1. **Parses** files (text, Markdown, HTML, PDF, Word, CSV, JSON, source code, config files) or raw strings.
2. **Chunks** the text. Chunks never cross a Markdown heading, a PDF page, or the edge of a code block, so each chunk belongs to one section.
3. **Embeds** each chunk with `all-MiniLM-L6-v2` (384 dimensions) through ONNX Runtime. Vectors are batched and L2-normalized.
4. **Stores** the vectors in a disk-backed HNSW index and the text, tags, sources and offsets in SQLite.
5. **Searches** by vector similarity, hybrid vector + keyword ranking, cross-encoder reranking, or diverse (MMR) results.

## Install

Until the crate is published, add it from the repository:

```toml
[dependencies]
tpt-rag-core = { git = "https://github.com/TPTSolutions/tpt-raglite" }
```

or from a local checkout:

```toml
[dependencies]
tpt-rag-core = { path = "../tpt-raglite/tpt-rag-core" }
```

Requirements: stable Rust, and a C toolchain (SQLite is bundled). On Windows, use a current Visual Studio 2022 install, because the prebuilt ONNX Runtime binaries need a newer MSVC standard library than toolset 14.38 provides.

### Get the model

Embedding needs two files: an ONNX model and its `vocab.txt`. The repository's scripts download both, check their SHA-256 hashes, and place them in `models/`:

```bash
bash scripts/fetch_model.sh          # Windows: powershell -File scripts/fetch_model.ps1
```

The reranker (`ms-marco-MiniLM-L-6-v2.onnx`) is optional. Only `query_reranked` needs it.

Tell the library where the files are with `RAGLITE_MODEL` and `RAGLITE_RERANKER`, or place them in the database folder. The full lookup order is under [Configuration](#model-lookup-order).

## Quick start

```rust
use std::path::Path;
use tpt_rag_core::{Metadata, RAGLite};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rag = RAGLite::open(Path::new("./my_db"))?;

    rag.add_document(Path::new("manual.pdf"), &["docs"])?;
    rag.add_text(
        "Some important information.",
        &Metadata { source: Some("notes".into()), tags: vec![] },
    )?;

    for r in rag.query("How do I configure the engine?", 5)? {
        println!("[{:.2}] {}", r.score, r.text);
    }

    rag.flush()?; // write the index now, so a write error is visible here
    Ok(())
}
```

`RAGLite::open` creates the database if the folder is new. Writes are applied at once and visible to later queries in the same process.

## Examples

The examples below are grouped by task. Each is self-contained apart from the `use` lines, and assumes a `rag` opened as in the quick start.

### 1. Ingest a folder, then answer from it

`add_directory` walks a folder recursively. It reports what was added and what failed, and it calls back for each file, so you can show progress. One bad file does not stop the rest.

```rust
use std::path::Path;
use tpt_rag_core::RAGLite;

let rag = RAGLite::open(Path::new("./kb"))?;

let report = rag.add_directory(Path::new("./docs"), &["docs"], &mut |i, total, path| {
    println!("[{i}/{total}] {}", path.display());
})?;

println!("added {}, failed {}", report.added.len(), report.failed.len());
for (path, reason) in &report.failed {
    eprintln!("skipped {}: {reason}", path.display());
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

For files matching a pattern, use `add_glob("notes/**/*.md", &["notes"])`, which returns the same report.

### 2. Keep a folder in sync

`sync_directory` is for folders that change over time. It hashes each file, ingests the new and changed ones, skips the rest, and removes chunks for files that no longer exist. Only files that a sync has ingested are tracked for removal, so documents added one at a time are never deleted by it.

```rust
use std::{path::Path, thread, time::Duration};
use tpt_rag_core::RAGLite;

let rag = RAGLite::open(Path::new("./kb"))?;

loop {
    let report = rag.sync_directory(Path::new("./notes"), &["notes"])?;
    println!(
        "indexed {}, unchanged {}, removed {}, failed {}",
        report.indexed.len(),
        report.unchanged,
        report.removed.len(),
        report.failed.len(),
    );
    rag.flush()?;
    thread::sleep(Duration::from_secs(30));
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Polling is used instead of file system events, so the same code works on every platform. `raglite watch` is this loop, ready to run.

### 3. Filter results by tag, source or date

The filter is applied to the best `10 × top_k` vector matches, so a very selective filter can return fewer than `top_k` results. Widen `top_k` if that happens.

```rust
use tpt_rag_core::QueryFilter;

let filter = QueryFilter {
    tags: vec!["ops".into(), "prod".into()], // a chunk must carry every tag listed
    source_prefix: Some("runbooks/".into()),  // and its source must start with this
    created_since: Some(1_735_689_600),       // and it must be ingested at or after this Unix time
};

for r in rag.query_filtered("how do I rotate the key", 5, &filter)? {
    println!("[{:.2}] {} ({})", r.score, r.text, r.source);
}
```

### 4. Cite passages with their location

Every result carries where it came from: the source, the section (for Markdown, the heading path such as `Setup > Backups`; for PDFs, the page), and the byte range in the extracted text. Use these to build answers that point back to the source.

```rust
for r in rag.query("when do backups run", 3)? {
    let location = match (&r.section, r.start_offset, r.end_offset) {
        (Some(section), Some(start), Some(end)) => {
            format!("{} › {section} (bytes {start}-{end})", r.source)
        }
        (None, Some(start), Some(end)) => format!("{} (bytes {start}-{end})", r.source),
        _ => r.source.clone(),
    };
    println!("[{:.2}] {location}\n    {}", r.score, r.text);
}
```

For plain text and `add_text`, `text[start..end]` gives back the chunk exactly. Offsets count bytes, so slice the UTF-8 bytes, not characters. For PDFs, Markdown and HTML the offsets refer to the extracted text, not the original file.

### 5. Expand a hit into surrounding context

A chunk on its own can start or end mid-thought. `neighbors` returns the chunks before and after a hit in the same document, in reading order. The hit itself is not included, so keep its score.

```rust
let hits = rag.query("retention period for logs", 1)?;
if let Some(hit) = hits.first() {
    let before_and_after = rag.neighbors(hit.chunk_id, 1)?; // one chunk each side

    println!("hit [{:.2}]: {}", hit.score, hit.text);
    for n in &before_and_after {
        println!("context: {}", n.text);
    }
}
```

Neighbours have a score of 0. Chunks stored before schema version 2 have no byte offsets, so they cannot be ordered and have no neighbours.

### 6. Compare ranking methods on one question

Scores are not comparable across methods, so compare results within one method. The repository has a runnable version of this: [`examples/rust_hybrid_rerank.rs`](examples/rust_hybrid_rerank.rs).

```rust
println!("vector:");
for r in rag.query(question, 5)? {
    println!("  [{:.3}] {}", r.score, r.source);
}

// alpha is the vector weight: 1.0 is vectors only, 0.0 is keywords only.
println!("hybrid:");
for r in rag.query_hybrid(question, 5, 0.5)? {
    println!("  [{:.3}] {}", r.score, r.source);
}

// Vector search first, then the cross-encoder rescores the top 20.
// Needs the reranker model; without it this returns an error.
println!("reranked:");
match rag.query_reranked(question, 5, 20) {
    Ok(results) => {
        for r in results {
            println!("  [{:.3}] {}", r.score, r.source);
        }
    }
    Err(e) => eprintln!("  unavailable: {e}"),
}
```

Use hybrid search when queries contain exact names, codes or identifiers that embeddings tend to blur. Use reranking when the top few results matter most and the extra latency is acceptable.

### 7. Fewer near-duplicate results

When many chunks repeat the same text (versioned manuals, copied notes), plain relevance order fills the results with copies. `query_diverse` applies maximal marginal relevance. `relevance_weight` 1.0 is plain relevance order, and 0.0 is the most varied set.

```rust
for r in rag.query_diverse("install on Linux", 5, 0.3)? {
    println!("[{:.2}] {}", r.score, r.source);
}
```

### 8. Tune the index and chunking

Set options on `RagConfig` and pass it to `open_with_config`. Start from `default()` and change only what you need:

```rust
use tpt_rag_core::{ChunkConfig, RAGLite, RagConfig};
use std::path::PathBuf;

let config = RagConfig {
    ef_search: 200,                                // more accurate search, a little slower
    chunk: ChunkConfig { window_size: 256, overlap: 32, ..ChunkConfig::default() },
    model_path: Some(PathBuf::from("/opt/models/all-MiniLM-L6-v2.onnx")),
    ..RagConfig::default()
};

let rag = RAGLite::open_with_config(std::path::Path::new("./kb"), config)?;
```

`validate()` rejects zero values for `m`, `ef_construction` and `ef_search` before the database is opened. The chunker checks its own settings, including that `overlap` is smaller than `window_size`.

### 9. Write, replace and delete

Adding a file again replaces its chunks, so re-ingesting an updated document never leaves stale copies behind. Deleting a document hides its vectors from search at once.

```rust
// Re-adding replaces the earlier chunks for this file.
rag.add_document(Path::new("./docs/manual.pdf"), &["docs"])?;

// Remove every chunk whose source is this label. Returns the number of chunks removed.
let removed = rag.delete_document("./docs/old-manual.pdf")?;
println!("removed {removed} chunks");
```

`add_text` stores a string under a source label. Without a label the text is stored as `inline`:

```rust
use tpt_rag_core::Metadata;

rag.add_text(
    "Rotate API keys on the first Monday of each month.",
    &Metadata { source: Some("policy".into()), tags: vec!["security".into()] },
)?;
```

### 10. Check the database

`check()` runs the same health checks as `raglite doctor` and returns one entry per check. `stats()` reports sizes and the model in use.

```rust
for c in rag.check()? {
    println!("{} {}: {}", if c.ok { "ok  " } else { "FAIL" }, c.name, c.detail);
}

let s = rag.stats()?;
println!("{} chunks, {} vectors, index {} bytes, model {}", s.chunks, s.vectors, s.index_bytes, s.model);
```

### 11. Measure retrieval quality

`eval_cases` builds test queries from the chunks themselves: a few words from one chunk, with that chunk's source as the expected answer. These check that a chunk can be found from its own words, which is easier than real questions, so use them to compare settings rather than as a quality score. Write real questions with the source each one should find, one JSON object per line, and run `raglite eval`.

```rust
use std::io::Write;

let mut out = std::fs::File::create("cases.jsonl")?;
for case in rag.eval_cases(5)? {
    writeln!(out, "{}", serde_json::json!({
        "query": case.query,
        "expected_source": case.expected_source,
    }))?;
}
```

```jsonl
{"query": "how do I rotate the key", "expected_source": "runbooks/keys.md"}
{"query": "nightly backup schedule", "expected_source": "runbooks/backups.md"}
```

### 12. Add a format the library does not read

Implement `Parser` and register it. A registered parser takes precedence over a built-in one for the same extension. See [Custom parsers](#custom-parsers).

## Search methods

| Method | What it does | Score |
|---|---|---|
| `query(text, top_k)` | Nearest chunks by cosine similarity | -1 to 1, higher is better |
| `query_filtered(text, top_k, &filter)` | `query`, keeping only chunks that pass a `QueryFilter` | -1 to 1 |
| `query_hybrid(text, top_k, alpha)` | Vector and BM25 keyword rankings fused with reciprocal rank fusion. `alpha` is the vector weight, 0 to 1 | 0 to 1 |
| `query_reranked(text, top_k, rerank_top_n)` | Takes the best `max(rerank_top_n, top_k)` vector candidates and rescores them with the cross-encoder. Needs the reranker | 0 to 1 |
| `query_diverse(text, top_k, relevance_weight)` | Maximal marginal relevance. Score is the query similarity | -1 to 1 |
| `neighbors(chunk_id, radius)` | Chunks before and after a hit, in reading order, excluding the hit | 0 |

## Results

`SearchResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `score` | `f32` | Relevance under the method's scale (see above) |
| `text` | `String` | The chunk text |
| `source` | `String` | The source path or label the chunk came from |
| `tags` | `Vec<String>` | Tags attached when the chunk was added |
| `chunk_id` | `u64` | Row id in the metadata store. Stable within one database, so it can be cited |
| `created_at` | `u64` | Ingestion time, Unix seconds |
| `start_offset`, `end_offset` | `Option<u64>` | Byte range of the chunk in the text it was cut from. `None` before schema version 2 |
| `section` | `Option<String>` | Heading path (Markdown) or `page N` (PDF). `None` otherwise |

## Configuration

`RagConfig::default()` uses these values:

| Setting | Default | Meaning |
|---|---|---|
| `m` | 16 | HNSW links per node (layer 0 allows `2 * m`). Changing it rebuilds the index on open |
| `ef_construction` | 200 | Index build effort. Higher is slower and more accurate |
| `ef_search` | 100 | Query effort. Raised to `top_k` when smaller |
| `chunk.window_size` | 512 | Words per chunk |
| `chunk.overlap` | 50 | Words shared by neighbouring chunks. Must be below `window_size` |
| `flush_every` | 1000 | Write the index after this many changes. `0` disables it |
| `reembed_on_model_change` | `false` | Re-embed every chunk instead of refusing to open when the model differs |
| `model_path` | `None` | Embedding model file |
| `reranker_path` | `None` | Reranker model file |

### Model lookup order

For the embedding model:

1. `RagConfig::model_path`, if set.
2. The `RAGLITE_MODEL` environment variable.
3. In the database folder: `model.onnx`, or `models/all-MiniLM-L6-v2.onnx`.
4. The model compiled into the binary, if the `embedded-model` feature is on.

For the reranker, the same order applies with `reranker_path`, `RAGLITE_RERANKER`, `reranker.onnx`, and `models/ms-marco-MiniLM-L-6-v2.onnx`. The model's `vocab.txt` must be in the same folder as the model. The current working directory is never searched.

## Persistence and safety

A database is a folder with two files:

- `metadata.sqlite3`: chunk text, source, tags, timestamps, offsets, document hashes, and the identity of the embedding model. Each operation is one SQLite transaction.
- `hnsw.bin`: the vector index, written atomically (temp file, `fsync`, rename) on `flush()`, every `flush_every` changes, and when the value is dropped.

`Drop` cannot report errors. Call `flush()` when you need to know the index is on disk.

- **Crash safety.** A crash can lose changes made since the last flush, but the two files are never left pointing at each other's wrong ids. The index carries an epoch that is also stored in SQLite. If they disagree, or `hnsw.bin` is missing or damaged, the index is rebuilt from the chunk text. That re-embeds everything, so it is slow on a large database.
- **Model identity.** The database records the model file name and vector size. Opening it with a different model fails, so vectors from two models never mix. Set `reembed_on_model_change` to switch deliberately.
- **Schema migrations.** Opening an older database runs the missing steps, each in its own transaction. A database written by a newer library is refused without being changed. The upgrade is one way: an upgraded database cannot be opened by an older release.
- **Ingest atomicity.** Adding a document is one SQLite transaction. If the metadata write fails, its vectors are hidden again.

## Supported formats

| Format | Extensions | Notes |
|---|---|---|
| Plain text | `.txt` | BOM removed; invalid UTF-8 replaced |
| Markdown | `.md`, `.markdown`, `.mdown` | Headings become `section`; markup removed; code blocks kept whole |
| HTML | `.html`, `.htm` | `<script>`, `<style>` and `<noscript>` dropped |
| PDF | `.pdf` | Text layer only, page by page. A PDF with no text is `EmptyDocument`; run OCR first |
| CSV | `.csv` | Each row becomes `header: value; header: value` |
| JSON | `.json` | Each leaf becomes `path: value`, such as `user.tags[0]: math` |
| Word | `.docx` | Body text in order, one line per paragraph. Headers, footers, comments and embedded objects are not read |
| Source and config | `.rs`, `.py`, `.js`, `.ts`, `.go`, `.java`, `.c`, `.cpp`, `.toml`, `.yaml`, and others | Indexed as text |

Files over 64 MB are rejected.

## Custom parsers

A parser turns a file into text. Implement `Parser` and register it. Override `parse_document` only if the format has headings, so chunks can record their section:

```rust
use std::path::Path;
use tpt_rag_core::{Parser, RAGLite};
use tpt_rag_core::error::Result;

/// Reads `.log` files, keeping only lines that mention an error.
struct ErrorLogParser;

impl Parser for ErrorLogParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let raw = std::fs::read_to_string(path)?;
        Ok(raw
            .lines()
            .filter(|line| line.contains("ERROR"))
            .collect::<Vec<_>>()
            .join("\n"))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["log"]
    }
}

let mut rag = RAGLite::open(Path::new("./kb"))?;   // register_parser takes &mut self
rag.register_parser(Box::new(ErrorLogParser));
```

Extensions are lowercase and have no dot.

## Errors

Every fallible method returns `RagError`:

| Variant | When |
|---|---|
| `Io` | A file could not be read or written |
| `Sqlite` | The metadata store returned an error |
| `Onnx(String)` | ONNX Runtime failed to load or run the model |
| `Model(String)` | The model file is missing or invalid |
| `EmptyDocument(String)` | The file has no text, such as a scanned PDF |
| `UnsupportedFormat(String)` | No parser handles the extension |
| `InvalidConfig(String)` | A setting is out of range |
| `TooLarge { .. }` | The file is over 64 MB |
| `Serialization` | JSON parsing or writing failed |
| `Hnsw(String)` | The vector index is corrupt or inconsistent with the metadata |

Match on the variant when the response depends on the cause:

```rust
use tpt_rag_core::RagError;

match rag.add_document(path, &[]) {
    Ok(()) => {}
    Err(RagError::EmptyDocument(_)) => eprintln!("{} has no text; run OCR first", path.display()),
    Err(RagError::UnsupportedFormat(ext)) => eprintln!("cannot read {ext}"),
    Err(e) => return Err(e.into()),
}
```

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `reranking` | on | Compiles the cross-encoder code. `query_reranked` still needs the reranker model at run time |
| `embedded-model` | off | Compiles `models/all-MiniLM-L6-v2.onnx` and `models/vocab.txt` into the binary, so a database opens without model files on disk. The files must exist at build time |

## The `raglite` command

The crate builds a command-line tool:

```bash
cargo run -p tpt-rag-core --bin raglite -- <command>
```

| Command | What it does |
|---|---|
| `ingest <db> <paths...> [--tag T]` | Add folders (recursive), files or glob patterns |
| `query <db> "<text>" [-k N] [--tag T] [--source-prefix P] [--since T] [--hybrid A] [--rerank N] [--diverse W]` | Search. Prints the source, section and byte range of each result. `--diverse` cannot be combined with the other modes or the filters, and `--rerank` cannot be combined with the filters |
| `sync <db> <folder>` | Index new and changed files, drop deleted ones |
| `watch <db> <folder> [--interval S]` | `sync` every `S` seconds (default 30) until stopped |
| `serve <db> [--mcp] [--addr A] [--allow-remote]` | JSON API over HTTP, or an MCP server over stdio with `--mcp` |
| `stats <db>` | Chunk, vector and index counts |
| `doctor <db>` | Health checks. Exits 1 if any fails |
| `eval <db> <cases.jsonl> [-k N]` | Recall@k, MRR, and nDCG@k over a set of labelled queries |
| `synth <db> [--per-source N]` | Print eval cases generated from the chunks, as JSONL |
| `delete <db> <source>` | Remove a document |
| `rebuild <db>` | Re-embed every chunk with the current model |

A typical session:

```bash
raglite ingest ./kb ./notes/*.md --tag kb
raglite query ./kb "how do I rotate the key" -k 3 --tag kb
raglite synth ./kb --per-source 5 > cases.jsonl
raglite eval ./kb cases.jsonl -k 5
```

`raglite serve` binds to `127.0.0.1:8080` and refuses other addresses unless `--allow-remote` is given. The HTTP API has **no authentication**, so keep it on loopback or put it behind an authenticating proxy. The MCP mode opens no network port.

## Testing

```bash
cargo test -p tpt-rag-core                  # unit tests and fixtures
cargo test -p tpt-rag-core --test integration   # needs the model files
```

Integration tests that need the model print a notice and pass when it is missing. Run `scripts/fetch_model.sh` first to run them for real.

## Limitations

- **Pre-release.** The API may change before 1.0.
- **Text layer only.** Scanned PDFs and images need OCR before ingestion.
- **Word files** lose headers, footers, comments and embedded objects.
- **Offsets are in the extracted text**, not the original file, for PDF, Markdown and HTML.
- **No authentication** in the HTTP API. Bind it to loopback.

## License

MIT OR Apache-2.0. © TPT Solutions. See the repository's `LICENSE-MIT` and `LICENSE-APACHE`.
