# tpt-raglite (Python)

Python bindings for tpt-raglite, the embeddable RAG engine. A database is a folder, and everything runs in your process: parsing, chunking, local embeddings, the vector index, and storage. There is no server, no API key, and no network call once the model files are on disk.

The bindings are built with [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs). The engine is [`tpt-rag-core`](../tpt-rag-core/).

> **Status: pre-release (0.1.0).** Not yet on PyPI. Build from source with maturin, as described below.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Reference](#reference)
- [Errors](#errors)
- [Threads](#threads)
- [Testing](#testing)

## Install

Requirements: Python 3.9 or newer, a Rust toolchain (stable), and a C compiler. On Windows, use a current Visual Studio 2022 install. The prebuilt ONNX Runtime binaries need a newer MSVC standard library than toolset 14.38 provides.

From the repository root:

```bash
bash scripts/fetch_model.sh                  # Windows: powershell -File scripts/fetch_model.ps1

pip install maturin
cd tpt-rag-py
maturin develop --release                    # builds and installs into the active environment
```

`maturin develop` installs the package `tpt_raglite` into your current virtual environment. To build a wheel instead, run `maturin build --release`.

### Model files

The engine needs an embedding model and its `vocab.txt`. The download script places them in `models/`. Point the library at them with an environment variable, or pass the path to the constructor:

```bash
export RAGLITE_MODEL=/path/to/models/all-MiniLM-L6-v2.onnx
export RAGLITE_RERANKER=/path/to/models/ms-marco-MiniLM-L-6-v2.onnx   # optional, for query_reranked
```

If neither is set, the library looks in the database folder (`model.onnx` or `models/all-MiniLM-L6-v2.onnx`). The current working directory is never searched.

## Quick start

```python
from tpt_raglite import RAGLite

with RAGLite("./my_db") as db:                 # flushes the index on exit
    db.add_files(["manual.pdf", "notes.md"], tags=["docs"])
    db.add_directory("./more_docs")             # recursive
    for r in db.query("How do I configure the engine?", top_k=5):
        print(f"[{r.score:.2f}] {r.source}: {r.text}")
```

## Examples

### 1. Ingest a folder and read the report

`add_directory` and `add_glob` do not stop at the first bad file. They return what was added and what failed, so you can log the failures and keep going:

```python
from tpt_raglite import RAGLite

db = RAGLite("./kb")
report = db.add_directory("./docs", tags=["docs"])

print(f"added {len(report['added'])} file(s)")
for path, reason in report["failed"]:
    print(f"skipped {path}: {reason}")

db.add_glob("notes/**/*.md", tags=["notes"])   # the same report shape
db.flush()
```

### 2. Keep a folder in sync

Call `sync_directory` on a schedule. It hashes each file, ingests the new and changed ones, skips the rest, and removes documents whose files were deleted. Only files that a sync ingested are removed by it, so documents you added one by one are safe.

```python
import time
from tpt_raglite import RAGLite

db = RAGLite("./kb")
while True:
    report = db.sync_directory("./notes", tags=["notes"])
    print(
        f"indexed {len(report['indexed'])}, unchanged {report['unchanged']}, "
        f"removed {len(report['removed'])}, failed {len(report['failed'])}"
    )
    db.flush()
    time.sleep(30)
```

### 3. Filter by tag, source, or date

Filters combine with AND. The filter is applied to the best `10 × top_k` vector matches, so a very selective filter can return fewer than `top_k` results. Raise `top_k` if you see that.

```python
import time
from tpt_raglite import RAGLite

db = RAGLite("./kb")

one_week_ago = int(time.time()) - 7 * 24 * 3600
results = db.query(
    "how do I rotate the key",
    top_k=5,
    tags=["ops", "prod"],          # the chunk must carry both tags
    source_prefix="runbooks/",      # and its source must start with this
    since=one_week_ago,             # and it must have been ingested after this Unix time
)
for r in results:
    print(f"[{r.score:.2f}] {r.source}: {r.text}")
```

### 4. Cite passages by location

Each result says where its text came from: the source, the section (for Markdown, the heading path such as `Setup > Backups`; for PDFs, `page 3`), and the byte range in the extracted text.

```python
from tpt_raglite import RAGLite

db = RAGLite("./kb")
for r in db.query("when do backups run", top_k=3):
    where = r.source
    if r.section:
        where += f" › {r.section}"
    print(f"[{r.score:.2f}] {where}")
    print(f"    {r.text}")
```

Offsets are byte positions, but Python strings index by character. For text with non-ASCII characters, slice the UTF-8 bytes:

```python
def original_span(result):
    """Returns the chunk's text from its byte offsets. For plain text and add_text, this equals result.text."""
    if result.start_offset is None or result.end_offset is None:
        return None  # chunk stored before schema version 2
    raw = result.text.encode("utf-8")
    return raw[result.start_offset:result.end_offset].decode("utf-8", errors="replace")
```

Offsets refer to the text the parser extracted, not the original file. For a PDF, that is the PDF's text layer, page by page.

### 5. Build a prompt for a language model

The library does not call a model. It gives you the context, and you send it. This shows a prompt that keeps each passage's source, so the model can cite it:

```python
from tpt_raglite import RAGLite

def build_prompt(db: RAGLite, question: str, top_k: int = 4) -> str:
    hits = db.query(question, top_k=top_k)
    passages = []
    for i, r in enumerate(hits, start=1):
        label = f"{r.source} › {r.section}" if r.section else r.source
        passages.append(f"[{i}] ({label})\n{r.text}")

    context = "\n\n".join(passages)
    return (
        "Answer the question using only the passages below. "
        "Cite passages by their number, such as [2]. "
        "If the passages do not answer it, say so.\n\n"
        f"{context}\n\nQuestion: {question}"
    )

db = RAGLite("./kb")
prompt = build_prompt(db, "How do I rotate the API key?")
# send `prompt` to the language model of your choice
```

### 6. Expand a hit into its neighbours

A chunk can start or end mid-thought. `neighbors` returns the chunks just before and after a hit in the same document, in reading order. The hit itself is left out, so keep its score:

```python
from tpt_raglite import RAGLite

db = RAGLite("./kb")
hits = db.query("retention period for logs", top_k=1)

if hits:
    hit = hits[0]
    before_and_after = db.neighbors(hit.chunk_id, radius=1)
    print(f"[{hit.score:.2f}] {hit.text}")
    for n in before_and_after:
        print("  context:", n.text)
```

Neighbours have a score of `0.0`. Chunks stored before schema version 2 have no offsets and so no neighbours.

### 7. Compare ranking methods

Scores are not comparable across methods, so compare results within one method. Here is one question through four:

```python
from tpt_raglite import RAGLite

db = RAGLite("./kb")
question = "API key rotation"

def show(label, results):
    print(label)
    for r in results:
        print(f"  [{r.score:.3f}] {r.source}")

show("vector", db.query(question, top_k=3))

# alpha is the vector weight: 1.0 is vectors only, 0.0 is keywords only.
show("hybrid, alpha 0.3 (favours keywords)", db.query_hybrid(question, top_k=3, alpha=0.3))

# relevance_weight 1.0 is plain relevance order; 0.0 is the most varied set.
show("diverse, weight 0.3", db.query_diverse(question, top_k=3, relevance_weight=0.3))

# Needs the reranker model. Without it this raises ModelError.
try:
    show("reranked", db.query_reranked(question, top_k=3, rerank_top_n=20))
except Exception as e:
    print("reranked unavailable:", e)
```

Use hybrid search for queries that contain exact names, codes or identifiers, because embeddings tend to blur them. Use diverse search when many chunks repeat the same text.

### 8. Tune the index and chunking

Keyword arguments to `RAGLite` change how the database is built and searched. The defaults suit most uses:

```python
from tpt_raglite import RAGLite

db = RAGLite(
    "./kb",
    model_path="/opt/models/all-MiniLM-L6-v2.onnx",   # overrides RAGLITE_MODEL
    ef_search=200,          # more accurate search, a little slower
    window_size=256,        # words per chunk
    overlap=32,             # words shared by neighbouring chunks; must be below window_size
    flush_every=500,        # write the index every 500 changes; 0 turns this off
)
```

Changing `m` rebuilds the index when the database opens, so set it once when you create the database. If you open a database with a different embedding model, the constructor raises `ModelError`. Pass `reembed_on_model_change=True` to re-embed every chunk with the new model instead.

### 9. Replace, delete, and check

Adding a file again replaces its chunks. Deleting by source removes them from search at once:

```python
from tpt_raglite import RAGLite

db = RAGLite("./kb")

db.add_files(["docs/manual.pdf"], tags=["docs"])   # replaces the earlier version, if any
removed = db.delete_document("docs/old-manual.pdf")
print(f"removed {removed} chunk(s)")

stats = db.stats()
print(f"{stats['chunks']} chunks, {stats['vectors']} vectors, index {stats['index_bytes']} bytes")
print("model:", stats["model"])

db.flush()   # writes the index now; a failure raises here, where your code can handle it
```

`add_text` stores a string, with an optional source label and tags. Without a label the source is `inline`:

```python
db.add_text(
    "Rotate API keys on the first Monday of each month.",
    source="policy",
    tags=["security"],
)
```

### 10. Handle errors by type

Most library errors are subclasses of `RAGLiteError`, so one `except` can catch them all. The subclasses let you respond to specific causes, and the standard `OSError` and `ValueError` cover file and settings problems:

```python
from tpt_raglite import (
    EmptyDocumentError,
    ModelError,
    RAGLite,
    RAGLiteError,
    UnsupportedFormatError,
)

db = RAGLite("./kb")

for path in ["scan.pdf", "data.xyz", "manual.pdf", "missing.md"]:
    try:
        db.add_files([path], tags=["docs"])
    except EmptyDocumentError:
        print(f"{path} has no text; run OCR first")
    except UnsupportedFormatError:
        print(f"{path}: no parser for this extension")
    except OSError as e:
        print(f"{path} could not be read: {e}")
    except RAGLiteError as e:
        print(f"{path}: {e}")

try:
    db.query_reranked("question", top_k=3)
except ModelError as e:
    print("reranker not installed:", e)
```

| Exception | When |
|---|---|
| `RAGLiteError` | Base class for library errors, including anything not listed below |
| `ModelError` | The embedding or reranker model is missing or cannot run, or the database was built with a different model |
| `UnsupportedFormatError` | No parser handles the file's extension |
| `EmptyDocumentError` | The file has no text, such as a scanned PDF |
| `OSError` | A file could not be read or written. This is Python's built-in exception |
| `ValueError` | A setting is out of range or contradicts another. This is Python's built-in exception |

### 11. Use it as a context manager

`with` flushes the index when the block exits, including when an exception is raised inside it. Use it for scripts and jobs:

```python
from tpt_raglite import RAGLite

with RAGLite("./kb") as db:
    db.add_text("Jupiter is the largest planet in the solar system.", source="space")
    print(db.query("How big is Jupiter?", top_k=1)[0].text)
# the index is written here
```

For a long-running service, keep one `RAGLite` object for the process lifetime and call `flush()` at points where you need the data on disk.

### 12. Share one database across threads

One `RAGLite` object can be used from several threads. Each method takes an internal lock, so calls run one at a time. The lock is released while the engine works, so the Python interpreter stays responsive during long ingestion. Searches from several threads do not run in parallel.

```python
from concurrent.futures import ThreadPoolExecutor
from tpt_raglite import RAGLite

db = RAGLite("./kb")
questions = ["how do I rotate the key", "when do backups run", "what is the retention period"]

with ThreadPoolExecutor(max_workers=3) as pool:
    for q, results in zip(questions, pool.map(lambda q: db.query(q, top_k=2), questions)):
        print(q, "->", [r.source for r in results])
```

## Reference

### `RAGLite(path, *, ...)`

Opens or creates the database folder at `path`.

| Keyword | Default | Meaning |
|---|---|---|
| `model_path` | `None` | Embedding model file. `None` uses the lookup order above |
| `reranker_path` | `None` | Reranker model file |
| `reembed_on_model_change` | `False` | Re-embed when the database was built with a different model |
| `m` | `16` | HNSW links per node. Changing it rebuilds the index on open |
| `ef_construction` | `200` | Index build effort |
| `ef_search` | `100` | Query effort |
| `window_size` | `512` | Words per chunk |
| `overlap` | `50` | Words shared by neighbouring chunks |
| `flush_every` | `1000` | Write the index after this many changes. `0` turns it off |

### Ingestion

| Method | Returns | Notes |
|---|---|---|
| `add_files(paths, tags=None)` | `None` | Ingests files. Re-adding a file replaces its chunks |
| `add_directory(path, tags=None)` | `{"added": [...], "failed": [(path, reason), ...]}` | Recursive |
| `add_glob(pattern, tags=None)` | Same as `add_directory` | Patterns such as `notes/**/*.md` |
| `sync_directory(path, tags=None)` | `{"indexed", "unchanged", "removed", "failed"}` | Keeps a folder in step with the database |
| `add_text(text, source=None, tags=None)` | `None` | Without `source`, stored as `inline` |

### Search

Every search method returns a list of `SearchResult`.

| Method | Notes |
|---|---|
| `query(text, top_k=5, tags=None, source_prefix=None, since=None)` | Vector search with optional filters. `since` is Unix seconds |
| `query_hybrid(text, top_k=5, alpha=0.5)` | Vector and keyword results fused. `alpha` is the vector weight |
| `query_diverse(text, top_k=5, relevance_weight=0.5)` | Relevant results that are not near-duplicates |
| `query_reranked(text, top_k=5, rerank_top_n=20)` | Cross-encoder rescoring. Needs the reranker |
| `neighbors(chunk_id, radius=1)` | Chunks around a hit, not including the hit |

Scores: `query` returns cosine similarity (-1 to 1). `query_hybrid` and `query_reranked` return 0 to 1. `query_diverse` returns the query similarity. Neighbours are `0.0`.

### `SearchResult`

| Attribute | Type | Meaning |
|---|---|---|
| `score` | `float` | Relevance. See the scales above |
| `text` | `str` | The chunk text |
| `source` | `str` | The source path or label |
| `tags` | `list[str]` | Tags attached when the chunk was added |
| `chunk_id` | `int` | Stable row id within one database. Use it to cite or to call `neighbors` |
| `created_at` | `int` | Ingestion time, Unix seconds |
| `start_offset`, `end_offset` | `int \| None` | Byte range in the extracted text. `None` for chunks stored before schema version 2 |
| `section` | `str \| None` | Heading path (Markdown) or `page N` (PDF) |

### Maintenance

| Method | Returns | Notes |
|---|---|---|
| `delete_document(source)` | `int` | Chunks removed. Their vectors leave search immediately |
| `flush()` | `None` | Writes the index to disk and reports any error |
| `rebuild_index()` | `int` | Re-embeds every chunk with the current model |
| `stats()` | `dict` | `chunks`, `vectors`, `index_bytes`, `model` |

### Formats

Plain text, Markdown, HTML, PDF (text layer only), Word (`.docx`, body text only), CSV, JSON, and source or configuration files. Files over 64 MB are rejected. For the full table, see the [core README](../tpt-rag-core/README.md#supported-formats).

## Errors

See [Examples, section 10](#10-handle-errors-by-type) for handling. `RAGLiteError` is the base class for library errors. `ModelError`, `UnsupportedFormatError`, and `EmptyDocumentError` are subclasses for specific causes. File errors raise `OSError`, and invalid settings raise `ValueError`.

## Threads

A `RAGLite` object is safe to share between threads. Calls are serialized by an internal lock, and the lock is released during the engine's work, so other Python threads keep running. Two threads cannot run engine work at once on the same object.

## Testing

```bash
cd tpt-rag-py && maturin develop --release && cd ..
pytest tpt-rag-py/tests
```

Tests that need the model are skipped when it is missing. Run `scripts/fetch_model.sh` first to run them all.

## License

MIT OR Apache-2.0. © TPT Solutions.
