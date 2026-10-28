use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::exceptions::{PyException, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{create_exception, PyErr};

use tpt_rag_core::{
    ChunkConfig, IngestReport, Metadata, QueryFilter, RAGLite, RagConfig, RagError, SearchResult,
};

create_exception!(
    tpt_raglite,
    RAGLiteError,
    PyException,
    "Base class for tpt-raglite errors."
);
create_exception!(
    tpt_raglite,
    ModelError,
    RAGLiteError,
    "The embedding or reranking model could not be loaded or run."
);
create_exception!(
    tpt_raglite,
    UnsupportedFormatError,
    RAGLiteError,
    "No parser handles the file's extension."
);
create_exception!(
    tpt_raglite,
    EmptyDocumentError,
    RAGLiteError,
    "The file contains no text, such as a scanned PDF."
);

fn to_py_err(e: RagError) -> PyErr {
    let message = e.to_string();
    match e {
        RagError::Io(_) => PyOSError::new_err(message),
        RagError::Model(_) | RagError::Onnx(_) => ModelError::new_err(message),
        RagError::UnsupportedFormat(_) => UnsupportedFormatError::new_err(message),
        RagError::EmptyDocument(_) => EmptyDocumentError::new_err(message),
        RagError::InvalidConfig(_) => PyValueError::new_err(message),
        _ => RAGLiteError::new_err(message),
    }
}

#[pyclass(name = "SearchResult", skip_from_py_object)]
#[derive(Clone)]
struct PySearchResult {
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    tags: Vec<String>,
    /// Row id of the chunk. Stable within one database.
    #[pyo3(get)]
    chunk_id: u64,
    /// Ingestion time, in seconds since the Unix epoch.
    #[pyo3(get)]
    created_at: u64,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        let preview: String = self.text.chars().take(60).collect();
        format!(
            "SearchResult(score={:.4}, source={:?}, text={:?})",
            self.score, self.source, preview
        )
    }
}

impl From<SearchResult> for PySearchResult {
    fn from(r: SearchResult) -> Self {
        Self {
            score: r.score,
            text: r.text,
            source: r.source,
            tags: r.tags,
            chunk_id: r.chunk_id,
            created_at: r.created_at,
        }
    }
}

/// `{"added": [paths], "failed": [(path, reason)]}`, the shape returned by ingestion calls.
fn report_to_dict<'py>(py: Python<'py>, report: &IngestReport) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "added",
        report
            .added
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "failed",
        report
            .failed
            .iter()
            .map(|(p, reason)| (p.to_string_lossy().into_owned(), reason.clone()))
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

fn to_py_results(results: Vec<SearchResult>) -> Vec<PySearchResult> {
    results.into_iter().map(PySearchResult::from).collect()
}

#[pyclass(name = "RAGLite")]
struct PyRAGLite {
    // The mutex serializes Python threads. Heavy calls release the GIL while they hold it.
    inner: Mutex<RAGLite>,
}

#[pymethods]
impl PyRAGLite {
    /// Opens or creates the database in `path`.
    ///
    /// `model_path` and `reranker_path` default to the search order described in the docs,
    /// which includes the `RAGLITE_MODEL` and `RAGLITE_RERANKER` environment variables.
    #[new]
    #[pyo3(signature = (
        path,
        *,
        model_path = None,
        reranker_path = None,
        reembed_on_model_change = false,
        m = 16,
        ef_construction = 200,
        ef_search = 100,
        window_size = 512,
        overlap = 50,
        flush_every = 1000,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: &str,
        model_path: Option<PathBuf>,
        reranker_path: Option<PathBuf>,
        reembed_on_model_change: bool,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        window_size: usize,
        overlap: usize,
        flush_every: usize,
    ) -> PyResult<Self> {
        let config = RagConfig {
            m,
            ef_construction,
            ef_search,
            chunk: ChunkConfig {
                window_size,
                overlap,
                ..ChunkConfig::default()
            },
            model_path,
            reranker_path,
            flush_every,
            reembed_on_model_change,
        };
        let path = PathBuf::from(path);
        let rag = py
            .detach(|| RAGLite::open_with_config(&path, config))
            .map_err(to_py_err)?;
        Ok(Self {
            inner: Mutex::new(rag),
        })
    }

    /// Ingests files. Re-adding a file replaces its earlier chunks.
    #[pyo3(signature = (paths, tags = None))]
    fn add_files(
        &self,
        py: Python<'_>,
        paths: Vec<String>,
        tags: Option<Vec<String>>,
    ) -> PyResult<()> {
        let tags = tags.unwrap_or_default();
        py.detach(|| -> Result<(), RagError> {
            let rag = self.lock()?;
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            for p in &paths {
                rag.add_document(PathBuf::from(p).as_path(), &tag_refs)?;
            }
            Ok(())
        })
        .map_err(to_py_err)
    }

    /// Ingests every supported file under `path`. Returns a dict with `added` (paths) and
    /// `failed` (list of `(path, reason)` pairs). A failed file does not stop the others.
    #[pyo3(signature = (path, tags = None))]
    fn add_directory<'py>(
        &self,
        py: Python<'py>,
        path: &str,
        tags: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let tags = tags.unwrap_or_default();
        let dir = PathBuf::from(path);
        let report = py
            .detach(|| -> Result<_, RagError> {
                let rag = self.lock()?;
                let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
                rag.add_directory(&dir, &tag_refs, &mut |_, _, _| {})
            })
            .map_err(to_py_err)?;
        report_to_dict(py, &report)
    }

    #[pyo3(signature = (text, source = None, tags = None))]
    fn add_text(
        &self,
        py: Python<'_>,
        text: &str,
        source: Option<String>,
        tags: Option<Vec<String>>,
    ) -> PyResult<()> {
        let metadata = Metadata {
            source,
            tags: tags.unwrap_or_default(),
        };
        py.detach(|| -> Result<(), RagError> { self.lock()?.add_text(text, &metadata) })
            .map_err(to_py_err)
    }

    /// Returns the `top_k` most similar chunks. Scores are cosine similarities. `tags` keeps
    /// only chunks that carry every given tag, and `source_prefix` only chunks whose source
    /// starts with the text. A selective filter can return fewer than `top_k` results.
    #[pyo3(signature = (text, top_k = 5, tags = None, source_prefix = None))]
    fn query(
        &self,
        py: Python<'_>,
        text: &str,
        top_k: usize,
        tags: Option<Vec<String>>,
        source_prefix: Option<String>,
    ) -> PyResult<Vec<PySearchResult>> {
        let filter = QueryFilter {
            tags: tags.unwrap_or_default(),
            source_prefix,
        };
        let filtered = !filter.tags.is_empty() || filter.source_prefix.is_some();
        py.detach(|| -> Result<_, RagError> {
            let rag = self.lock()?;
            if filtered {
                rag.query_filtered(text, top_k, &filter)
            } else {
                rag.query(text, top_k)
            }
        })
        .map(to_py_results)
        .map_err(to_py_err)
    }

    /// Ingests the supported files matching a glob pattern such as `notes/**/*.md`. Returns
    /// the same dict as `add_directory`.
    #[pyo3(signature = (pattern, tags = None))]
    fn add_glob<'py>(
        &self,
        py: Python<'py>,
        pattern: &str,
        tags: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let tags = tags.unwrap_or_default();
        let report = py
            .detach(|| -> Result<_, RagError> {
                let rag = self.lock()?;
                let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
                rag.add_glob(pattern, &tag_refs, &mut |_, _, _| {})
            })
            .map_err(to_py_err)?;
        report_to_dict(py, &report)
    }

    /// Health checks as a list of dicts with `name`, `ok` and `detail`. Changes nothing.
    fn check<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let checks = self.lock_py()?.check().map_err(to_py_err)?;
        let items = PyList::empty(py);
        for c in checks {
            let item = PyDict::new(py);
            item.set_item("name", c.name)?;
            item.set_item("ok", c.ok)?;
            item.set_item("detail", c.detail)?;
            items.append(item)?;
        }
        Ok(items)
    }

    /// Vector and keyword search fused with reciprocal rank fusion. `alpha` is the vector
    /// weight, from 0 (keywords only) to 1 (vectors only).
    #[pyo3(signature = (text, top_k = 5, alpha = 0.5))]
    fn query_hybrid(
        &self,
        py: Python<'_>,
        text: &str,
        top_k: usize,
        alpha: f32,
    ) -> PyResult<Vec<PySearchResult>> {
        py.detach(|| -> Result<_, RagError> { self.lock()?.query_hybrid(text, top_k, alpha) })
            .map(to_py_results)
            .map_err(to_py_err)
    }

    /// Vector search followed by cross-encoder reranking. Needs a reranker model.
    #[pyo3(signature = (text, top_k = 5, rerank_top_n = 20))]
    fn query_reranked(
        &self,
        py: Python<'_>,
        text: &str,
        top_k: usize,
        rerank_top_n: usize,
    ) -> PyResult<Vec<PySearchResult>> {
        py.detach(|| -> Result<_, RagError> {
            self.lock()?.query_reranked(text, top_k, rerank_top_n)
        })
        .map(to_py_results)
        .map_err(to_py_err)
    }

    /// Removes every chunk ingested from `source`. Returns the number of chunks removed.
    fn delete_document(&self, py: Python<'_>, source: &str) -> PyResult<usize> {
        py.detach(|| -> Result<usize, RagError> { self.lock()?.delete_document(source) })
            .map_err(to_py_err)
    }

    /// Writes the index to disk now, so that any write error is raised here.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| -> Result<(), RagError> { self.lock()?.flush() })
            .map_err(to_py_err)
    }

    /// Re-embeds every chunk with the current model. Returns the number of chunks embedded.
    fn rebuild_index(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| -> Result<usize, RagError> { self.lock()?.rebuild_index() })
            .map_err(to_py_err)
    }

    /// Chunk count, vector count, index size in bytes, and model file name.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.lock_py()?.stats().map_err(to_py_err)?;
        let out = PyDict::new(py);
        out.set_item("chunks", stats.chunks)?;
        out.set_item("vectors", stats.vectors)?;
        out.set_item("index_bytes", stats.index_bytes)?;
        out.set_item("model", stats.model)?;
        Ok(out)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<Bound<'_, PyAny>>,
        _exc_value: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.flush(py)?;
        Ok(false)
    }
}

impl PyRAGLite {
    /// Locks the database. Used inside `detach`, where no Python objects may be touched.
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, RAGLite>, RagError> {
        self.inner
            .lock()
            .map_err(|e| RagError::Hnsw(format!("database lock poisoned: {e}")))
    }

    fn lock_py(&self) -> PyResult<std::sync::MutexGuard<'_, RAGLite>> {
        self.inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("database lock poisoned: {e}")))
    }
}

#[pymodule]
fn tpt_raglite(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyRAGLite>()?;
    m.add_class::<PySearchResult>()?;
    m.add("RAGLiteError", py.get_type::<RAGLiteError>())?;
    m.add("ModelError", py.get_type::<ModelError>())?;
    m.add(
        "UnsupportedFormatError",
        py.get_type::<UnsupportedFormatError>(),
    )?;
    m.add("EmptyDocumentError", py.get_type::<EmptyDocumentError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
