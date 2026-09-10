use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyList;

use tpt_rag_core::{Metadata, RAGLite, SearchResult};

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
}

impl From<SearchResult> for PySearchResult {
    fn from(r: SearchResult) -> Self {
        Self {
            score: r.score,
            text: r.text,
            source: r.source,
            tags: r.tags,
        }
    }
}

#[pyclass(name = "RAGLite")]
struct PyRAGLite {
    inner: std::sync::Mutex<RAGLite>,
}

#[pymethods]
impl PyRAGLite {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let rag = RAGLite::open(&PathBuf::from(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            inner: std::sync::Mutex::new(rag),
        })
    }

    fn add_files(&self, paths: Vec<String>, tags: Option<Vec<String>>) -> PyResult<()> {
        let rag = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let tag_strs: Vec<&str> = tags
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|s| s.as_str())
            .collect();
        for p in &paths {
            rag.add_document(PathBuf::from(p).as_path(), &tag_strs)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    fn add_text(&self, text: &str) -> PyResult<()> {
        let rag = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        rag.add_text(text, &Metadata::default())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn query<'py>(
        &self,
        query_text: &str,
        top_k: Option<usize>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyList>> {
        let k = top_k.unwrap_or(5);
        let rag = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let results = rag
            .query(query_text, k)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let py_results: Vec<PySearchResult> =
            results.into_iter().map(PySearchResult::from).collect();
        PyList::new(py, py_results)
    }
}

#[pymodule]
fn tpt_raglite(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRAGLite>()?;
    m.add_class::<PySearchResult>()?;
    Ok(())
}
