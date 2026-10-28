use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::chunker::ChunkConfig;
use crate::error::{RagError, Result};

/// Settings for opening a database. `RagConfig::default()` suits most uses:
///
/// ```no_run
/// # use tpt_rag_core::{RAGLite, RagConfig};
/// let config = RagConfig { ef_search: 200, ..RagConfig::default() };
/// let rag = RAGLite::open_with_config("./kb".as_ref(), config)?;
/// # Ok::<(), tpt_rag_core::RagError>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// HNSW links per node. Layer 0 allows `2 * m`. Changing it rebuilds the index on open.
    pub m: usize,
    /// Candidate list size while the index is built. Higher is slower and more accurate.
    pub ef_construction: usize,
    /// Candidate list size per query. Values below `top_k` are raised to `top_k`.
    pub ef_search: usize,
    /// How documents are split into chunks.
    pub chunk: ChunkConfig,
    /// Embedding model file. When `None`, the model is looked up in this order:
    /// `RAGLITE_MODEL`, then `<db>/model.onnx`, then `<db>/models/all-MiniLM-L6-v2.onnx`.
    /// The model's `vocab.txt` must sit in the same folder.
    pub model_path: Option<PathBuf>,
    /// Cross-encoder reranker file. When `None`, `RAGLITE_RERANKER` is used, then
    /// `<db>/reranker.onnx`, then `<db>/models/ms-marco-MiniLM-L-6-v2.onnx`. If none
    /// exists, reranking is unavailable and `query_reranked` returns an error.
    pub reranker_path: Option<PathBuf>,
    /// Write the index to disk after this many vectors are added or removed. `0` turns this
    /// off, and the index is then written only by `flush()` and on drop.
    pub flush_every: usize,
    /// When the database was built with a different embedding model than the one given,
    /// re-embed every chunk with the new model instead of failing to open.
    pub reembed_on_model_change: bool,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            chunk: ChunkConfig::default(),
            model_path: None,
            reranker_path: None,
            flush_every: 1000,
            reembed_on_model_change: false,
        }
    }
}

impl RagConfig {
    /// Rejects values that cannot work. The chunk settings are checked by the chunker.
    pub fn validate(&self) -> Result<()> {
        if self.m == 0 {
            return Err(RagError::InvalidConfig("m must be greater than 0".into()));
        }
        if self.ef_construction == 0 {
            return Err(RagError::InvalidConfig(
                "ef_construction must be greater than 0".into(),
            ));
        }
        if self.ef_search == 0 {
            return Err(RagError::InvalidConfig(
                "ef_search must be greater than 0".into(),
            ));
        }
        Ok(())
    }
}
