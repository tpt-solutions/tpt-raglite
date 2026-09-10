use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("ONNX Runtime error: {0}")]
    Onnx(String),

    #[error("No text could be extracted from file: {path}")]
    EmptyDocument { path: PathBuf },

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("Chunk index out of bounds: {index} >= {len}")]
    ChunkIndexOutOfBounds { index: usize, len: usize },

    #[error("Model error: {0}")]
    Model(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HNSW index error: {0}")]
    Hnsw(String),
}

pub type Result<T> = std::result::Result<T, RagError>;
