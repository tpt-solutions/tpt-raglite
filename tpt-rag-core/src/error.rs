#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("ONNX Runtime error: {0}")]
    Onnx(String),

    #[error("No text could be extracted from file: {0}")]
    EmptyDocument(String),

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HNSW index error: {0}")]
    Hnsw(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("File {path} is {size} bytes, over the {limit} byte limit")]
    TooLarge { path: String, size: u64, limit: u64 },

    #[error("database schema version {stored} is newer than this library supports ({supported}); upgrade tpt-raglite")]
    SchemaTooNew { stored: u32, supported: u32 },
}

pub type Result<T> = std::result::Result<T, RagError>;
