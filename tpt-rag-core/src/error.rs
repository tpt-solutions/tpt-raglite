/// Errors returned by tpt-raglite.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    /// Reading or writing a file failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The metadata database returned an error.
    #[cfg(feature = "native")]
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// ONNX Runtime failed to load or run the model.
    #[error("ONNX Runtime error: {0}")]
    Onnx(String),

    /// The file parsed but contained no text, for example a scanned PDF.
    #[error("No text could be extracted from file: {0}")]
    EmptyDocument(String),

    /// No parser handles this file extension.
    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    /// The model file is missing or invalid.
    #[error("Model error: {0}")]
    Model(String),

    /// JSON serialization or parsing failed.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// The vector index is corrupt or inconsistent with the metadata store.
    #[error("HNSW index error: {0}")]
    Hnsw(String),

    /// A setting is out of range or contradicts another setting.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// The file is larger than the size limit.
    #[error("File {path} is {size} bytes, over the {limit} byte limit")]
    TooLarge {
        /// The file that was too large.
        path: String,
        /// Its size in bytes.
        size: u64,
        /// The configured limit in bytes.
        limit: u64,
    },

    /// The database was written by a newer version of tpt-raglite.
    #[error("database schema version {stored} is newer than this library supports ({supported}); upgrade tpt-raglite")]
    SchemaTooNew {
        /// The schema version stored in the database.
        stored: u32,
        /// The newest schema version this library reads.
        supported: u32,
    },
}

/// Shorthand for a `Result` whose error is [`RagError`].
pub type Result<T> = std::result::Result<T, RagError>;
