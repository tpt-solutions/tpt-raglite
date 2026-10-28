//! tpt-raglite: an embeddable, offline RAG engine. Ingest documents with [RAGLite::add_document] or [RAGLite::add_text], then search them with [RAGLite::query].

/// Splitting documents into overlapping, sentence-aware chunks.
pub mod chunker;
/// Settings for opening a database.
pub mod config;
/// Local ONNX sentence embeddings.
#[cfg(feature = "native")]
pub mod embedding;
/// Error type for every fallible operation.
pub mod error;
/// Labelled queries made from the database itself, for `raglite eval`.
pub mod eval;
/// In-memory HNSW vector index with disk persistence.
pub mod hnsw;
/// SQLite storage for chunk text, sources, tags and the full-text index.
#[cfg(feature = "native")]
pub mod metadata;
/// Turning files into plain text.
pub mod parser;
/// The RAGLite database: ingest documents, then search them.
#[cfg(feature = "native")]
pub mod raglite;
/// Cross-encoder reranking of search candidates.
#[cfg(feature = "reranking")]
pub mod reranker;
/// WordPiece tokenizer for the embedding model.
pub mod tokenizer;
/// Vector arithmetic: dot products, normalization and cosine similarity.
pub mod vector;

pub use chunker::{ChunkConfig, SemanticChunker};
pub use config::RagConfig;
pub use error::RagError;
pub use parser::{Parser, ParserRegistry};
#[cfg(feature = "native")]
pub use raglite::{
    Check, IngestReport, Metadata, QueryFilter, RAGLite, SearchResult, Stats, SyncReport,
    SCHEMA_VERSION,
};
