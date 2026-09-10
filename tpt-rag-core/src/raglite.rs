use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::chunker::{ChunkConfig, Chunker};
use crate::embedding::EmbeddingModel;
use crate::error::Result;
use crate::hnsw::HnswIndex;
use crate::metadata::{ChunkMetadata, MetadataStore};
use crate::parser::get_parser;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    pub source: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub text: String,
    pub source: String,
    pub tags: Vec<String>,
}

pub struct RAGLite {
    index: RwLock<HnswIndex>,
    store: MetadataStore,
    model: EmbeddingModel,
    chunker: Chunker,
    db_path: PathBuf,
}

impl RAGLite {
    pub fn open(path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;

        let model_path = find_model(path);
        let model = if model_path.exists() {
            EmbeddingModel::new(&model_path)?
        } else {
            return Err(crate::error::RagError::Model(format!(
                "ONNX model not found at {}. Place a quantized all-MiniLM-L6-v2 ONNX model in the models/ directory.",
                model_path.display()
            )));
        };

        let store = MetadataStore::open(path)?;
        let hnsw_path = path.join("hnsw.bin");
        let index = if hnsw_path.exists() {
            HnswIndex::load_from_disk(&hnsw_path, model.dimensions, 16, 200)?
        } else {
            HnswIndex::new(model.dimensions, 16, 200)
        };
        let chunker = Chunker::new(ChunkConfig::default());

        Ok(Self {
            index: RwLock::new(index),
            store,
            model,
            chunker,
            db_path: path.to_path_buf(),
        })
    }

    pub fn open_with_model(path: &Path, model_path: &Path) -> Result<Self> {
        std::fs::create_dir_all(path)?;

        let model = EmbeddingModel::new(model_path)?;
        let store = MetadataStore::open(path)?;
        let hnsw_path = path.join("hnsw.bin");
        let index = if hnsw_path.exists() {
            HnswIndex::load_from_disk(&hnsw_path, model.dimensions, 16, 200)?
        } else {
            HnswIndex::new(model.dimensions, 16, 200)
        };
        let chunker = Chunker::new(ChunkConfig::default());

        Ok(Self {
            index: RwLock::new(index),
            store,
            model,
            chunker,
            db_path: path.to_path_buf(),
        })
    }

    pub fn flush(&self) -> Result<()> {
        let index = self.index.read().map_err(|e| {
            crate::error::RagError::Hnsw(format!("Lock poisoned: {}", e))
        })?;
        let hnsw_path = self.db_path.join("hnsw.bin");
        index.flush(&hnsw_path)
    }

    pub fn add_document(&self, path: &Path, tags: &[&str]) -> Result<()> {
        let parser = get_parser(path).ok_or_else(|| {
            crate::error::RagError::UnsupportedFormat(
                path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
            )
        })?;

        let text = parser.parse(path)?;
        if text.trim().is_empty() {
            return Err(crate::error::RagError::EmptyDocument {
                path: path.to_path_buf(),
            });
        }

        let chunks = self.chunker.chunk(&text);
        let source = path.to_string_lossy().to_string();
        let tag_vec: Vec<String> = tags.iter().map(|s| s.to_string()).collect();

        let mut index = self.index.write().map_err(|e| {
            crate::error::RagError::Hnsw(format!("Lock poisoned: {}", e))
        })?;

        for chunk in &chunks {
            let embeddings = self.model.embed_batch(&[chunk.text.as_str()])?;
            let embedding = &embeddings[0];
            let vector_id = index.insert(embedding.clone());

            self.store.insert_chunk(&ChunkMetadata {
                vector_id,
                text: chunk.text.clone(),
                source_path: source.clone(),
                created_at: chrono_now(),
                tags: tag_vec.clone(),
            })?;
        }

        Ok(())
    }

    pub fn add_text(&self, text: &str, metadata: &Metadata) -> Result<()> {
        if text.trim().is_empty() {
            return Ok(());
        }

        let chunks = self.chunker.chunk(text);
        let source = metadata.source.clone().unwrap_or_else(|| "inline".to_string());

        let mut index = self.index.write().map_err(|e| {
            crate::error::RagError::Hnsw(format!("Lock poisoned: {}", e))
        })?;

        for chunk in &chunks {
            let embeddings = self.model.embed_batch(&[chunk.text.as_str()])?;
            let embedding = &embeddings[0];
            let vector_id = index.insert(embedding.clone());

            self.store.insert_chunk(&ChunkMetadata {
                vector_id,
                text: chunk.text.clone(),
                source_path: source.clone(),
                created_at: chrono_now(),
                tags: metadata.tags.clone(),
            })?;
        }

        Ok(())
    }

    pub fn query(&self, text: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let embeddings = self.model.embed_batch(&[text])?;
        let query_embedding = &embeddings[0];

        let index = self.index.read().map_err(|e| {
            crate::error::RagError::Hnsw(format!("Lock poisoned: {}", e))
        })?;
        let results = index.search(query_embedding, top_k, 100);
        drop(index);

        let mut search_results = Vec::new();
        for (vector_id, score) in results {
            if let Some(meta) = self.store.get_chunk_by_vector_id(vector_id)? {
                search_results.push(SearchResult {
                    score,
                    text: meta.text,
                    source: meta.source_path,
                    tags: meta.tags,
                });
            }
        }

        Ok(search_results)
    }

    #[cfg(feature = "reranking")]
    pub fn query_reranked(&self, text: &str, top_k: usize, rerank_top_n: usize) -> Result<Vec<SearchResult>> {
        let base_results = self.query(text, top_k)?;
        if base_results.is_empty() {
            return Ok(Vec::new());
        }

        let model_path = find_model(&self.db_path);
        let reranker = crate::reranker::Reranker::new(&model_path)?;

        let chunks: Vec<String> = base_results.iter().map(|r| r.text.clone()).collect();
        let reranked = reranker.rerank(text, &chunks, rerank_top_n)?;

        let mut results = Vec::new();
        for (orig_idx, score) in reranked {
            let base = &base_results[orig_idx];
            results.push(SearchResult {
                score,
                text: base.text.clone(),
                source: base.source.clone(),
                tags: base.tags.clone(),
            });
        }
        Ok(results)
    }

    pub fn query_hybrid(&self, text: &str, top_k: usize, alpha: f32) -> Result<Vec<SearchResult>> {
        let embeddings = self.model.embed_batch(&[text])?;
        let query_embedding = &embeddings[0];

        let index = self.index.read().map_err(|e| {
            crate::error::RagError::Hnsw(format!("Lock poisoned: {}", e))
        })?;
        let vector_results = index.search(query_embedding, top_k * 2, 100);
        drop(index);

        let bm25_results = self.store.bm25_search(text, top_k * 2)?;

        let k = 60.0f32;

        let mut rrf_scores: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();

        for (rank, (vector_id, _)) in vector_results.iter().enumerate() {
            let rrf_contribution = alpha / (k + rank as f32 + 1.0);
            *rrf_scores.entry(*vector_id).or_insert(0.0) += rrf_contribution;
        }

        for (rank, (vector_id, _)) in bm25_results.iter().enumerate() {
            let rrf_contribution = (1.0 - alpha) / (k + rank as f32 + 1.0);
            *rrf_scores.entry(*vector_id).or_insert(0.0) += rrf_contribution;
        }

        let mut scored: Vec<(usize, f32)> = rrf_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let mut results = Vec::new();
        for (vector_id, score) in scored {
            if let Some(meta) = self.store.get_chunk_by_vector_id(vector_id)? {
                results.push(SearchResult {
                    score,
                    text: meta.text,
                    source: meta.source_path,
                    tags: meta.tags,
                });
            }
        }

        Ok(results)
    }

    pub fn chunk_count(&self) -> Result<usize> {
        self.store.count()
    }

    pub fn vector_count(&self) -> usize {
        self.index.read().map(|i| i.len()).unwrap_or(0)
    }
}

fn find_model(db_path: &Path) -> PathBuf {
    let candidates = [
        db_path.join("model.onnx"),
        db_path.join("models").join("all-MiniLM-L6-v2.onnx"),
        std::env::current_dir()
            .unwrap_or_default()
            .join("models")
            .join("all-MiniLM-L6-v2.onnx"),
    ];
    for p in &candidates {
        if p.exists() {
            return p.clone();
        }
    }
    candidates[0].clone()
}

fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{now}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_default() {
        let m = Metadata::default();
        assert!(m.source.is_none());
        assert!(m.tags.is_empty());
    }

    #[test]
    fn test_search_result_fields() {
        let r = SearchResult {
            score: 0.95,
            text: "hello".to_string(),
            source: "test.txt".to_string(),
            tags: vec!["tag".to_string()],
        };
        assert_eq!(r.score, 0.95);
        assert_eq!(r.text, "hello");
    }
}
