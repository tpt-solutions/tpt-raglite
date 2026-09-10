use std::path::Path;
use std::sync::Mutex;

use crate::embedding::{cosine_similarity, EmbeddingModel};
use crate::error::{RagError, Result};

pub struct Reranker {
    embedding_model: Mutex<EmbeddingModel>,
}

impl Reranker {
    pub fn new(model_path: &Path) -> Result<Self> {
        let embedding_model = EmbeddingModel::new(model_path)?;

        Ok(Self {
            embedding_model: Mutex::new(embedding_model),
        })
    }

    pub fn rerank(&self, query: &str, chunks: &[String], top_n: usize) -> Result<Vec<(usize, f32)>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let embedding_model = self
            .embedding_model
            .lock()
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let mut all_texts: Vec<&str> = Vec::with_capacity(chunks.len() + 1);
        all_texts.push(query);
        all_texts.extend(chunks.iter().map(|s| s.as_str()));

        let embeddings = embedding_model.embed_batch(&all_texts)?;

        let query_embedding = &embeddings[0];
        let chunk_embeddings = &embeddings[1..];

        let mut scores: Vec<(usize, f32)> = chunk_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let sim = cosine_similarity(query_embedding, emb);
                (i, sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_n);

        Ok(scores)
    }
}
