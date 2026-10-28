use std::path::Path;
use std::sync::Mutex;

use crate::embedding::{load_model, pad_batch, DEFAULT_BATCH_SIZE, MAX_SEQ_LEN};
use crate::error::{RagError, Result};
use crate::tokenizer::WordPieceTokenizer;

/// A cross-encoder reranker. It reads the query and a passage together as one sequence,
/// which is slower than comparing separate embeddings but scores relevance more accurately.
///
/// The model must be a BERT-style sequence classifier with a single logit output, such as
/// `cross-encoder/ms-marco-MiniLM-L-6-v2` exported to ONNX. Its `vocab.txt` sits next to it.
pub struct Reranker {
    session: Mutex<ort::session::Session>,
    tokenizer: WordPieceTokenizer,
    input_name: String,
    output_name: String,
    has_token_type_ids: bool,
}

impl Reranker {
    /// Loads the cross-encoder once. Load failures are returned, not ignored.
    pub fn new(model_path: &Path) -> Result<Self> {
        let loaded = load_model(model_path)?;
        Ok(Self {
            session: Mutex::new(loaded.session),
            tokenizer: loaded.tokenizer,
            input_name: loaded.input_name,
            output_name: loaded.output_name,
            has_token_type_ids: loaded.has_token_type_ids,
        })
    }

    /// Scores every passage against `query`. Returns up to `top_n` `(passage index, score)`
    /// pairs, best first. Scores are sigmoid(logit), so they lie in (0, 1).
    pub fn rerank(
        &self,
        query: &str,
        passages: &[String],
        top_n: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if passages.is_empty() || top_n == 0 {
            return Ok(Vec::new());
        }

        let mut scores = Vec::with_capacity(passages.len());
        for (offset, batch) in passages.chunks(DEFAULT_BATCH_SIZE).enumerate() {
            let batch_scores = self.score_batch(query, batch)?;
            scores.extend(
                batch_scores
                    .into_iter()
                    .enumerate()
                    .map(|(i, s)| (offset * DEFAULT_BATCH_SIZE + i, s)),
            );
        }

        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        scores.truncate(top_n);
        Ok(scores)
    }

    fn score_batch(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
        let encoded: Vec<(Vec<i64>, Vec<i64>)> = passages
            .iter()
            .map(|p| self.tokenizer.encode_pair(query, p, MAX_SEQ_LEN))
            .collect();
        let rows = encoded.len();

        let ids: Vec<Vec<i64>> = encoded.iter().map(|(ids, _)| ids.clone()).collect();
        let types: Vec<Vec<i64>> = encoded.iter().map(|(_, t)| t.clone()).collect();
        let (input_ids, attention_mask, seq_len) = pad_batch(&ids, self.tokenizer.pad_id());
        // Token types are padded with 0, which pad_batch cannot do for us, so pad here.
        let mut token_types = Vec::with_capacity(rows * seq_len);
        for seq in &types {
            token_types.extend_from_slice(seq);
            token_types.extend(std::iter::repeat_n(0i64, seq_len - seq.len()));
        }
        let shape = [rows, seq_len];

        let ids_tensor = ort::value::Tensor::from_array((shape, input_ids))
            .map_err(|e| RagError::Onnx(e.to_string()))?;
        let mask_tensor = ort::value::Tensor::from_array((shape, attention_mask))
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| RagError::Onnx(e.to_string()))?;
        let outputs = if self.has_token_type_ids {
            let type_tensor = ort::value::Tensor::from_array((shape, token_types))
                .map_err(|e| RagError::Onnx(e.to_string()))?;
            session.run(ort::inputs![
                self.input_name.as_str() => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => type_tensor,
            ])
        } else {
            session.run(ort::inputs![
                self.input_name.as_str() => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
        }
        .map_err(|e| RagError::Onnx(e.to_string()))?;

        let output = outputs
            .get(&*self.output_name)
            .ok_or_else(|| RagError::Onnx("Output not found".into()))?;
        let (_, logits) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| RagError::Onnx(e.to_string()))?;
        if logits.len() < rows {
            return Err(RagError::Onnx(format!(
                "reranker returned {} logits for {rows} passages",
                logits.len()
            )));
        }

        Ok(logits[..rows].iter().map(|&l| sigmoid(l)).collect())
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_range() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
