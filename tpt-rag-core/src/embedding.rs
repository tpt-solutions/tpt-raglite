use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Mutex;

use crate::error::{RagError, Result};

pub struct EmbeddingModel {
    session: Mutex<ort::session::Session>,
    input_name: String,
    output_name: String,
    pub dimensions: usize,
}

impl EmbeddingModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        Self::new_with_dimensions(model_path, 384)
    }

    pub fn new_with_dimensions(model_path: &Path, dimensions: usize) -> Result<Self> {
        let session = ort::session::Session::builder()
            .map_err(|e| RagError::Onnx(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| RagError::Model("No inputs found in ONNX model".into()))?
            .name()
            .to_string();

        let output_name = session
            .outputs()
            .first()
            .ok_or_else(|| RagError::Model("No outputs found in ONNX model".into()))?
            .name()
            .to_string();

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            output_name,
            dimensions,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = tokenize(text, 512);
        let seq_len = tokens.len();

        let input_ids: Vec<i64> = tokens.iter().map(|t| t.0).collect();
        let attention_mask: Vec<i64> = tokens.iter().map(|t| t.1).collect();
        let token_type_ids: Vec<i64> = tokens.iter().map(|_| 0i64).collect();

        let attention_mask_for_pool = attention_mask.clone();

        let input_ids_tensor = ort::value::Tensor::from_array(([1, seq_len], input_ids))
            .map_err(|e| RagError::Onnx(e.to_string()))?;
        let attention_mask_tensor = ort::value::Tensor::from_array(([1, seq_len], attention_mask))
            .map_err(|e| RagError::Onnx(e.to_string()))?;
        let token_type_ids_tensor = ort::value::Tensor::from_array(([1, seq_len], token_type_ids))
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let mut session = self.session.lock().map_err(|e| RagError::Onnx(e.to_string()))?;
        let outputs = session
            .run(ort::inputs![
                self.input_name.as_str() => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let output = outputs
            .get(&*self.output_name)
            .ok_or_else(|| RagError::Onnx("Output not found".into()))?;

        let extracted = output
            .try_extract_tensor::<f32>()
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let (shape, data) = extracted;

        let pooled = match shape.len() {
            3 => {
                let seq_len_out = shape[1] as usize;
                let hidden = shape[2] as usize;
                let mut pooled = vec![0.0f32; hidden];
                let mut count = 0.0f32;
                for s in 0..seq_len_out {
                    if s < attention_mask_for_pool.len() && attention_mask_for_pool[s] == 1 {
                        for h in 0..hidden {
                            pooled[h] += data[h + s * hidden];
                        }
                        count += 1.0;
                    }
                }
                if count > 0.0 {
                    for h in &mut pooled {
                        *h /= count;
                    }
                }
                pooled
            }
            2 => {
                data.to_vec()
            }
            _ => {
                return Err(RagError::Onnx(format!(
                    "Unexpected output shape: {:?}",
                    shape
                )));
            }
        };

        Ok(pooled)
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }
}

/// Tokenize text into (token_id, attention_mask) pairs.
/// Uses hash-based wordpiece approximation — maps words to vocabulary IDs
/// via a deterministic hash within the valid token range.
fn tokenize(text: &str, max_len: usize) -> Vec<(i64, i64)> {
    let vocab_size = 30_522i64;
    let cls = 101i64;
    let sep = 102i64;
    let pad = 0i64;

    let mut tokens: Vec<(i64, i64)> = Vec::with_capacity(max_len);
    tokens.push((cls, 1));

    for word in text.split_whitespace() {
        if tokens.len() >= (max_len - 1) {
            break;
        }
        let word_lower = word.to_lowercase();
        for subword in wordpiece_split(&word_lower) {
            if tokens.len() >= (max_len - 1) {
                break;
            }
            let id = hash_to_vocab_id(&subword, vocab_size);
            tokens.push((id, 1));
        }
    }

    tokens.push((sep, 1));

    while tokens.len() < max_len {
        tokens.push((pad, 0));
    }

    tokens
}

fn wordpiece_split(word: &str) -> Vec<String> {
    let max_subword_len = 20;
    if word.len() <= max_subword_len {
        return vec![word.to_string()];
    }

    let mut parts = Vec::new();
    let chars: Vec<char> = word.chars().collect();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + max_subword_len).min(chars.len());
        let part: String = chars[start..end].iter().collect();
        parts.push(part);
        start = end;
    }
    parts
}

fn hash_to_vocab_id(word: &str, vocab_size: i64) -> i64 {
    let mut hasher = DefaultHasher::new();
    word.hash(&mut hasher);
    let hash = hasher.finish();
    (hash % (vocab_size as u64 - 2) + 2) as i64
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn tokenize_ids(text: &str) -> Vec<i64> {
    tokenize(text, 512).into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("hello world", 128);
        assert_eq!(tokens[0].0, 101); // CLS
        assert_eq!(tokens[0].1, 1);
        assert!(tokens.len() == 128);
        assert_eq!(tokens[2].0, 102);
        assert_eq!(tokens[2].1, 1);
        assert_eq!(tokens[3].1, 0);
    }

    #[test]
    fn test_tokenize_max_len() {
        let tokens = tokenize("a b c", 4);
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].0, 101); // CLS
        assert_eq!(tokens[3].0, 102); // SEP
    }

    #[test]
    fn test_wordpiece_split() {
        let result = wordpiece_split("hello");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "hello");

        let long_word = "a".repeat(50);
        let result = wordpiece_split(&long_word);
        assert!(result.len() > 1);
    }
}
