use std::path::Path;
use std::sync::Mutex;

use rayon::prelude::*;

use crate::error::{RagError, Result};

pub struct EmbeddingModel {
    session: Mutex<ort::session::Session>,
    input_name: String,
    pub dimensions: usize,
}

impl EmbeddingModel {
    pub fn new(model_path: &Path) -> Result<Self> {
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

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            dimensions: 384,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = tokenize(text);
        let shape = vec![1i64, tokens.len() as i64];
        let input_tensor = ort::value::Tensor::from_array((shape, tokens))
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let mut session = self.session.lock().map_err(|e| RagError::Onnx(e.to_string()))?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => input_tensor])
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let output = &outputs[0];
        let extracted = output
            .try_extract_array::<f32>()
            .map_err(|e| RagError::Onnx(e.to_string()))?;
        Ok(extracted.as_slice().unwrap_or_default().to_vec())
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts
            .par_iter()
            .map(|text| self.embed(text))
            .collect()
    }
}

fn tokenize(text: &str) -> Vec<i64> {
    let mut tokens = vec![101i64]; // [CLS]
    for word in text.split_whitespace() {
        for byte in word.bytes() {
            tokens.push(byte as i64);
        }
        tokens.push(102i64); // [SEP] as word separator
    }
    tokens.push(102i64);
    tokens
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
        let tokens = tokenize("hello world");
        assert_eq!(tokens[0], 101); // CLS
        assert!(tokens.len() > 3);
    }
}
