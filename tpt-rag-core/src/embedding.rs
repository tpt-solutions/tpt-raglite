use std::path::Path;
use std::sync::Mutex;

use crate::error::{RagError, Result};
use crate::tokenizer::WordPieceTokenizer;

/// Longest input the model accepts (BERT-style position embeddings).
pub(crate) const MAX_SEQ_LEN: usize = 512;

/// Texts per ONNX run. Larger batches are faster per text but need more memory.
pub(crate) const DEFAULT_BATCH_SIZE: usize = 32;

/// Loads an ONNX session and the `vocab.txt` that sits next to the model file.
pub(crate) struct LoadedModel {
    pub session: ort::session::Session,
    pub tokenizer: WordPieceTokenizer,
    pub input_name: String,
    pub output_name: String,
    pub has_token_type_ids: bool,
}

pub(crate) fn load_model(model_path: &Path) -> Result<LoadedModel> {
    let vocab_path = model_path.with_file_name("vocab.txt");
    if !vocab_path.exists() {
        return Err(RagError::Model(format!(
            "vocab.txt not found at {}. Place the vocabulary file from all-MiniLM-L6-v2 next to the model.",
            vocab_path.display()
        )));
    }
    let tokenizer = WordPieceTokenizer::from_vocab_file(&vocab_path)?;
    let session = ort::session::Session::builder()
        .map_err(|e| RagError::Onnx(e.to_string()))?
        .commit_from_file(model_path)
        .map_err(|e| RagError::Onnx(e.to_string()))?;
    describe(session, tokenizer)
}

/// Loads a model whose bytes are compiled into the binary (the `embedded-model` feature).
#[cfg(feature = "embedded-model")]
pub(crate) fn load_embedded_model() -> Result<LoadedModel> {
    static MODEL: &[u8] = include_bytes!("../../models/all-MiniLM-L6-v2.onnx");
    static VOCAB: &str = include_str!("../../models/vocab.txt");
    let tokenizer = WordPieceTokenizer::from_vocab(VOCAB.lines())?;
    let session = ort::session::Session::builder()
        .map_err(|e| RagError::Onnx(e.to_string()))?
        .commit_from_memory(MODEL)
        .map_err(|e| RagError::Onnx(e.to_string()))?;
    describe(session, tokenizer)
}

fn describe(session: ort::session::Session, tokenizer: WordPieceTokenizer) -> Result<LoadedModel> {
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
    // Some exported models have no token_type_ids input. Passing it to those fails.
    let has_token_type_ids = session
        .inputs()
        .iter()
        .any(|input| input.name() == "token_type_ids");

    Ok(LoadedModel {
        session,
        tokenizer,
        input_name,
        output_name,
        has_token_type_ids,
    })
}

/// Turns a session output into one vector per row. Token-level output `[rows, seq, hidden]`
/// is mean-pooled over the positions the attention mask marks as real tokens. Sentence-level
/// output `[rows, hidden]` is taken as is.
pub(crate) fn pool_output(
    shape: &[i64],
    data: &[f32],
    rows: usize,
    seq_len: usize,
    mask: &[i64],
) -> Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::with_capacity(rows);
    match shape.len() {
        3 => {
            let seq_out = shape[1] as usize;
            let hidden = shape[2] as usize;
            if shape[0] as usize != rows
                || seq_out != seq_len
                || data.len() < rows * seq_out * hidden
            {
                return Err(RagError::Onnx(format!(
                    "Output shape {shape:?} does not match a batch of {rows} sequences of length {seq_len}"
                )));
            }
            for r in 0..rows {
                let mut pooled = vec![0.0f32; hidden];
                let mut count = 0.0f32;
                for s in 0..seq_len {
                    if mask[r * seq_len + s] == 0 {
                        continue;
                    }
                    count += 1.0;
                    let base = (r * seq_len + s) * hidden;
                    for (h, value) in pooled.iter_mut().enumerate() {
                        *value += data[base + h];
                    }
                }
                if count > 0.0 {
                    for value in &mut pooled {
                        *value /= count;
                    }
                }
                vectors.push(pooled);
            }
        }
        2 => {
            let hidden = shape[1] as usize;
            if shape[0] as usize != rows || data.len() < rows * hidden {
                return Err(RagError::Onnx(format!(
                    "Output shape {shape:?} does not match a batch of {rows} sequences"
                )));
            }
            for r in 0..rows {
                vectors.push(data[r * hidden..(r + 1) * hidden].to_vec());
            }
        }
        _ => {
            return Err(RagError::Onnx(format!(
                "Unexpected output shape: {shape:?}"
            )));
        }
    }
    Ok(vectors)
}

/// Scales `vector` to unit length, so cosine similarity is a plain dot product.
pub(crate) fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vector {
            *x /= norm;
        }
    }
}

/// Pads encoded sequences to a common length and builds the input tensors' data.
/// Returns `(input_ids, attention_mask, seq_len)`, all row-major.
pub(crate) fn pad_batch(sequences: &[Vec<i64>], pad_id: i64) -> (Vec<i64>, Vec<i64>, usize) {
    let seq_len = sequences.iter().map(Vec::len).max().unwrap_or(0);
    let mut ids = Vec::with_capacity(sequences.len() * seq_len);
    let mut mask = Vec::with_capacity(sequences.len() * seq_len);
    for seq in sequences {
        let padding = seq_len - seq.len();
        ids.extend_from_slice(seq);
        ids.extend(std::iter::repeat_n(pad_id, padding));
        mask.extend(std::iter::repeat_n(1i64, seq.len()));
        mask.extend(std::iter::repeat_n(0i64, padding));
    }
    (ids, mask, seq_len)
}

pub struct EmbeddingModel {
    session: Mutex<ort::session::Session>,
    tokenizer: WordPieceTokenizer,
    input_name: String,
    output_name: String,
    has_token_type_ids: bool,
    batch_size: usize,
    /// Output size, read from the model when it is loaded.
    pub dimensions: usize,
}

impl EmbeddingModel {
    /// Loads the model and checks its output: it must produce a non-empty float32 vector
    /// for each input. The vector size is taken from the model, not assumed.
    pub fn new(model_path: &Path) -> Result<Self> {
        Self::from_loaded(load_model(model_path)?, &model_path.display().to_string())
    }

    /// The model compiled into the binary. Needs the `embedded-model` feature.
    #[cfg(feature = "embedded-model")]
    pub fn embedded() -> Result<Self> {
        Self::from_loaded(load_embedded_model()?, "embedded all-MiniLM-L6-v2")
    }

    fn from_loaded(loaded: LoadedModel, origin: &str) -> Result<Self> {
        let mut model = Self {
            session: Mutex::new(loaded.session),
            tokenizer: loaded.tokenizer,
            input_name: loaded.input_name,
            output_name: loaded.output_name,
            has_token_type_ids: loaded.has_token_type_ids,
            batch_size: DEFAULT_BATCH_SIZE,
            dimensions: 0,
        };

        let probe = model.run_batch(&["probe"])?;
        let dimensions = probe.first().map_or(0, Vec::len);
        if dimensions == 0 {
            return Err(RagError::Model(format!(
                "model {origin} produced an empty embedding"
            )));
        }
        model.dimensions = dimensions;
        Ok(model)
    }

    /// Embeds one text. The vector is unit length.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embed_batch(&[text])?;
        vectors
            .pop()
            .ok_or_else(|| RagError::Onnx("model returned no embedding".into()))
    }

    /// Embeds many texts with batched ONNX runs. Texts are grouped by length so that each
    /// batch pads to a similar size. The result keeps the order of `texts`.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| texts[i].len());

        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        for group in order.chunks(self.batch_size) {
            let batch: Vec<&str> = group.iter().map(|&i| texts[i]).collect();
            let vectors = self.run_batch(&batch)?;
            for (&index, vector) in group.iter().zip(vectors) {
                if vector.len() != self.dimensions {
                    return Err(RagError::Onnx(format!(
                        "Model produced {} dimensions, expected {}",
                        vector.len(),
                        self.dimensions
                    )));
                }
                results[index] = Some(vector);
            }
        }

        results
            .into_iter()
            .map(|v| v.ok_or_else(|| RagError::Onnx("missing embedding in batch".into())))
            .collect()
    }

    /// Runs one ONNX session call over `texts` and returns normalized vectors.
    fn run_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encoded: Vec<Vec<i64>> = texts
            .iter()
            .map(|t| self.tokenizer.encode(t, MAX_SEQ_LEN))
            .collect();
        let rows = encoded.len();
        let (input_ids, attention_mask, seq_len) = pad_batch(&encoded, self.tokenizer.pad_id());
        let shape = [rows, seq_len];

        let ids_tensor = ort::value::Tensor::from_array((shape, input_ids))
            .map_err(|e| RagError::Onnx(e.to_string()))?;
        let mask_tensor = ort::value::Tensor::from_array((shape, attention_mask.clone()))
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| RagError::Onnx(e.to_string()))?;
        let outputs = if self.has_token_type_ids {
            let type_tensor = ort::value::Tensor::from_array((shape, vec![0i64; rows * seq_len]))
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
        let (out_shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| RagError::Onnx(e.to_string()))?;

        let mut vectors = pool_output(out_shape, data, rows, seq_len, &attention_mask)?;
        for vector in &mut vectors {
            normalize(vector);
        }
        Ok(vectors)
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot(a, b) / (norm_a * norm_b)
    }
}

/// Dot product of the common prefix of `a` and `b`. The sum is kept in eight independent
/// lanes. A single running sum is a chain of dependent additions that the compiler cannot
/// vectorize, and this is the hot loop of the index.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let main = n - n % 8;
    let (a_main, a_tail) = a[..n].split_at(main);
    let (b_main, b_tail) = b[..n].split_at(main);

    let mut lanes = [0.0f32; 8];
    for (ca, cb) in a_main.chunks_exact(8).zip(b_main.chunks_exact(8)) {
        for (lane, (x, y)) in lanes.iter_mut().zip(ca.iter().zip(cb)) {
            *lane += x * y;
        }
    }
    let tail: f32 = a_tail.iter().zip(b_tail).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f32>() + tail
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
    fn test_pool_output_masks_padding() {
        // Two sequences of length 3. Sequence 0 has one padding position, sequence 1 has none.
        // Position values are chosen so that including padding would change the mean.
        let shape = [2i64, 3, 2];
        let data = [
            1.0, 1.0, // seq 0, token 0
            3.0, 3.0, // seq 0, token 1
            100.0, 100.0, // seq 0, padding
            2.0, 2.0, // seq 1, token 0
            4.0, 4.0, // seq 1, token 1
            6.0, 6.0, // seq 1, token 2
        ];
        let mask = [1, 1, 0, 1, 1, 1];
        let pooled = pool_output(&shape, &data, 2, 3, &mask).unwrap();
        assert_eq!(pooled[0], vec![2.0, 2.0]);
        assert_eq!(pooled[1], vec![4.0, 4.0]);
    }

    #[test]
    fn test_pool_output_sentence_level() {
        let shape = [2i64, 2];
        let data = [1.0, 2.0, 3.0, 4.0];
        let pooled = pool_output(&shape, &data, 2, 5, &[]).unwrap();
        assert_eq!(pooled, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn test_pool_output_rejects_mismatched_shape() {
        let shape = [1i64, 4, 2];
        let data = [0.0f32; 8];
        assert!(pool_output(&shape, &data, 2, 4, &[1; 8]).is_err());
        assert!(pool_output(&[1i64, 2, 2, 2], &data, 1, 2, &[1; 2]).is_err());
    }

    #[test]
    fn test_pad_batch() {
        let seqs = vec![vec![101, 5, 102], vec![101, 102]];
        let (ids, mask, seq_len) = pad_batch(&seqs, 0);
        assert_eq!(seq_len, 3);
        assert_eq!(ids, vec![101, 5, 102, 101, 102, 0]);
        assert_eq!(mask, vec![1, 1, 1, 1, 1, 0]);
    }

    #[test]
    fn test_dot_matches_naive_sum_at_every_length() {
        for len in 0..40 {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();
            let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - naive).abs() < 1e-4, "length {len}");
        }
    }

    #[test]
    fn test_normalize_gives_unit_length() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        let mut zero = vec![0.0, 0.0];
        normalize(&mut zero);
        assert_eq!(zero, vec![0.0, 0.0]);
    }
}
