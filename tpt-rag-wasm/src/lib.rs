//! WebAssembly bindings for tpt-raglite.
//!
//! The browser cannot run the native embedding model, so the host computes the embeddings. The
//! flow is:
//!
//! 1. `plan(kind, content)` splits a document into chunks. `kind` is `"text"`, `"markdown"` or
//!    `"html"`. The chunks are JSON: `{text, start_offset, end_offset, section}`.
//! 2. The host embeds each chunk's `text` with a model of the same dimension, such as the
//!    quantized all-MiniLM-L6-v2 (384 dimensions), and collects the vectors in one Float32Array.
//! 3. `add(source, tags, chunksJson, vectors)` stores the chunks. A source that is already
//!    stored is replaced.
//! 4. `query(vector, topK)` searches by embedding, and `keyword(text, topK)` searches by BM25
//!    with no model at all.
//! 5. `export()` returns the whole database as bytes. Keep them in IndexedDB or OPFS, and load
//!    them again with `TptRag.import(bytes)`.
//!
//! Results are returned as JSON strings. Errors are thrown as JavaScript strings.

mod engine;

use engine::{Engine, PlannedChunk};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TptRag {
    engine: Engine,
}

fn to_js_error(message: String) -> JsValue {
    JsValue::from_str(&message)
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<String, JsValue> {
    serde_json::to_string(value).map_err(|e| to_js_error(e.to_string()))
}

#[wasm_bindgen]
impl TptRag {
    /// An empty database for embeddings of `dimensions` values each.
    #[wasm_bindgen(constructor)]
    pub fn new(dimensions: usize) -> Result<TptRag, JsValue> {
        let engine = Engine::new(dimensions).map_err(to_js_error)?;
        Ok(TptRag { engine })
    }

    /// Splits a document into chunks, without embedding them. Returns JSON.
    pub fn plan(&self, kind: &str, content: &str) -> Result<String, JsValue> {
        let chunks = self.engine.plan(kind, content).map_err(to_js_error)?;
        to_json(&chunks)
    }

    /// Stores chunks from `plan` with their embeddings: one vector of `dimensions` values per
    /// chunk, laid out one after another. Replaces chunks already stored for `source`. Returns
    /// the number of chunks stored.
    pub fn add(
        &mut self,
        source: &str,
        tags: Vec<String>,
        chunks_json: &str,
        vectors: &[f32],
    ) -> Result<usize, JsValue> {
        let chunks: Vec<PlannedChunk> =
            serde_json::from_str(chunks_json).map_err(|e| to_js_error(e.to_string()))?;
        self.engine
            .add(source, &tags, &chunks, vectors)
            .map_err(to_js_error)
    }

    /// The `topK` chunks nearest to `vector`, best first. Returns JSON.
    pub fn query(&self, vector: &[f32], top_k: usize) -> Result<String, JsValue> {
        let hits = self.engine.query(vector, top_k).map_err(to_js_error)?;
        to_json(&hits)
    }

    /// The `topK` chunks that best match the words of `text`, by BM25. Returns JSON.
    pub fn keyword(&self, text: &str, top_k: usize) -> Result<String, JsValue> {
        to_json(&self.engine.keyword(text, top_k))
    }

    /// Vector and keyword results fused with reciprocal rank fusion, as in the native
    /// `query_hybrid`. `alpha` is the vector weight from 0 to 1. Returns JSON.
    pub fn hybrid(
        &self,
        vector: &[f32],
        text: &str,
        top_k: usize,
        alpha: f32,
    ) -> Result<String, JsValue> {
        let hits = self
            .engine
            .hybrid(vector, text, top_k, alpha)
            .map_err(to_js_error)?;
        to_json(&hits)
    }

    /// Removes every chunk stored for `source`. Returns how many were removed.
    pub fn delete(&mut self, source: &str) -> Result<usize, JsValue> {
        self.engine.delete(source).map_err(to_js_error)
    }

    /// Chunk count and vector size, as JSON: `{"chunks": n, "dimensions": d}`.
    pub fn stats(&self) -> Result<String, JsValue> {
        let (chunks, dimensions) = self.engine.stats();
        to_json(&serde_json::json!({ "chunks": chunks, "dimensions": dimensions }))
    }

    /// The whole database as bytes.
    pub fn export(&self) -> Result<Vec<u8>, JsValue> {
        self.engine.export().map_err(to_js_error)
    }

    /// Loads bytes written by `export`.
    pub fn import(bytes: &[u8]) -> Result<TptRag, JsValue> {
        let engine = Engine::import(bytes).map_err(to_js_error)?;
        Ok(TptRag { engine })
    }
}
