use std::path::PathBuf;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use tpt_rag_core::{Metadata, RAGLite};

#[wasm_bindgen]
pub struct RAGLiteWasm {
    inner: RwLock<RAGLite>,
}

#[derive(Serialize, Deserialize)]
pub struct JsSearchResult {
    pub score: f32,
    pub text: String,
    pub source: String,
    pub tags: Vec<String>,
}

#[wasm_bindgen]
impl RAGLiteWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(db_name: &str) -> Result<RAGLiteWasm, JsValue> {
        let path = PathBuf::from(db_name);
        let rag = RAGLite::open(&path).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self {
            inner: RwLock::new(rag),
        })
    }

    pub fn add_text(&self, text: &str) -> Result<(), JsValue> {
        let rag = self
            .inner
            .write()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        rag.add_text(text, &Metadata::default())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn query(&self, query_text: &str, top_k: usize) -> Result<JsValue, JsValue> {
        let rag = self
            .inner
            .read()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let results = rag
            .query(query_text, top_k)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let js_results: Vec<JsSearchResult> = results
            .into_iter()
            .map(|r| JsSearchResult {
                score: r.score,
                text: r.text,
                source: r.source,
                tags: r.tags,
            })
            .collect();
        let json = serde_json::to_string(&js_results)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(JsValue::from_str(&json))
    }
}
