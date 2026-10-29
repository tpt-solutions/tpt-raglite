//! The browser engine: a vector index and a keyword index over the same chunks.
//!
//! The host does the parts that need a model. It asks for the chunks of a document, computes
//! one embedding per chunk, and passes the vectors back. Everything else is here: the chunk
//! boundaries (the same as the native library's), the HNSW index, and BM25 keyword scoring.
//! This module is plain Rust, so `cargo test` checks it on the host, and the bindings in
//! `lib.rs` only convert types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tpt_rag_core::chunker::{chunk_document, ChunkConfig, Chunker};
use tpt_rag_core::hnsw::HnswIndex;
use tpt_rag_core::parser::parse_content;

/// HNSW parameters. They match the native library's defaults.
const HNSW_M: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_EF_SEARCH: usize = 100;

/// BM25 parameters, the usual defaults.
/// Reciprocal rank fusion constant from the original RRF paper. Matches the native library.
const RRF_K: f32 = 60.0;

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// Identifies the byte format of `Engine::export`.
const MAGIC: &[u8; 8] = b"TPTWASM1";

/// A chunk the host should embed, and where it sits in the parsed text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedChunk {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    /// Heading path (Markdown) or page (PDF) the chunk starts in. Not used by HTML or text.
    pub section: Option<String>,
}

/// One search result.
#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub chunk_id: usize,
    /// Cosine similarity for `query`, BM25 score for `keyword`.
    pub score: f32,
    pub text: String,
    pub source: String,
    pub tags: Vec<String>,
    pub start_offset: usize,
    pub end_offset: usize,
    pub section: Option<String>,
}

/// A chunk as it is stored. Removed chunks stay in the list, so chunk ids never change.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredChunk {
    text: String,
    source: String,
    tags: Vec<String>,
    start_offset: usize,
    end_offset: usize,
    section: Option<String>,
    live: bool,
}

/// The header of an exported database: everything except the index, which has its own format.
#[derive(Serialize, Deserialize)]
struct Header {
    dimensions: usize,
    chunks: Vec<StoredChunk>,
}

pub struct Engine {
    dimensions: usize,
    index: HnswIndex,
    /// Indexed by chunk id, which is also the vector id in `index`.
    chunks: Vec<StoredChunk>,
    /// Term to `(chunk id, term count)` for every live chunk that has the term.
    postings: HashMap<String, Vec<(usize, u32)>>,
    /// Number of terms in each chunk, indexed by chunk id.
    lengths: Vec<usize>,
    chunker: Chunker,
}

impl Engine {
    pub fn new(dimensions: usize) -> Result<Self, String> {
        if dimensions == 0 {
            return Err("dimensions must be greater than 0".to_string());
        }
        Ok(Self {
            dimensions,
            index: HnswIndex::new(dimensions, HNSW_M, HNSW_EF_CONSTRUCTION),
            chunks: Vec::new(),
            postings: HashMap::new(),
            lengths: Vec::new(),
            chunker: Chunker::new(ChunkConfig::default())
                .expect("the default chunk config is valid"),
        })
    }

    /// Splits content into chunks, without embedding them. `kind` is `"text"`, `"markdown"`
    /// or `"html"`.
    pub fn plan(&self, kind: &str, content: &str) -> Result<Vec<PlannedChunk>, String> {
        let document = parse_content(kind, content).map_err(|e| e.to_string())?;
        if document.text.trim().is_empty() {
            return Err("the document has no text".to_string());
        }
        Ok(chunk_document(&self.chunker, &document)
            .into_iter()
            .map(|chunk| PlannedChunk {
                section: document.section_at(chunk.start_offset),
                text: chunk.text,
                start_offset: chunk.start_offset,
                end_offset: chunk.end_offset,
            })
            .collect())
    }

    /// Stores chunks from `plan` and their embeddings. `vectors` holds one embedding per chunk,
    /// laid out one after another. Chunks already stored for `source` are replaced. Returns the
    /// number of chunks stored.
    pub fn add(
        &mut self,
        source: &str,
        tags: &[String],
        chunks: &[PlannedChunk],
        vectors: &[f32],
    ) -> Result<usize, String> {
        if vectors.len() != chunks.len() * self.dimensions {
            return Err(format!(
                "expected {} values ({} chunks of {} dimensions), got {}",
                chunks.len() * self.dimensions,
                chunks.len(),
                self.dimensions,
                vectors.len()
            ));
        }
        self.delete(source)?;

        for (planned, vector) in chunks.iter().zip(vectors.chunks_exact(self.dimensions)) {
            let id = self
                .index
                .insert(vector.to_vec())
                .map_err(|e| e.to_string())?;
            debug_assert_eq!(id, self.chunks.len());

            let terms = tokenize(&planned.text);
            self.lengths.push(terms.len());
            let mut counts: HashMap<String, u32> = HashMap::new();
            for term in terms {
                *counts.entry(term).or_insert(0) += 1;
            }
            for (term, count) in counts {
                self.postings.entry(term).or_default().push((id, count));
            }

            self.chunks.push(StoredChunk {
                text: planned.text.clone(),
                source: source.to_string(),
                tags: tags.to_vec(),
                start_offset: planned.start_offset,
                end_offset: planned.end_offset,
                section: planned.section.clone(),
                live: true,
            });
        }
        Ok(chunks.len())
    }

    /// Vector and keyword results fused with reciprocal rank fusion, the same method and the
    /// same scale as the native `query_hybrid`: `alpha` is the vector weight from 0 to 1, and
    /// the best possible score is 1.
    pub fn hybrid(
        &self,
        vector: &[f32],
        text: &str,
        top_k: usize,
        alpha: f32,
    ) -> Result<Vec<Hit>, String> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let alpha = if alpha.is_nan() {
            0.5
        } else {
            alpha.clamp(0.0, 1.0)
        };
        let candidate_k = top_k.saturating_mul(2);
        let by_vector = self.query(vector, candidate_k)?;
        let by_keyword = self.keyword(text, candidate_k);

        let mut fused: HashMap<usize, f32> = HashMap::new();
        for (rank, hit) in by_vector.iter().enumerate() {
            *fused.entry(hit.chunk_id).or_insert(0.0) += alpha / (RRF_K + rank as f32 + 1.0);
        }
        for (rank, hit) in by_keyword.iter().enumerate() {
            *fused.entry(hit.chunk_id).or_insert(0.0) +=
                (1.0 - alpha) / (RRF_K + rank as f32 + 1.0);
        }

        let scale = RRF_K + 1.0;
        let mut ranked: Vec<(usize, f32)> = fused
            .into_iter()
            .map(|(id, score)| (id, score * scale))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(top_k);
        Ok(ranked
            .into_iter()
            .filter_map(|(id, score)| self.hit(id, score))
            .collect())
    }

    /// Removes every chunk stored for `source`. Returns how many were removed.
    pub fn delete(&mut self, source: &str) -> Result<usize, String> {
        let mut removed = 0;
        for id in 0..self.chunks.len() {
            if self.chunks[id].live && self.chunks[id].source == source {
                self.chunks[id].live = false;
                self.index.remove(id);
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// The `top_k` chunks whose embeddings are closest to `vector`, best first.
    pub fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<Hit>, String> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let found = self
            .index
            .search(vector, top_k, HNSW_EF_SEARCH.max(top_k))
            .map_err(|e| e.to_string())?;
        Ok(found
            .into_iter()
            .filter_map(|(id, score)| self.hit(id, score))
            .collect())
    }

    /// The `top_k` chunks that best match the words of `text`, by BM25. Needs no embedding.
    pub fn keyword(&self, text: &str, top_k: usize) -> Vec<Hit> {
        let live: Vec<usize> = (0..self.chunks.len())
            .filter(|&id| self.is_live(id))
            .collect();
        if top_k == 0 || live.is_empty() {
            return Vec::new();
        }
        let average_length =
            live.iter().map(|&id| self.lengths[id] as f32).sum::<f32>() / live.len() as f32;
        let average_length = average_length.max(1.0);

        let mut terms = tokenize(text);
        terms.sort();
        terms.dedup();

        let mut scores: HashMap<usize, f32> = HashMap::new();
        for term in &terms {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let live_postings: Vec<&(usize, u32)> = postings
                .iter()
                .filter(|(id, _)| self.is_live(*id))
                .collect();
            let documents = live_postings.len() as f32;
            let total = live.len() as f32;
            let idf = ((total - documents + 0.5) / (documents + 0.5) + 1.0).ln();
            for &&(id, count) in &live_postings {
                let count = count as f32;
                let length = self.lengths[id] as f32;
                let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * length / average_length);
                let score = idf * count * (BM25_K1 + 1.0) / (count + norm);
                *scores.entry(id).or_insert(0.0) += score;
            }
        }

        let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(top_k);
        ranked
            .into_iter()
            .filter_map(|(id, score)| self.hit(id, score))
            .collect()
    }

    /// Chunk, source and vector counts, and the vector size.
    pub fn stats(&self) -> (usize, usize) {
        let live = self.chunks.iter().filter(|c| c.live).count();
        (live, self.dimensions)
    }

    /// The whole database as bytes. Keep the bytes (for example in IndexedDB) and load them
    /// again with `import`.
    pub fn export(&self) -> Result<Vec<u8>, String> {
        let header = serde_json::to_vec(&Header {
            dimensions: self.dimensions,
            chunks: self.chunks.clone(),
        })
        .map_err(|e| e.to_string())?;
        let index = self.index.to_bytes().map_err(|e| e.to_string())?;

        let mut out = Vec::with_capacity(16 + header.len() + index.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&index);
        Ok(out)
    }

    /// Loads a database written by `export`. The keyword index is rebuilt from the chunk text.
    pub fn import(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err("not a tpt-raglite browser database".to_string());
        }
        let mut length = [0u8; 8];
        length.copy_from_slice(&bytes[8..16]);
        let header_end = 16usize
            .checked_add(u64::from_le_bytes(length) as usize)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| "the database is truncated".to_string())?;
        let header: Header =
            serde_json::from_slice(&bytes[16..header_end]).map_err(|e| e.to_string())?;

        let index = HnswIndex::from_bytes(
            &bytes[header_end..],
            header.dimensions,
            HNSW_M,
            HNSW_EF_CONSTRUCTION,
        )
        .map_err(|e| e.to_string())?;
        let mut engine = Engine {
            dimensions: header.dimensions,
            index,
            chunks: header.chunks,
            postings: HashMap::new(),
            lengths: Vec::new(),
            chunker: Chunker::new(ChunkConfig::default())
                .expect("the default chunk config is valid"),
        };
        for id in 0..engine.chunks.len() {
            let terms = tokenize(&engine.chunks[id].text);
            engine.lengths.push(terms.len());
            if !engine.chunks[id].live {
                continue;
            }
            let mut counts: HashMap<String, u32> = HashMap::new();
            for term in terms {
                *counts.entry(term).or_insert(0) += 1;
            }
            for (term, count) in counts {
                engine.postings.entry(term).or_default().push((id, count));
            }
        }
        Ok(engine)
    }

    fn is_live(&self, id: usize) -> bool {
        self.chunks.get(id).is_some_and(|c| c.live)
    }

    fn hit(&self, id: usize, score: f32) -> Option<Hit> {
        let chunk = self.chunks.get(id).filter(|c| c.live)?;
        Some(Hit {
            chunk_id: id,
            score,
            text: chunk.text.clone(),
            source: chunk.source.clone(),
            tags: chunk.tags.clone(),
            start_offset: chunk.start_offset,
            end_offset: chunk.end_offset,
            section: chunk.section.clone(),
        })
    }
}

/// Lower-case words, split at anything that is not a letter or a digit.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOTES: &str = "# Setup\n\nBackups run every night at two.\n\n# Garden\n\nTomato plants need full sun and regular watering.\n";

    /// Stores every chunk of `text` with a made-up embedding: `axis` picks one of three axes.
    fn store(
        engine: &mut Engine,
        kind: &str,
        source: &str,
        text: &str,
        axis: usize,
    ) -> Vec<PlannedChunk> {
        let planned = engine.plan(kind, text).unwrap();
        let mut vectors = Vec::new();
        for _ in &planned {
            let mut v = vec![0.0f32; engine.dimensions];
            v[axis] = 1.0;
            vectors.extend(v);
        }
        engine.add(source, &[], &planned, &vectors).unwrap();
        planned
    }

    #[test]
    fn plan_splits_at_headings_and_reports_sections() {
        let engine = Engine::new(3).unwrap();
        let planned = engine.plan("markdown", NOTES).unwrap();
        assert!(planned
            .iter()
            .any(|c| c.section.as_deref() == Some("Setup")));
        assert!(planned
            .iter()
            .any(|c| c.section.as_deref() == Some("Garden")));
        // No chunk spans both sections.
        for chunk in &planned {
            let both = chunk.text.contains("Backups") && chunk.text.contains("Tomato");
            assert!(!both, "{:?}", chunk.text);
        }
        // Offsets point into the parsed text, so they give the chunk back.
        let parsed = parse_content("markdown", NOTES).unwrap().text;
        for chunk in &planned {
            assert_eq!(&parsed[chunk.start_offset..chunk.end_offset], chunk.text);
        }
    }

    #[test]
    fn query_returns_the_nearest_chunk() {
        let mut engine = Engine::new(3).unwrap();
        store(
            &mut engine,
            "text",
            "space",
            "Jupiter is the largest planet.",
            0,
        );
        store(&mut engine, "text", "garden", "Tomatoes need sun.", 1);
        let hits = engine.query(&[0.9, 0.1, 0.0], 1).unwrap();
        assert_eq!(hits[0].source, "space");
        assert!(hits[0].score > 0.9);
    }

    #[test]
    fn keyword_search_ranks_by_matching_terms() {
        let mut engine = Engine::new(3).unwrap();
        store(&mut engine, "text", "ops", "Backups run every night.", 0);
        store(
            &mut engine,
            "text",
            "garden",
            "Tomato plants need sun and water.",
            1,
        );
        store(&mut engine, "text", "misc", "Nothing relevant here.", 2);

        let hits = engine.keyword("tomato water", 3);
        assert_eq!(hits[0].source, "garden");
        assert!(hits.iter().all(|h| h.source != "misc"));
        assert!(engine.keyword("zebra", 3).is_empty());
    }

    #[test]
    fn adding_a_source_again_replaces_its_chunks() {
        let mut engine = Engine::new(3).unwrap();
        store(&mut engine, "text", "ops", "Backups run every night.", 0);
        store(&mut engine, "text", "ops", "Backups run every week.", 0);

        assert_eq!(engine.stats(), (1, 3));
        let hits = engine.keyword("nightly backups", 5);
        assert!(hits.iter().all(|h| !h.text.contains("night")));
        assert!(engine
            .keyword("week", 5)
            .iter()
            .any(|h| h.text.contains("week")));
    }

    #[test]
    fn delete_removes_chunks_from_both_indexes() {
        let mut engine = Engine::new(3).unwrap();
        store(&mut engine, "text", "ops", "Backups run every night.", 0);
        store(&mut engine, "text", "garden", "Tomatoes need sun.", 1);

        assert_eq!(engine.delete("ops").unwrap(), 1);
        assert_eq!(engine.stats(), (1, 3));
        assert!(engine.keyword("backups", 5).is_empty());
        assert!(engine
            .query(&[1.0, 0.0, 0.0], 5)
            .unwrap()
            .iter()
            .all(|h| h.source != "ops"));
    }

    #[test]
    fn export_and_import_give_the_same_answers() {
        let mut engine = Engine::new(3).unwrap();
        store(&mut engine, "markdown", "notes", NOTES, 0);
        store(
            &mut engine,
            "text",
            "space",
            "Jupiter is the largest planet.",
            1,
        );
        engine.delete("space").unwrap();

        let bytes = engine.export().unwrap();
        let loaded = Engine::import(&bytes).unwrap();
        assert_eq!(loaded.stats(), engine.stats());
        let before = engine.keyword("backups watering", 5);
        let after = loaded.keyword("backups watering", 5);
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.text, b.text);
            assert!((a.score - b.score).abs() < 1e-6);
        }
        let query = loaded.query(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(query[0].section.as_deref(), Some("Setup"));
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        assert!(Engine::new(0).is_err());
        let mut engine = Engine::new(3).unwrap();
        assert!(engine.plan("pdf", "anything").is_err());
        assert!(engine.plan("text", "   ").is_err());
        let planned = engine.plan("text", "One sentence about gardens.").unwrap();
        assert!(
            engine.add("x", &[], &planned, &[1.0]).is_err(),
            "wrong vector size"
        );
        assert!(Engine::import(b"not a database").is_err());
        assert!(Engine::import(&[0u8; 4]).is_err());
    }

    #[test]
    fn hybrid_follows_the_vector_side_at_one_and_the_keyword_side_at_zero() {
        let mut engine = Engine::new(3).unwrap();
        // The vector matches the query exactly, but no word matches. The second chunk has the
        // word but a vector pointing elsewhere.
        store(&mut engine, "text", "vector", "Alpha beta gamma.", 0);
        store(&mut engine, "text", "keyword", "Tomato plants need sun.", 1);

        let by_vector = engine.hybrid(&[1.0, 0.0, 0.0], "tomato", 2, 1.0).unwrap();
        assert_eq!(by_vector[0].source, "vector");
        let by_keyword = engine.hybrid(&[1.0, 0.0, 0.0], "tomato", 2, 0.0).unwrap();
        assert_eq!(by_keyword[0].source, "keyword");

        let fused = engine.hybrid(&[1.0, 0.0, 0.0], "tomato", 2, 0.5).unwrap();
        assert_eq!(fused.len(), 2);
        assert!(fused[0].score > 0.0 && fused[0].score <= 1.0);
        assert!(engine
            .hybrid(&[1.0, 0.0, 0.0], "tomato", 0, 0.5)
            .unwrap()
            .is_empty());
    }
}
