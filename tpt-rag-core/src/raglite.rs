use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::chunker::{Chunk, Chunker};
use crate::config::RagConfig;
use crate::embedding::EmbeddingModel;
use crate::error::{RagError, Result};
use crate::hnsw::HnswIndex;
use crate::metadata::{ChunkMetadata, MetadataStore, META_INDEX_EPOCH};
use crate::parser::{Parser, ParserRegistry};
#[cfg(feature = "reranking")]
use crate::reranker::Reranker;

/// Reciprocal rank fusion constant from the original RRF paper.
const RRF_K: f32 = 60.0;
/// Chunks re-embedded per model call when the index is rebuilt.
const REBUILD_BATCH: usize = 256;
const INDEX_FILE: &str = "hnsw.bin";
const META_MODEL: &str = "model";
const META_DIMENSIONS: &str = "dimensions";
const META_SCHEMA_VERSION: &str = "schema_version";

/// Version of the SQLite layout this library writes. A database with a higher version was
/// written by a newer library and is refused, because this library cannot read it safely.
pub const SCHEMA_VERSION: u32 = 1;

/// Restricts a query to chunks that match every condition set here.
#[derive(Debug, Clone, Default)]
pub struct QueryFilter {
    /// Chunks must carry every one of these tags.
    pub tags: Vec<String>,
    /// Chunks whose source path starts with this text.
    pub source_prefix: Option<String>,
    /// Chunks ingested at or after this time, in seconds since the Unix epoch.
    pub created_since: Option<u64>,
}

impl QueryFilter {
    pub fn matches(&self, result: &SearchResult) -> bool {
        self.tags.iter().all(|t| result.tags.contains(t))
            && self
                .source_prefix
                .as_deref()
                .is_none_or(|prefix| result.source.starts_with(prefix))
            && self
                .created_since
                .is_none_or(|since| result.created_at >= since)
    }
}

/// One line of the report from [`RAGLite::check`].
#[derive(Debug, Clone)]
pub struct Check {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    pub source: Option<String>,
    pub tags: Vec<String>,
}

/// One retrieved chunk. Higher `score` is better. The scale depends on the query method:
///
/// - [`RAGLite::query`]: cosine similarity, from -1 to 1.
/// - [`RAGLite::query_hybrid`]: fused rank score, from 0 to 1.
/// - [`RAGLite::query_reranked`]: cross-encoder relevance, from 0 to 1.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub text: String,
    pub source: String,
    pub tags: Vec<String>,
    /// Row id of the chunk in the metadata store. Stable until the document is re-ingested
    /// or the index is rebuilt, so it can be cited within one database.
    pub chunk_id: u64,
    /// When the chunk was ingested, in seconds since the Unix epoch.
    pub created_at: u64,
}

/// Snapshot of what a database contains.
#[derive(Debug, Clone)]
pub struct Stats {
    /// Chunks with metadata. Deleted documents are not counted.
    pub chunks: usize,
    /// Vectors in the index, including those whose chunks were deleted.
    pub vectors: usize,
    /// Size of the index file on disk. 0 until the first flush.
    pub index_bytes: u64,
    /// File name of the embedding model in use.
    pub model: String,
}

/// Outcome of [`RAGLite::add_directory`].
#[derive(Debug, Clone, Default)]
pub struct IngestReport {
    /// Files that were ingested.
    pub added: Vec<PathBuf>,
    /// Files that could not be ingested, with the reason. Other files are still ingested.
    pub failed: Vec<(PathBuf, String)>,
}

/// A local RAG database: one folder holding `metadata.sqlite3` and `hnsw.bin`.
///
/// Writes are applied to the in-memory index at once and written to `hnsw.bin` on
/// [`flush`](RAGLite::flush), after every `flush_every` changes, and when the value is
/// dropped. Metadata is written to SQLite as each operation finishes. Call `flush()` to
/// see write errors, because `Drop` cannot report them.
pub struct RAGLite {
    index: RwLock<HnswIndex>,
    store: MetadataStore,
    model: EmbeddingModel,
    model_name: String,
    chunker: Chunker,
    config: RagConfig,
    parsers: ParserRegistry,
    db_path: PathBuf,
    /// Index changes made since the last flush.
    unflushed: AtomicUsize,
    #[cfg(feature = "reranking")]
    reranker: Option<Reranker>,
}

impl Drop for RAGLite {
    fn drop(&mut self) {
        // Drop cannot return an error. Call `flush()` directly to see write failures.
        if self.unflushed.load(Ordering::Acquire) > 0 {
            if let Err(e) = self.flush() {
                warn!(error = %e, "failed to write index on drop");
            }
        }
    }
}

impl RAGLite {
    /// Opens (or creates) the database in `path` with default settings.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_config(path, RagConfig::default())
    }

    /// Opens the database in `path` with the embedding model at `model_path`.
    pub fn open_with_model(path: &Path, model_path: &Path) -> Result<Self> {
        let config = RagConfig {
            model_path: Some(model_path.to_path_buf()),
            ..RagConfig::default()
        };
        Self::open_with_config(path, config)
    }

    /// Opens (or creates) the database in `path`.
    ///
    /// The index is rebuilt from the metadata store when the index file is missing, damaged,
    /// or out of step with it, which re-embeds every chunk. Opening fails if the database was
    /// built with a different embedding model, unless `config.reembed_on_model_change` is set.
    pub fn open_with_config(path: &Path, config: RagConfig) -> Result<Self> {
        config.validate()?;
        std::fs::create_dir_all(path)?;

        let (model, model_name) = match config.model_path.clone().or_else(|| find_model(path)) {
            Some(model_path) => {
                if !model_path.exists() {
                    return Err(RagError::Model(format!(
                        "ONNX model not found at {}. Check RAGLITE_MODEL or the model_path setting.",
                        model_path.display()
                    )));
                }
                let name = model_path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (EmbeddingModel::new(&model_path)?, name)
            }
            None => default_model()?,
        };
        let dimensions = model.dimensions;

        let store = MetadataStore::open(path)?;
        if let Some(stored) = store.get_meta(META_SCHEMA_VERSION)? {
            let stored: u32 = stored.parse().unwrap_or(u32::MAX);
            if stored > SCHEMA_VERSION {
                return Err(RagError::SchemaTooNew {
                    stored,
                    supported: SCHEMA_VERSION,
                });
            }
        }
        store.set_meta(META_SCHEMA_VERSION, &SCHEMA_VERSION.to_string())?;
        let model_changed = check_model_identity(&store, &model_name, dimensions)?;
        if model_changed && !config.reembed_on_model_change {
            return Err(RagError::Model(format!(
                "database was built with a different embedding model (stored: {}), not {} ({dimensions} dimensions). Use the original model, or set reembed_on_model_change to re-embed every chunk.",
                store.get_meta(META_MODEL)?.unwrap_or_default(),
                model_name
            )));
        }

        let hnsw_path = path.join(INDEX_FILE);
        let loaded = if hnsw_path.exists() {
            match HnswIndex::load_from_disk(
                &hnsw_path,
                dimensions,
                config.m,
                config.ef_construction,
            ) {
                Ok(index) => Some(index),
                Err(e) => {
                    warn!(error = %e, "index file is unusable, rebuilding from metadata");
                    None
                }
            }
        } else {
            None
        };

        // The index file is trusted only if it is the one the metadata was written against
        // (same epoch), it has seen every vector id the metadata refers to, and the model is
        // unchanged. Anything else is rebuilt from the chunk text.
        let index_is_current = match &loaded {
            Some(index) => {
                let stored_epoch = store.get_meta(META_INDEX_EPOCH)?;
                let same_epoch =
                    stored_epoch.as_deref() == Some(index.epoch().to_string().as_str());
                let behind = store
                    .max_vector_id()?
                    .is_some_and(|max_id| max_id >= index.len());
                same_epoch && !behind && !model_changed
            }
            None => false,
        };

        let index = match loaded {
            Some(index) if index_is_current => index,
            _ if store.count()? == 0 => {
                HnswIndex::new(dimensions, config.m, config.ef_construction)
            }
            _ => {
                info!(
                    chunks = store.count()?,
                    "rebuilding vector index from metadata"
                );
                let (index, mapping) = build_index(&model, &store, dimensions, &config)?;
                // The index file is written before the metadata moves to its ids. A crash
                // in between leaves an epoch mismatch, which forces another rebuild.
                index.flush(&hnsw_path)?;
                store.remap_vector_ids(&mapping, index.epoch())?;
                index
            }
        };

        store.set_meta(META_MODEL, &model_name)?;
        store.set_meta(META_DIMENSIONS, &dimensions.to_string())?;
        if store.get_meta(META_INDEX_EPOCH)?.is_none() {
            // Empty database: no vector ids to keep in step with, so record the epoch of the
            // index that is about to be used.
            store.set_meta(META_INDEX_EPOCH, &index.epoch().to_string())?;
        }

        let chunker = Chunker::new(config.chunk.clone())?;
        #[cfg(feature = "reranking")]
        let reranker = match find_reranker(path, &config) {
            Some(p) => Some(Reranker::new(&p)?),
            None => None,
        };
        info!(
            path = %path.display(),
            model = %model_name,
            vectors = index.len(),
            "database opened"
        );

        Ok(Self {
            index: RwLock::new(index),
            store,
            model,
            model_name,
            chunker,
            config,
            parsers: ParserRegistry::builtin(),
            db_path: path.to_path_buf(),
            unflushed: AtomicUsize::new(0),
            #[cfg(feature = "reranking")]
            reranker,
        })
    }

    /// Adds a parser for its file extensions. It takes precedence over the built-in parser
    /// for the same extension.
    pub fn register_parser(&mut self, parser: Box<dyn Parser>) {
        self.parsers.register(parser);
    }

    /// Writes the index to `hnsw.bin` atomically.
    pub fn flush(&self) -> Result<()> {
        let index = self
            .index
            .read()
            .map_err(|e| RagError::Hnsw(format!("Lock poisoned: {}", e)))?;
        index.flush(&self.db_path.join(INDEX_FILE))?;
        self.unflushed.store(0, Ordering::Release);
        debug!(vectors = index.len(), "index flushed");
        Ok(())
    }

    /// Re-embeds every chunk with the current model and builds a new index from scratch.
    /// Use it after the model file changes. Returns the number of chunks embedded.
    pub fn rebuild_index(&self) -> Result<usize> {
        let mut index = self
            .index
            .write()
            .map_err(|e| RagError::Hnsw(format!("Lock poisoned: {}", e)))?;
        let (rebuilt, mapping) = build_index(
            &self.model,
            &self.store,
            self.model.dimensions,
            &self.config,
        )?;
        rebuilt.flush(&self.db_path.join(INDEX_FILE))?;
        self.store.remap_vector_ids(&mapping, rebuilt.epoch())?;
        self.store.set_meta(META_MODEL, &self.model_name)?;
        let count = rebuilt.len();
        *index = rebuilt;
        self.unflushed.store(0, Ordering::Release);
        info!(chunks = count, "index rebuilt");
        Ok(count)
    }

    pub fn stats(&self) -> Result<Stats> {
        let index_bytes = std::fs::metadata(self.db_path.join(INDEX_FILE))
            .map(|m| m.len())
            .unwrap_or(0);
        Ok(Stats {
            chunks: self.store.count()?,
            vectors: self.vector_count(),
            index_bytes,
            model: self.model_name.clone(),
        })
    }

    pub fn add_document(&self, path: &Path, tags: &[&str]) -> Result<()> {
        let parser = self.parsers.for_path(path).ok_or_else(|| {
            RagError::UnsupportedFormat(
                path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
            )
        })?;

        let text = parser.parse(path)?;
        if text.trim().is_empty() {
            return Err(RagError::EmptyDocument(path.to_string_lossy().to_string()));
        }

        let chunks = self.chunker.chunk(&text);
        let source = path.to_string_lossy().to_string();
        let tag_vec: Vec<String> = tags.iter().map(|s| s.to_string()).collect();
        // Re-adding a file replaces its earlier chunks rather than duplicating them.
        self.ingest(&chunks, &source, &tag_vec, true)
    }

    /// Ingests every supported file under `dir`, recursively, in path order. A file that
    /// fails is recorded in the report and skipped. `on_progress` is called before each file
    /// with its position (starting at 1), the total, and its path.
    pub fn add_directory(
        &self,
        dir: &Path,
        tags: &[&str],
        on_progress: &mut dyn FnMut(usize, usize, &Path),
    ) -> Result<IngestReport> {
        let mut files = Vec::new();
        collect_files(dir, &self.parsers, &mut files)?;
        Ok(self.ingest_files(files, tags, on_progress))
    }

    /// Ingests every supported file matching a glob pattern such as `notes/**/*.md`. Matches
    /// that are not files, or that no parser handles, are skipped. Errors are reported like
    /// [`add_directory`](RAGLite::add_directory).
    pub fn add_glob(
        &self,
        pattern: &str,
        tags: &[&str],
        on_progress: &mut dyn FnMut(usize, usize, &Path),
    ) -> Result<IngestReport> {
        let matches = glob::glob(pattern).map_err(|e| {
            RagError::InvalidConfig(format!("invalid glob pattern {pattern:?}: {e}"))
        })?;
        let files: Vec<PathBuf> = matches
            .filter_map(|m| m.ok())
            .filter(|p| p.is_file() && self.parsers.for_path(p).is_some())
            .collect();
        Ok(self.ingest_files(files, tags, on_progress))
    }

    /// Ingests the files in path order. A file that fails is recorded and skipped.
    fn ingest_files(
        &self,
        mut files: Vec<PathBuf>,
        tags: &[&str],
        on_progress: &mut dyn FnMut(usize, usize, &Path),
    ) -> IngestReport {
        files.sort();
        let total = files.len();
        let mut report = IngestReport::default();
        for (i, file) in files.iter().enumerate() {
            on_progress(i + 1, total, file);
            match self.add_document(file, tags) {
                Ok(()) => report.added.push(file.clone()),
                Err(e) => report.failed.push((file.clone(), e.to_string())),
            }
        }
        report
    }

    pub fn add_text(&self, text: &str, metadata: &Metadata) -> Result<()> {
        if text.trim().is_empty() {
            return Ok(());
        }

        let chunks = self.chunker.chunk(text);
        let source = metadata
            .source
            .clone()
            .unwrap_or_else(|| "inline".to_string());
        self.ingest(&chunks, &source, &metadata.tags, false)
    }

    /// Removes every chunk ingested from `source`. Their vectors are hidden from search
    /// and not returned by queries. Returns the number of chunks removed.
    pub fn delete_document(&self, source: &str) -> Result<usize> {
        let mut index = self
            .index
            .write()
            .map_err(|e| RagError::Hnsw(format!("Lock poisoned: {}", e)))?;
        // Metadata goes first. If that fails, the vectors stay visible and nothing is lost.
        let chunks = self.store.get_chunks_by_source(source)?;
        let deleted = self.store.delete_by_source(source)?;
        for chunk in &chunks {
            if index.remove(chunk.vector_id) {
                self.unflushed.fetch_add(1, Ordering::AcqRel);
            }
        }
        drop(index);
        self.flush_if_due()?;
        Ok(deleted)
    }

    /// Embeds `chunks`, adds their vectors to the index, and writes their metadata in one
    /// SQLite transaction. Embedding happens before anything is written, so a model failure
    /// leaves no trace. With `replace`, the earlier chunks of `source` are removed in the same
    /// transaction. If the metadata write fails, the new vectors are hidden again, so index
    /// and metadata stay consistent.
    fn ingest(&self, chunks: &[Chunk], source: &str, tags: &[String], replace: bool) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = self.model.embed_batch(&chunk_texts)?;

        let mut index = self
            .index
            .write()
            .map_err(|e| RagError::Hnsw(format!("Lock poisoned: {}", e)))?;

        let old_ids: Vec<usize> = if replace {
            self.store
                .get_chunks_by_source(source)?
                .iter()
                .map(|c| c.vector_id)
                .collect()
        } else {
            Vec::new()
        };

        let mut new_ids = Vec::with_capacity(embeddings.len());
        for embedding in embeddings {
            match index.insert(embedding) {
                Ok(id) => new_ids.push(id),
                Err(e) => {
                    hide_all(&mut index, &new_ids);
                    return Err(e);
                }
            }
        }

        let created_at = chrono_now();
        let rows: Vec<ChunkMetadata> = chunks
            .iter()
            .zip(&new_ids)
            .map(|(chunk, &vector_id)| ChunkMetadata {
                vector_id,
                text: chunk.text.clone(),
                source_path: source.to_string(),
                created_at: created_at.clone(),
                tags: tags.to_vec(),
            })
            .collect();

        let written = if replace {
            self.store.replace_source(source, &rows)
        } else {
            self.store.insert_chunks(&rows)
        };
        if let Err(e) = written {
            hide_all(&mut index, &new_ids);
            return Err(e);
        }
        // The new rows are committed, so the old vectors can go.
        hide_all(&mut index, &old_ids);
        self.unflushed
            .fetch_add(new_ids.len() + old_ids.len(), Ordering::AcqRel);
        drop(index);

        debug!(
            source,
            chunks = rows.len(),
            replaced = replace,
            "chunks ingested"
        );
        self.flush_if_due()
    }

    /// Writes the index when enough changes have built up. See `RagConfig::flush_every`.
    fn flush_if_due(&self) -> Result<()> {
        let every = self.config.flush_every;
        if every > 0 && self.unflushed.load(Ordering::Acquire) >= every {
            self.flush()?;
        }
        Ok(())
    }

    /// Returns up to `top_k` chunks by vector similarity. Scores are cosine similarities.
    pub fn query(&self, text: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let query_embedding = self.model.embed(text)?;

        let index = self
            .index
            .read()
            .map_err(|e| RagError::Hnsw(format!("Lock poisoned: {}", e)))?;
        let results = index.search(&query_embedding, top_k, self.config.ef_search)?;
        drop(index);

        self.resolve(results)
    }

    /// Like [`query`](RAGLite::query), but keeps only chunks that match `filter`. The search
    /// looks at the best `10 * top_k` vector matches (at least 100), so a very selective
    /// filter can return fewer than `top_k` results.
    pub fn query_filtered(
        &self,
        text: &str,
        top_k: usize,
        filter: &QueryFilter,
    ) -> Result<Vec<SearchResult>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let candidate_k = top_k.saturating_mul(10).max(100);
        let query_embedding = self.model.embed(text)?;

        let index = self
            .index
            .read()
            .map_err(|e| RagError::Hnsw(format!("Lock poisoned: {}", e)))?;
        let scored = index.search(
            &query_embedding,
            candidate_k,
            self.config.ef_search.max(candidate_k),
        )?;
        drop(index);

        let mut results = self.resolve(scored)?;
        results.retain(|r| filter.matches(r));
        results.truncate(top_k);
        Ok(results)
    }

    /// Checks the database and reports each result. Each check is `ok` or not, with a
    /// detail line. Nothing is changed. The CLI's `doctor` command prints this.
    pub fn check(&self) -> Result<Vec<Check>> {
        let chunks = self.store.count()?;
        let (vectors, removed, epoch) = {
            let index = self
                .index
                .read()
                .map_err(|e| RagError::Hnsw(format!("Lock poisoned: {}", e)))?;
            (index.len(), index.removed_count(), index.epoch())
        };
        let stored_model = self.store.get_meta(META_MODEL)?;
        let stored_epoch = self.store.get_meta(META_INDEX_EPOCH)?;
        let stored_schema = self.store.get_meta(META_SCHEMA_VERSION)?;
        let indexed_chunks = self.store.fts_row_count()?;
        let max_vector_id = self.store.max_vector_id()?;

        Ok(vec![
            Check {
                name: "embedding model",
                ok: true,
                detail: format!(
                    "{} ({} dimensions)",
                    self.model_name, self.model.dimensions
                ),
            },
            Check {
                name: "model identity",
                ok: stored_model.as_deref() == Some(self.model_name.as_str()),
                detail: format!(
                    "database recorded {}",
                    stored_model.unwrap_or_else(|| "nothing".into())
                ),
            },
            Check {
                name: "index epoch",
                ok: stored_epoch.as_deref() == Some(epoch.to_string().as_str()),
                detail: format!("index file epoch {epoch}"),
            },
            Check {
                name: "vector ids",
                ok: max_vector_id.is_none_or(|max| max < vectors),
                detail: format!(
                    "{chunks} chunks, {vectors} vectors, {removed} removed from search"
                ),
            },
            Check {
                name: "full-text index",
                ok: indexed_chunks == chunks,
                detail: format!("{indexed_chunks} indexed rows for {chunks} chunks"),
            },
            Check {
                name: "schema version",
                ok: stored_schema.as_deref() == Some(SCHEMA_VERSION.to_string().as_str()),
                detail: format!(
                    "database {}, library {SCHEMA_VERSION}",
                    stored_schema.unwrap_or_else(|| "unset".into())
                ),
            },
        ])
    }

    /// Reranks the top candidates with the cross-encoder. The candidate pool is
    /// `max(rerank_top_n, top_k)` chunks from vector search, and the best `top_k` are
    /// returned. Fails if no reranker model is installed.
    #[cfg(feature = "reranking")]
    pub fn query_reranked(
        &self,
        text: &str,
        top_k: usize,
        rerank_top_n: usize,
    ) -> Result<Vec<SearchResult>> {
        let reranker = self.reranker.as_ref().ok_or_else(|| {
            RagError::Model(
                "reranker model not found. Set RAGLITE_RERANKER or place ms-marco-MiniLM-L-6-v2.onnx in the models/ directory.".into(),
            )
        })?;
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let pool = self.query(text, rerank_top_n.max(top_k))?;
        if pool.is_empty() {
            return Ok(Vec::new());
        }

        let passages: Vec<String> = pool.iter().map(|r| r.text.clone()).collect();
        let reranked = reranker.rerank(text, &passages, top_k)?;

        Ok(reranked
            .into_iter()
            .map(|(i, score)| SearchResult {
                score,
                ..pool[i].clone()
            })
            .collect())
    }

    /// Combines vector search and BM25 keyword search with reciprocal rank fusion. `alpha`
    /// is the weight of the vector side, from 0 (keywords only) to 1 (vectors only); values
    /// outside that range are clamped. Scores are scaled so the best possible one is 1.
    pub fn query_hybrid(&self, text: &str, top_k: usize, alpha: f32) -> Result<Vec<SearchResult>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let alpha = if alpha.is_nan() {
            0.5
        } else {
            alpha.clamp(0.0, 1.0)
        };
        let candidate_k = top_k.saturating_mul(2);
        let query_embedding = self.model.embed(text)?;

        let index = self
            .index
            .read()
            .map_err(|e| RagError::Hnsw(format!("Lock poisoned: {}", e)))?;
        let vector_results = index.search(&query_embedding, candidate_k, self.config.ef_search)?;
        drop(index);

        let bm25_results = self.store.bm25_search(text, candidate_k)?;

        let mut fused: HashMap<usize, f32> = HashMap::new();
        for (rank, (vector_id, _)) in vector_results.iter().enumerate() {
            *fused.entry(*vector_id).or_insert(0.0) += alpha / (RRF_K + rank as f32 + 1.0);
        }
        for (rank, (vector_id, _)) in bm25_results.iter().enumerate() {
            *fused.entry(*vector_id).or_insert(0.0) += (1.0 - alpha) / (RRF_K + rank as f32 + 1.0);
        }

        // The best possible fused score is 1 / (RRF_K + 1), when a chunk ranks first in both lists.
        let scale = RRF_K + 1.0;
        let mut scored: Vec<(usize, f32)> = fused
            .into_iter()
            .map(|(id, score)| (id, score * scale))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(top_k);

        self.resolve(scored)
    }

    /// Turns `(vector id, score)` pairs into search results, skipping ids with no metadata.
    fn resolve(&self, scored: Vec<(usize, f32)>) -> Result<Vec<SearchResult>> {
        let mut results = Vec::with_capacity(scored.len());
        for (vector_id, score) in scored {
            if let Some((row_id, meta)) = self.store.get_chunk_row_by_vector_id(vector_id)? {
                results.push(SearchResult {
                    score,
                    text: meta.text,
                    source: meta.source_path,
                    tags: meta.tags,
                    chunk_id: row_id.max(0) as u64,
                    created_at: meta.created_at.parse().unwrap_or(0),
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

/// Hides vectors that a failed or replaced write left behind.
fn hide_all(index: &mut HnswIndex, ids: &[usize]) {
    for &id in ids {
        index.remove(id);
    }
}

/// Stores the model identity and reports whether it differs from the one in the database.
/// A database with no stored identity (created before it was recorded) adopts the current one.
fn check_model_identity(
    store: &MetadataStore,
    model_name: &str,
    dimensions: usize,
) -> Result<bool> {
    let stored_model = store.get_meta(META_MODEL)?;
    let stored_dims = store.get_meta(META_DIMENSIONS)?;
    match stored_model {
        None => Ok(false),
        Some(name) => {
            let dims_match = stored_dims.as_deref() == Some(dimensions.to_string().as_str());
            Ok(name != model_name || !dims_match)
        }
    }
}

/// Re-embeds every chunk in the store and builds a fresh index. Also returns, for each chunk
/// row, the vector id it has in the new index. The caller stores that mapping with
/// [`MetadataStore::remap_vector_ids`].
fn build_index(
    model: &EmbeddingModel,
    store: &MetadataStore,
    dimensions: usize,
    config: &RagConfig,
) -> Result<(HnswIndex, Vec<(i64, usize)>)> {
    let rows = store.all_chunk_texts()?;
    let mut index = HnswIndex::new(dimensions, config.m, config.ef_construction);
    let mut mapping = Vec::with_capacity(rows.len());
    for batch in rows.chunks(REBUILD_BATCH) {
        let texts: Vec<&str> = batch.iter().map(|(_, text)| text.as_str()).collect();
        let vectors = model.embed_batch(&texts)?;
        for ((row_id, _), vector) in batch.iter().zip(vectors) {
            mapping.push((*row_id, index.insert(vector)?));
        }
    }
    Ok((index, mapping))
}

/// Recursively lists files that some parser can read.
fn collect_files(dir: &Path, parsers: &ParserRegistry, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, parsers, out)?;
        } else if parsers.for_path(&path).is_some() {
            out.push(path);
        }
    }
    Ok(())
}

/// Finds the embedding model: `RAGLITE_MODEL`, then `<db>/model.onnx`, then
/// `<db>/models/all-MiniLM-L6-v2.onnx`. The current directory is not searched.
/// Finds the model file: `RAGLITE_MODEL` (used even if it does not exist, so the error names
/// it), then `<db>/model.onnx`, then `<db>/models/all-MiniLM-L6-v2.onnx`. `None` means no
/// model file is configured, and the embedded model is used if it was compiled in.
fn find_model(db_path: &Path) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RAGLITE_MODEL") {
        return Some(PathBuf::from(p));
    }
    [
        db_path.join("model.onnx"),
        db_path.join("models").join("all-MiniLM-L6-v2.onnx"),
    ]
    .into_iter()
    .find(|p| p.exists())
}

#[cfg(feature = "embedded-model")]
fn default_model() -> Result<(EmbeddingModel, String)> {
    Ok((
        EmbeddingModel::embedded()?,
        "all-MiniLM-L6-v2 (embedded)".to_string(),
    ))
}

#[cfg(not(feature = "embedded-model"))]
fn default_model() -> Result<(EmbeddingModel, String)> {
    Err(RagError::Model(
        "no embedding model configured. Set RAGLITE_MODEL, or place all-MiniLM-L6-v2.onnx and vocab.txt in <db>/models/, or build with the embedded-model feature.".into(),
    ))
}

/// Finds the optional reranker model, or `None` when no candidate exists.
#[cfg(feature = "reranking")]
fn find_reranker(db_path: &Path, config: &RagConfig) -> Option<PathBuf> {
    if let Some(p) = &config.reranker_path {
        return Some(p.clone());
    }
    if let Some(p) = std::env::var_os("RAGLITE_RERANKER") {
        return Some(PathBuf::from(p));
    }
    [
        db_path.join("reranker.onnx"),
        db_path.join("models").join("ms-marco-MiniLM-L-6-v2.onnx"),
    ]
    .into_iter()
    .find(|p| p.exists())
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
            chunk_id: 1,
            created_at: 0,
        };
        assert_eq!(r.score, 0.95);
        assert_eq!(r.text, "hello");
    }
}
