use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};

use crate::error::{RagError, Result};

/// `meta` key holding the epoch of the index file that matches the stored vector ids.
pub const META_INDEX_EPOCH: &str = "index_epoch";

#[derive(Debug, Clone)]
/// The stored record for one chunk.
pub struct ChunkMetadata {
    /// Id of the chunk's vector in the HNSW index.
    pub vector_id: usize,
    /// The chunk text.
    pub text: String,
    /// Path of the document the chunk came from.
    pub source_path: String,
    /// Ingestion time in Unix seconds, stored as text.
    pub created_at: String,
    /// Tags attached when the chunk was ingested.
    pub tags: Vec<String>,
    /// Byte offset where the chunk starts in the text that was chunked. `None` for chunks
    /// stored before schema version 2.
    pub start_offset: Option<u64>,
    /// Byte offset just past the end of the chunk. `None` for chunks stored before schema
    /// version 2.
    pub end_offset: Option<u64>,
    /// Heading path of the section the chunk starts in, such as `Setup > Backups`. `None`
    /// when the document has no headings, or for chunks stored before schema version 3.
    pub section: Option<String>,
}

/// SQLite store for chunk text, source paths, tags and the full-text (BM25) index.
pub struct MetadataStore {
    conn: Connection,
    _db_path: PathBuf,
}

impl MetadataStore {
    /// Opens or creates the SQLite file at `path` and makes sure the schema exists.
    pub fn open(path: &Path) -> Result<Self> {
        let db_path = path.join("metadata.sqlite3");
        let conn = Connection::open(&db_path)?;

        run_migrations(&conn)?;

        Ok(Self {
            conn,
            _db_path: db_path,
        })
    }

    /// Stores one chunk record.
    pub fn insert_chunk(&self, chunk: &ChunkMetadata) -> Result<()> {
        insert_row(&self.conn, chunk)
    }

    /// Inserts all `chunks` in one transaction. Either every row is written or none is.
    pub fn insert_chunks(&self, chunks: &[ChunkMetadata]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        for chunk in chunks {
            insert_row(&tx, chunk)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Reads a value from the `meta` table, such as the embedding model id.
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let value = self.conn.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![key],
            |row| row.get(0),
        );
        match value {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Sets a key in the `meta` table, such as the model id.
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Replaces every chunk of `source_path` with `chunks` in one transaction. Either the
    /// old rows are gone and the new rows are in, or nothing has changed.
    pub fn replace_source(&self, source_path: &str, chunks: &[ChunkMetadata]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        delete_source_rows(&tx, source_path)?;
        for chunk in chunks {
            insert_row(&tx, chunk)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// `(row id, text)` for every chunk, in row order. Used to rebuild the index.
    pub fn all_chunk_texts(&self) -> Result<Vec<(i64, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, text FROM chunks ORDER BY id")?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Sets new vector ids on existing rows, given as `(row id, vector id)`, and records the
    /// epoch of the index they now refer to. Both happen in one transaction, so the metadata
    /// never points at ids from a different index. The ids are first moved to negative values
    /// so that no step breaks the UNIQUE constraint.
    pub fn remap_vector_ids(&self, mapping: &[(i64, usize)], index_epoch: u64) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("UPDATE chunks SET vector_id = -1 - id", [])?;
        for &(row_id, vector_id) in mapping {
            tx.execute(
                "UPDATE chunks SET vector_id = ?2 WHERE id = ?1",
                params![row_id, vector_id],
            )?;
        }
        tx.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![META_INDEX_EPOCH, index_epoch.to_string()],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// The largest vector id referenced by any chunk, or `None` for an empty store.
    pub fn max_vector_id(&self) -> Result<Option<usize>> {
        let max: Option<i64> =
            self.conn
                .query_row("SELECT MAX(vector_id) FROM chunks", [], |row| row.get(0))?;
        Ok(max.and_then(|m| usize::try_from(m).ok()))
    }

    /// Like `get_chunk_by_vector_id`, also returning the row id, which results cite.
    pub fn get_chunk_row_by_vector_id(
        &self,
        vector_id: usize,
    ) -> Result<Option<(i64, ChunkMetadata)>> {
        let sql = format!("SELECT {CHUNK_COLUMNS} FROM chunks WHERE vector_id = ?1");
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![vector_id], read_chunk_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Returns the chunk stored for a vector id, if there is one.
    pub fn get_chunk_by_vector_id(&self, vector_id: usize) -> Result<Option<ChunkMetadata>> {
        Ok(self.get_chunk_row_by_vector_id(vector_id)?.map(|(_, c)| c))
    }

    /// Every chunk with its row id, in the order they were stored.
    pub fn all_chunks(&self) -> Result<Vec<(i64, ChunkMetadata)>> {
        let sql = format!("SELECT {CHUNK_COLUMNS} FROM chunks ORDER BY id");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], read_chunk_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Returns every chunk ingested from `source_path`.
    pub fn get_chunks_by_source(&self, source_path: &str) -> Result<Vec<ChunkMetadata>> {
        let sql =
            format!("SELECT {CHUNK_COLUMNS} FROM chunks WHERE source_path = ?1 ORDER BY vector_id");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![source_path], read_chunk_row)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?.1);
        }
        Ok(result)
    }

    /// Full-text search. Returns up to `top_k` matching chunks as (vector id, BM25 score) pairs.
    pub fn bm25_search(&self, query: &str, top_k: usize) -> Result<Vec<(usize, f32)>> {
        // Each term is quoted so FTS5 treats it literally, and terms are OR-ed so a chunk
        // matching any query word is a candidate. BM25 ranks chunks with more matches higher.
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let match_expr = terms.join(" OR ");

        let mut stmt = self.conn.prepare(
            "SELECT c.vector_id, bm25(chunks_fts) as rank \
             FROM chunks_fts fts \
             JOIN chunks c ON c.id = fts.rowid \
             WHERE chunks_fts MATCH ?1 \
             ORDER BY rank \
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![match_expr, top_k as i64], |row| {
            let vector_id: usize = row.get(0)?;
            let rank: f32 = row.get(1)?;
            Ok((vector_id, rank))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Removes every chunk from `source_path` and returns how many were removed.
    pub fn delete_by_source(&self, source_path: &str) -> Result<usize> {
        let tx = self.conn.unchecked_transaction()?;
        let deleted = delete_source_rows(&tx, source_path)?;
        // The file is no longer indexed, so a later sync must not treat it as up to date.
        tx.execute(
            "DELETE FROM documents WHERE source_path = ?1",
            params![source_path],
        )?;
        tx.commit()?;
        Ok(deleted)
    }

    /// The content hash that `sync_directory` stored for a file, if it has one.
    pub fn document_hash(&self, source_path: &str) -> Result<Option<String>> {
        match self.conn.query_row(
            "SELECT content_hash FROM documents WHERE source_path = ?1",
            params![source_path],
            |row| row.get(0),
        ) {
            Ok(hash) => Ok(Some(hash)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Records the content hash of a file that was just ingested.
    pub fn set_document_hash(&self, source_path: &str, hash: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO documents (source_path, content_hash) VALUES (?1, ?2)
             ON CONFLICT(source_path) DO UPDATE SET content_hash = excluded.content_hash",
            params![source_path, hash],
        )?;
        Ok(())
    }

    /// Forgets the content hash of a file, so the next sync ingests it again.
    pub fn forget_document_hash(&self, source_path: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM documents WHERE source_path = ?1",
            params![source_path],
        )?;
        Ok(())
    }

    /// Every source that `sync_directory` tracks.
    pub fn document_sources(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT source_path FROM documents ORDER BY source_path")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Rows in the full-text index. Matches `count()` when the two stores agree.
    pub fn fts_row_count(&self) -> Result<usize> {
        // FTS5 keeps one `_docsize` row per indexed document. Counting the external content
        // table would always match `chunks`, so the docsize table is the one to count.
        let count: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM chunks_fts_docsize", [], |row| {
                    row.get(0)
                })?;
        Ok(count)
    }

    /// Number of chunks stored.
    pub fn count(&self) -> Result<usize> {
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        Ok(count)
    }

    /// The chunks just before and after row `row_id` in the same document, at most `radius`
    /// on each side, in document order. The chunk itself is not included. Chunks without
    /// offsets have no order, so a chunk without offsets gets no neighbours.
    pub fn get_neighbors(&self, row_id: i64, radius: usize) -> Result<Vec<(i64, ChunkMetadata)>> {
        let Some((_, target)) = self.chunk_row_by_id(row_id)? else {
            return Ok(Vec::new());
        };
        let Some(start) = target.start_offset else {
            return Ok(Vec::new());
        };
        let before = self.neighbor_rows(&target.source_path, start, row_id, radius, false)?;
        let after = self.neighbor_rows(&target.source_path, start, row_id, radius, true)?;
        let mut out: Vec<(i64, ChunkMetadata)> = before.into_iter().rev().collect();
        out.extend(after);
        Ok(out)
    }

    /// The chunk stored under row id `row_id`, if there is one.
    fn chunk_row_by_id(&self, row_id: i64) -> Result<Option<(i64, ChunkMetadata)>> {
        let sql = format!("SELECT {CHUNK_COLUMNS} FROM chunks WHERE id = ?1");
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![row_id], read_chunk_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Up to `radius` chunks of `source` that start before (or after) `start`, nearest first.
    /// Ties on the start offset are broken by row id.
    fn neighbor_rows(
        &self,
        source: &str,
        start: u64,
        row_id: i64,
        radius: usize,
        following: bool,
    ) -> Result<Vec<(i64, ChunkMetadata)>> {
        let (cmp, order) = if following {
            (">", "ASC")
        } else {
            ("<", "DESC")
        };
        let sql = format!(
            "SELECT {CHUNK_COLUMNS}
             FROM chunks
             WHERE source_path = ?1 AND start_offset IS NOT NULL
               AND (start_offset {cmp} ?2 OR (start_offset = ?2 AND id {cmp} ?3))
             ORDER BY start_offset {order}, id {order}
             LIMIT ?4"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(
            params![source, start as i64, row_id, radius as i64],
            read_chunk_row,
        )?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Vector ids of every stored chunk.
    pub fn get_all_vector_ids(&self) -> Result<Vec<usize>> {
        let mut stmt = self
            .conn
            .prepare("SELECT vector_id FROM chunks ORDER BY vector_id")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row?);
        }
        Ok(ids)
    }
}

/// Removes the rows and FTS entries of `source_path`. Callers decide the transaction boundary.
fn delete_source_rows(conn: &Connection, source_path: &str) -> Result<usize> {
    // FTS5 external-content tables must be told the exact text that was indexed for
    // each row, so the FTS entries are removed before the source rows are deleted.
    conn.execute(
        "INSERT INTO chunks_fts(chunks_fts, rowid, text)
         SELECT 'delete', id, text FROM chunks WHERE source_path = ?1",
        params![source_path],
    )?;
    Ok(conn.execute(
        "DELETE FROM chunks WHERE source_path = ?1",
        params![source_path],
    )?)
}

/// Writes one chunk row and its FTS entry. Callers decide the transaction boundary.
/// The columns `read_chunk_row` reads, in the order it reads them.
const CHUNK_COLUMNS: &str =
    "id, vector_id, text, source_path, created_at, tags_json, start_offset, end_offset, section";

/// Reads one row selected with `CHUNK_COLUMNS`. Returns the row id with the chunk.
fn read_chunk_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(i64, ChunkMetadata)> {
    let tags_json: String = row.get(5)?;
    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
    Ok((
        row.get(0)?,
        ChunkMetadata {
            vector_id: row.get(1)?,
            text: row.get(2)?,
            source_path: row.get(3)?,
            created_at: row.get(4)?,
            tags,
            start_offset: row_offset(row, 6)?,
            end_offset: row_offset(row, 7)?,
            section: row.get(8)?,
        },
    ))
}

/// Reads a nullable offset column. Offsets are stored as signed integers because SQLite has
/// no unsigned type.
fn row_offset(row: &rusqlite::Row<'_>, index: usize) -> rusqlite::Result<Option<u64>> {
    Ok(row.get::<_, Option<i64>>(index)?.map(|v| v as u64))
}

/// `meta` key holding the schema version the database was last migrated to.
pub const META_SCHEMA_VERSION: &str = "schema_version";

/// The schema version this library writes: the version of the last migration.
pub const LATEST_SCHEMA_VERSION: u32 = 4;

/// One step that moves a database to `version`. Steps run in order. Each runs in a transaction
/// together with its version bump, so a step that fails leaves the database as it was.
struct Migration {
    version: u32,
    apply: fn(&Connection) -> Result<()>,
}

const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        apply: create_schema,
    },
    Migration {
        version: 2,
        apply: add_chunk_offsets,
    },
    Migration {
        version: 3,
        apply: add_chunk_section,
    },
    Migration {
        version: 4,
        apply: add_documents,
    },
];

/// Brings the database up to `LATEST_SCHEMA_VERSION`. A database written by a newer library is
/// refused before anything is changed.
fn run_migrations(conn: &Connection) -> Result<()> {
    // The version record lives in `meta`, so the table exists before any migration runs. Older
    // databases may not have it yet.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )?;
    let current = current_version(conn)?;
    if current > LATEST_SCHEMA_VERSION {
        return Err(RagError::SchemaTooNew {
            stored: current,
            supported: LATEST_SCHEMA_VERSION,
        });
    }
    for migration in MIGRATIONS.iter().filter(|m| m.version > current) {
        let tx = conn.unchecked_transaction()?;
        (migration.apply)(&tx)?;
        tx.execute(
            "INSERT INTO meta (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![META_SCHEMA_VERSION, migration.version.to_string()],
        )?;
        tx.commit()?;
    }
    Ok(())
}

/// The version the database is at. A database with no version record is at version 1 when it
/// has a chunks table, which is how databases from before versions were recorded look. An
/// empty database is at version 0.
fn current_version(conn: &Connection) -> Result<u32> {
    if table_exists(conn, "meta")? {
        let stored = match conn.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![META_SCHEMA_VERSION],
            |row| row.get::<_, String>(0),
        ) {
            Ok(value) => Some(value),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => return Err(e.into()),
        };
        if let Some(value) = stored {
            return Ok(value.parse().unwrap_or(u32::MAX));
        }
    }
    Ok(if table_exists(conn, "chunks")? { 1 } else { 0 })
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
        params![table],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

/// Version 1: chunks, their full-text index, and the meta table.
fn create_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector_id INTEGER UNIQUE NOT NULL,
            text TEXT NOT NULL,
            source_path TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            tags_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON chunks(vector_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_path);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content=chunks, content_rowid=id);
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )?;
    Ok(())
}

/// Version 2: byte offsets of each chunk. Existing chunks have none.
fn add_chunk_offsets(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "ALTER TABLE chunks ADD COLUMN start_offset INTEGER;
        ALTER TABLE chunks ADD COLUMN end_offset INTEGER;",
    )?;
    Ok(())
}

/// Version 3: the heading path or page each chunk starts in. Existing chunks have none.
fn add_chunk_section(conn: &Connection) -> Result<()> {
    conn.execute_batch("ALTER TABLE chunks ADD COLUMN section TEXT;")?;
    Ok(())
}

/// Version 4: content hashes of the files that `sync_directory` ingested.
fn add_documents(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS documents (
            source_path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL
        );",
    )?;
    Ok(())
}

fn insert_row(conn: &Connection, chunk: &ChunkMetadata) -> Result<()> {
    let tags_json = serde_json::to_string(&chunk.tags)?;
    conn.execute(
        "INSERT INTO chunks (vector_id, text, source_path, created_at, tags_json, start_offset, end_offset, section) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            chunk.vector_id,
            chunk.text,
            chunk.source_path,
            chunk.created_at,
            tags_json,
            chunk.start_offset.map(|v| v as i64),
            chunk.end_offset.map(|v| v as i64),
            chunk.section,
        ],
    )?;
    let rowid: i64 = conn.last_insert_rowid();
    conn.execute(
        "INSERT INTO chunks_fts(rowid, text) VALUES (?1, ?2)",
        params![rowid, chunk.text],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each call gets its own directory, so tests never share a database file.
    fn temp_store() -> MetadataStore {
        let dir = std::env::temp_dir().join(format!("tpt_rag_meta_test_{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        MetadataStore::open(&dir).unwrap()
    }

    fn chunk(vector_id: usize, text: &str, source: &str) -> ChunkMetadata {
        ChunkMetadata {
            vector_id,
            text: text.to_string(),
            source_path: source.to_string(),
            created_at: "2026-01-01".to_string(),
            tags: vec![],
            start_offset: None,
            end_offset: None,
            section: None,
        }
    }

    #[test]
    fn test_delete_keeps_other_sources_searchable() {
        let store = temp_store();
        store
            .insert_chunk(&chunk(0, "alpha lighthouse", "a.txt"))
            .unwrap();
        store
            .insert_chunk(&chunk(1, "beta harbour", "b.txt"))
            .unwrap();

        store.delete_by_source("a.txt").unwrap();

        assert!(store.bm25_search("alpha", 10).unwrap().is_empty());
        let hits = store.bm25_search("beta", 10).unwrap();
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn test_insert_chunks_rolls_back_on_failure() {
        let store = temp_store();
        // The second row reuses vector_id 0, which violates the UNIQUE constraint.
        let batch = vec![chunk(0, "first", "a.txt"), chunk(0, "second", "a.txt")];
        assert!(store.insert_chunks(&batch).is_err());
        assert_eq!(store.count().unwrap(), 0);
        assert!(store.bm25_search("first", 10).unwrap().is_empty());
    }

    #[test]
    fn test_max_vector_id() {
        let store = temp_store();
        assert_eq!(store.max_vector_id().unwrap(), None);
        store
            .insert_chunks(&[chunk(3, "x", "a.txt"), chunk(7, "y", "a.txt")])
            .unwrap();
        assert_eq!(store.max_vector_id().unwrap(), Some(7));
    }

    #[test]
    fn test_bm25_matches_any_term() {
        let store = temp_store();
        store
            .insert_chunk(&chunk(0, "alpha only", "a.txt"))
            .unwrap();
        store.insert_chunk(&chunk(1, "beta only", "b.txt")).unwrap();

        let mut ids: Vec<usize> = store
            .bm25_search("alpha beta", 10)
            .unwrap()
            .iter()
            .map(|h| h.0)
            .collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1]);
        assert!(store.bm25_search("   ", 10).unwrap().is_empty());
    }

    #[test]
    fn test_insert_and_get() {
        let store = temp_store();
        let chunk = ChunkMetadata {
            vector_id: 0,
            text: "Hello world".to_string(),
            source_path: "test.txt".to_string(),
            created_at: "2026-01-01 00:00:00".to_string(),
            tags: vec!["test".to_string()],
            start_offset: None,
            end_offset: None,
            section: None,
        };
        store.insert_chunk(&chunk).unwrap();

        let retrieved = store.get_chunk_by_vector_id(0).unwrap().unwrap();
        assert_eq!(retrieved.text, "Hello world");
        assert_eq!(retrieved.tags, vec!["test"]);
    }

    #[test]
    fn test_delete_by_source() {
        let store = temp_store();
        for i in 0..3 {
            store
                .insert_chunk(&ChunkMetadata {
                    vector_id: i,
                    text: format!("text {i}"),
                    source_path: "file.txt".to_string(),
                    created_at: "2026-01-01".to_string(),
                    tags: vec![],
                    start_offset: None,
                    end_offset: None,
                    section: None,
                })
                .unwrap();
        }
        let deleted = store.delete_by_source("file.txt").unwrap();
        assert_eq!(deleted, 3);
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_count() {
        let store = temp_store();
        assert_eq!(store.count().unwrap(), 0);
        store
            .insert_chunk(&ChunkMetadata {
                vector_id: 0,
                text: "a".to_string(),
                source_path: "x".to_string(),
                created_at: "now".to_string(),
                tags: vec![],
                start_offset: None,
                end_offset: None,
                section: None,
            })
            .unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn databases_from_schema_1_gain_offset_columns() {
        let dir = std::env::temp_dir().join(format!("tpt_rag_migrate_{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        {
            // The schema 1 layout, from before the offset columns existed.
            let conn = Connection::open(dir.join("metadata.sqlite3")).unwrap();
            conn.execute_batch(
                "CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_id INTEGER UNIQUE NOT NULL,
                    text TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    tags_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=id);
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO meta (key, value) VALUES ('schema_version', '1');
                INSERT INTO chunks (vector_id, text, source_path) VALUES (0, 'old text', 'old.txt');",
            )
            .unwrap();
        }

        let store = MetadataStore::open(&dir).unwrap();
        let old = store.get_chunk_by_vector_id(0).unwrap().unwrap();
        assert_eq!(old.text, "old text");
        assert_eq!((old.start_offset, old.end_offset), (None, None));
        assert_eq!(old.section, None);
        assert_eq!(
            store.get_meta(META_SCHEMA_VERSION).unwrap().as_deref(),
            Some("4")
        );

        store
            .insert_chunk(&ChunkMetadata {
                vector_id: 1,
                text: "new".to_string(),
                source_path: "new.txt".to_string(),
                created_at: "now".to_string(),
                tags: vec![],
                start_offset: Some(3),
                end_offset: Some(6),
                section: None,
            })
            .unwrap();
        let new = store.get_chunk_by_vector_id(1).unwrap().unwrap();
        assert_eq!((new.start_offset, new.end_offset), (Some(3), Some(6)));
        drop(store);

        // Opening an upgraded database again must not add the columns a second time.
        MetadataStore::open(&dir).unwrap();
    }

    #[test]
    fn neighbors_are_the_chunks_around_a_row_in_document_order() {
        let store = temp_store();
        // Inserted out of document order, so the ordering has to come from the offsets.
        for (vector_id, start) in [(0usize, 20u64), (1, 0), (2, 40), (3, 10), (4, 30)] {
            store
                .insert_chunk(&ChunkMetadata {
                    vector_id,
                    text: format!("chunk at {start}"),
                    source_path: "doc.txt".to_string(),
                    created_at: "now".to_string(),
                    tags: vec![],
                    start_offset: Some(start),
                    end_offset: Some(start + 10),
                    section: None,
                })
                .unwrap();
        }
        // A chunk stored without offsets has no place in the document.
        store.insert_chunk(&chunk(5, "legacy", "doc.txt")).unwrap();

        let row = |vector_id| {
            store
                .get_chunk_row_by_vector_id(vector_id)
                .unwrap()
                .unwrap()
                .0
        };
        let starts = |rows: Vec<(i64, ChunkMetadata)>| {
            rows.into_iter()
                .map(|(_, c)| c.start_offset)
                .collect::<Vec<_>>()
        };
        // Document order is 0, 10, 20, 30, 40. The chunk at 20 is vector 0.
        assert_eq!(
            starts(store.get_neighbors(row(0), 1).unwrap()),
            vec![Some(10), Some(30)]
        );
        assert_eq!(
            starts(store.get_neighbors(row(0), 2).unwrap()),
            vec![Some(0), Some(10), Some(30), Some(40)]
        );
        // The first and last chunks have a neighbour on one side only.
        assert_eq!(
            starts(store.get_neighbors(row(1), 1).unwrap()),
            vec![Some(10)]
        );
        assert_eq!(
            starts(store.get_neighbors(row(2), 1).unwrap()),
            vec![Some(30)]
        );
        assert!(store.get_neighbors(row(5), 1).unwrap().is_empty());
        assert!(store.get_neighbors(9999, 1).unwrap().is_empty());
    }

    #[test]
    fn migrations_run_in_order_and_end_at_the_latest_version() {
        let versions: Vec<u32> = MIGRATIONS.iter().map(|m| m.version).collect();
        assert_eq!(versions, (1..=LATEST_SCHEMA_VERSION).collect::<Vec<u32>>());
    }

    #[test]
    fn a_new_database_is_created_at_the_latest_version() {
        let dir = std::env::temp_dir().join(format!("tpt_rag_fresh_{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        let store = MetadataStore::open(&dir).unwrap();
        assert_eq!(
            store.get_meta(META_SCHEMA_VERSION).unwrap(),
            Some(LATEST_SCHEMA_VERSION.to_string())
        );
    }

    #[test]
    fn a_database_from_a_newer_library_is_refused_and_left_alone() {
        let dir = std::env::temp_dir().join(format!("tpt_rag_newer_{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        {
            let store = MetadataStore::open(&dir).unwrap();
            store.set_meta(META_SCHEMA_VERSION, "99").unwrap();
        }
        let err = MetadataStore::open(&dir).err().expect("refused");
        assert!(matches!(
            err,
            RagError::SchemaTooNew {
                stored: 99,
                supported: LATEST_SCHEMA_VERSION
            }
        ));
        let conn = Connection::open(dir.join("metadata.sqlite3")).unwrap();
        let version: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![META_SCHEMA_VERSION],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, "99");
    }

    #[test]
    fn a_failing_migration_leaves_the_database_at_the_previous_version() {
        let dir = std::env::temp_dir().join(format!("tpt_rag_failed_{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        {
            // Version 2 on paper, but the section column is already there, so migration 3 fails.
            let conn = Connection::open(dir.join("metadata.sqlite3")).unwrap();
            conn.execute_batch(
                "CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_id INTEGER UNIQUE NOT NULL,
                    text TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    start_offset INTEGER,
                    end_offset INTEGER,
                    section TEXT
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=id);
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO meta (key, value) VALUES ('schema_version', '2');",
            )
            .unwrap();
        }

        assert!(MetadataStore::open(&dir).is_err());

        let conn = Connection::open(dir.join("metadata.sqlite3")).unwrap();
        let version: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![META_SCHEMA_VERSION],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, "2");
        // Migration 4 must not have run after the failure.
        assert!(!table_exists(&conn, "documents").unwrap());
    }
}
