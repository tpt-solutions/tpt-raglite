use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};

use crate::error::Result;

/// `meta` key holding the epoch of the index file that matches the stored vector ids.
pub const META_INDEX_EPOCH: &str = "index_epoch";

#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub vector_id: usize,
    pub text: String,
    pub source_path: String,
    pub created_at: String,
    pub tags: Vec<String>,
}

pub struct MetadataStore {
    conn: Connection,
    _db_path: PathBuf,
}

impl MetadataStore {
    pub fn open(path: &Path) -> Result<Self> {
        let db_path = path.join("metadata.sqlite3");
        let conn = Connection::open(&db_path)?;

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

        Ok(Self {
            conn,
            _db_path: db_path,
        })
    }

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
    pub fn get_chunk_row_by_vector_id(&self, vector_id: usize) -> Result<Option<(i64, ChunkMetadata)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, vector_id, text, source_path, created_at, tags_json FROM chunks WHERE vector_id = ?1",
        )?;
        let mut rows = stmt.query_map(params![vector_id], |row| {
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
                },
            ))
        })?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    pub fn get_chunk_by_vector_id(&self, vector_id: usize) -> Result<Option<ChunkMetadata>> {
        let mut stmt = self.conn.prepare(
            "SELECT vector_id, text, source_path, created_at, tags_json FROM chunks WHERE vector_id = ?1",
        )?;
        let mut rows = stmt.query_map(params![vector_id], |row| {
            let tags_json: String = row.get(4)?;
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            Ok(ChunkMetadata {
                vector_id: row.get(0)?,
                text: row.get(1)?,
                source_path: row.get(2)?,
                created_at: row.get(3)?,
                tags,
            })
        })?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    pub fn get_chunks_by_source(&self, source_path: &str) -> Result<Vec<ChunkMetadata>> {
        let mut stmt = self.conn.prepare(
            "SELECT vector_id, text, source_path, created_at, tags_json FROM chunks WHERE source_path = ?1 ORDER BY vector_id",
        )?;
        let rows = stmt.query_map(params![source_path], |row| {
            let tags_json: String = row.get(4)?;
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            Ok(ChunkMetadata {
                vector_id: row.get(0)?,
                text: row.get(1)?,
                source_path: row.get(2)?,
                created_at: row.get(3)?,
                tags,
            })
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

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

    pub fn delete_by_source(&self, source_path: &str) -> Result<usize> {
        let tx = self.conn.unchecked_transaction()?;
        let deleted = delete_source_rows(&tx, source_path)?;
        tx.commit()?;
        Ok(deleted)
    }

    /// Rows in the full-text index. Matches `count()` when the two stores agree.
    pub fn fts_row_count(&self) -> Result<usize> {
        // FTS5 keeps one `_docsize` row per indexed document. Counting the external content
        // table would always match `chunks`, so the docsize table is the one to count.
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks_fts_docsize", [], |row| row.get(0))?;
        Ok(count)
    }

    pub fn count(&self) -> Result<usize> {
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        Ok(count)
    }

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
fn insert_row(conn: &Connection, chunk: &ChunkMetadata) -> Result<()> {
    let tags_json = serde_json::to_string(&chunk.tags)?;
    conn.execute(
        "INSERT INTO chunks (vector_id, text, source_path, created_at, tags_json) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![chunk.vector_id, chunk.text, chunk.source_path, chunk.created_at, tags_json],
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
            })
            .unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }
}
