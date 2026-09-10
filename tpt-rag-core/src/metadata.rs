use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};

use crate::error::Result;

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

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content=chunks, content_rowid=id);",
        )?;

        Ok(Self { conn, _db_path: db_path })
    }

    pub fn insert_chunk(&self, chunk: &ChunkMetadata) -> Result<()> {
        let tags_json = serde_json::to_string(&chunk.tags)?;
        self.conn.execute(
            "INSERT INTO chunks (vector_id, text, source_path, created_at, tags_json) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![chunk.vector_id, chunk.text, chunk.source_path, chunk.created_at, tags_json],
        )?;
        let rowid: i64 = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?1, ?2)",
            params![rowid, chunk.text],
        )?;
        Ok(())
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
        // Escape FTS5 special characters by wrapping in double quotes
        let escaped_query = format!("\"{}\"", query.replace('"', "\"\""));
        let mut stmt = self.conn.prepare(
            "SELECT c.vector_id, bm25(chunks_fts) as rank \
             FROM chunks_fts fts \
             JOIN chunks c ON c.id = fts.rowid \
             WHERE chunks_fts MATCH ?1 \
             ORDER BY rank \
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![escaped_query, top_k as i64], |row| {
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
        let deleted = self.conn.execute(
            "DELETE FROM chunks WHERE source_path = ?1",
            params![source_path],
        )?;
        self.conn.execute("DELETE FROM chunks_fts", [])?;
        Ok(deleted)
    }

    pub fn count(&self) -> Result<usize> {
        let count: usize = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    pub fn get_all_vector_ids(&self) -> Result<Vec<usize>> {
        let mut stmt = self.conn.prepare("SELECT vector_id FROM chunks ORDER BY vector_id")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row?);
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> MetadataStore {
        let dir = std::env::temp_dir().join("tpt_rag_meta_test");
        std::fs::create_dir_all(&dir).ok();
        MetadataStore::open(&dir).unwrap()
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
