//! Ingests a folder of documents and answers a question from it.
//!
//! cargo run -p tpt-rag-core --example rust_basic -- ./my_db ./docs "How do I configure it?"
//!
//! The embedding model is found through RAGLITE_MODEL, or in ./my_db/models/.

use std::path::Path;

use tpt_rag_core::{Metadata, RAGLite};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (db, docs, question) = match args.as_slice() {
        [_, db, docs, question] => (db, docs, question),
        _ => {
            eprintln!("usage: rust_basic <db folder> <docs folder> <question>");
            std::process::exit(2);
        }
    };

    let rag = RAGLite::open(Path::new(db))?;

    let report = rag.add_directory(Path::new(docs), &["docs"], &mut |i, total, path| {
        println!("[{i}/{total}] {}", path.display());
    })?;
    println!(
        "ingested {} file(s), {} failed",
        report.added.len(),
        report.failed.len()
    );
    for (path, reason) in &report.failed {
        println!("  skipped {}: {reason}", path.display());
    }

    // A one-off note can go in as plain text, with its own source label.
    rag.add_text(
        "The default port is 8080 unless the PORT variable is set.",
        &Metadata {
            source: Some("notes".into()),
            tags: vec!["ops".into()],
        },
    )?;

    for result in rag.query(question, 5)? {
        println!("[{:.2}] {} ({})", result.score, result.text, result.source);
    }

    // Writes are flushed when `rag` is dropped. Flush explicitly to see any error.
    rag.flush()?;
    Ok(())
}
