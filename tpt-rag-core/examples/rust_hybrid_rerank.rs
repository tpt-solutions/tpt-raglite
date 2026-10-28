//! Compares three ways to rank the same question: vector search, hybrid search, and vector
//! search followed by cross-encoder reranking.
//!
//! cargo run -p tpt-rag-core --example rust_hybrid_rerank -- ./my_db "your question"
//!
//! Needs the embedding model and the reranker. Set RAGLITE_MODEL and RAGLITE_RERANKER, or
//! place them in ./my_db/models/ (see scripts/fetch_model.sh).

use std::path::Path;

use tpt_rag_core::{Metadata, RAGLite, RagConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, db, question] = args.as_slice() else {
        eprintln!("usage: rust_hybrid_rerank <db folder> <question>");
        std::process::exit(2);
    };

    let rag = RAGLite::open_with_config(Path::new(db), RagConfig::default())?;

    // Some sample passages. A real database would already contain these.
    for (source, text) in [
        (
            "space",
            "Jupiter is the largest planet in the solar system.",
        ),
        (
            "garden",
            "Tomato plants need full sun and regular watering.",
        ),
        (
            "bread",
            "Sourdough needs a long overnight rise before baking.",
        ),
    ] {
        rag.add_text(
            text,
            &Metadata {
                source: Some(source.into()),
                tags: Vec::new(),
            },
        )?;
    }

    println!("vector search:");
    for r in rag.query(question, 3)? {
        println!("  [{:.3}] {}", r.score, r.source);
    }

    // alpha = 0.5 weights vector and keyword rankings equally.
    println!("hybrid search (alpha 0.5):");
    for r in rag.query_hybrid(question, 3, 0.5)? {
        println!("  [{:.3}] {}", r.score, r.source);
    }

    // Reranking needs the cross-encoder. Without it this returns an error.
    println!("reranked (top 2 of 3 candidates):");
    match rag.query_reranked(question, 2, 3) {
        Ok(results) => {
            for r in results {
                println!("  [{:.3}] {}", r.score, r.source);
            }
        }
        Err(e) => println!("  unavailable: {e}"),
    }
    Ok(())
}
