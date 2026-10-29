//! Times the search methods on a real database: vector, hybrid and reranked search, with the
//! real embedding model and cross-encoder. The corpus is the README (a few dozen chunks), so
//! the numbers show the cost of each step, not behavior at scale.
//!
//! cargo run --release -p tpt-rag-core --example bench_search -- <README.md> [queries]
//!
//! Needs the models in models/ (run `raglite models pull` or scripts/fetch_model first).

use std::path::{Path, PathBuf};
use std::time::Instant;

use tpt_rag_core::RAGLite;

const QUESTIONS: &[&str] = &[
    "how do I install from source",
    "what is the license",
    "how do I use it from Python",
    "which score ranges do the search methods return",
    "how are results cited",
    "how do I keep a folder in sync",
];

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[index]
}

fn time<F: FnMut() -> usize>(name: &str, rounds: usize, mut search: F) {
    let mut millis = Vec::with_capacity(rounds);
    let mut hits = 0;
    for _ in 0..rounds {
        let start = Instant::now();
        hits = search();
        millis.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    millis.sort_by(|a, b| a.total_cmp(b));
    println!(
        "{name:<10} p50 {:>8.2} ms   p95 {:>8.2} ms   ({} results)",
        percentile(&millis, 0.5),
        percentile(&millis, 0.95),
        hits
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let document = PathBuf::from(args.first().cloned().unwrap_or_else(|| "README.md".into()));
    let rounds: usize = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(20);

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
    let model = root.join("all-MiniLM-L6-v2.onnx");
    let reranker = root.join("ms-marco-MiniLM-L-6-v2.onnx");
    if !model.exists() || !reranker.exists() {
        eprintln!(
            "models not found in {}; run raglite models pull first",
            root.display()
        );
        std::process::exit(2);
    }

    let dir = std::env::temp_dir().join(format!("tpt_bench_search_{}", rand::random::<u32>()));
    let rag = RAGLite::open_with_config(
        &dir,
        tpt_rag_core::RagConfig {
            model_path: Some(model),
            reranker_path: Some(reranker),
            ..Default::default()
        },
    )
    .expect("open database");
    rag.add_document(&document, &[]).expect("ingest document");
    let chunks = rag.chunk_count().expect("count chunks");
    println!(
        "corpus: {} ({} chunks), {} rounds over {} questions\n",
        document.display(),
        chunks,
        rounds,
        QUESTIONS.len()
    );

    // Warm up the models so the first call does not pay for loading them.
    for question in QUESTIONS {
        let _ = rag.query(question, 5);
        let _ = rag.query_reranked(question, 5, 20);
    }

    // Each timed loop walks the questions in turn, with its own counter.
    let mut vector_round = 0usize;
    time("vector", rounds * QUESTIONS.len(), || {
        vector_round += 1;
        rag.query(QUESTIONS[vector_round % QUESTIONS.len()], 5)
            .unwrap()
            .len()
    });
    let mut hybrid_round = 0usize;
    time("hybrid", rounds * QUESTIONS.len(), || {
        hybrid_round += 1;
        rag.query_hybrid(QUESTIONS[hybrid_round % QUESTIONS.len()], 5, 0.5)
            .unwrap()
            .len()
    });
    let mut rerank_round = 0usize;
    time("reranked", rounds * QUESTIONS.len(), || {
        rerank_round += 1;
        rag.query_reranked(QUESTIONS[rerank_round % QUESTIONS.len()], 5, 20)
            .unwrap()
            .len()
    });
    drop(rag);
    let _ = std::fs::remove_dir_all(&dir);
}
