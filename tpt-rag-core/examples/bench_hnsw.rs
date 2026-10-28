//! Measures the HNSW index on its own: insert throughput, query latency, recall@10 against
//! brute force, and flush and load time. Vectors are random unit vectors of the same size as
//! the default model (384), so the numbers describe the index, not the embedding model.
//!
//! cargo run --release -p tpt-rag-core --example bench_hnsw -- [vectors] [queries] [ef_search]
//!
//! Defaults: 100000 vectors, 1000 queries, ef_search 100. Build in release mode; debug
//! numbers are meaningless.

use std::path::Path;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tpt_rag_core::embedding::cosine_similarity;
use tpt_rag_core::hnsw::HnswIndex;

const DIM: usize = 384;
const TOP_K: usize = 10;
const RECALL_QUERIES: usize = 100;

fn random_unit_vector(rng: &mut StdRng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Vectors around random cluster centers, like sentence embeddings are. Points in one topic
/// sit close together, so the nearest neighbors are well separated.
fn clustered_unit_vectors(rng: &mut StdRng, count: usize) -> Vec<Vec<f32>> {
    // Noise has a norm of about 0.5 against a unit-length center. Each coordinate is uniform
    // in [-a, a] with a = 0.5 * sqrt(3 / DIM), since the squared norm averages a^2 * DIM / 3.
    let amplitude = 0.5 * (3.0 / DIM as f32).sqrt();
    let centers: Vec<Vec<f32>> = (0..1000).map(|_| random_unit_vector(rng)).collect();
    (0..count)
        .map(|_| {
            let center = &centers[rng.gen_range(0..centers.len())];
            let mut v: Vec<f32> = center
                .iter()
                .map(|c| c + amplitude * rng.gen_range(-1.0f32..1.0))
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

fn percentile(sorted_us: &[f64], p: f64) -> f64 {
    let idx = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[idx]
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("arguments are whole numbers"))
        .collect();
    let vectors = args.first().copied().unwrap_or(100_000);
    let queries = args.get(1).copied().unwrap_or(1000);
    let ef_search = args.get(2).copied().unwrap_or(100);
    // Third positional argument selects the data: 0 = uniform random, 1 = clustered.
    let clustered = args.get(3).copied().unwrap_or(1) == 1;

    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f32>> = if clustered {
        clustered_unit_vectors(&mut rng, vectors)
    } else {
        (0..vectors).map(|_| random_unit_vector(&mut rng)).collect()
    };
    println!(
        "data: {} vectors, {}",
        vectors,
        if clustered {
            "clustered"
        } else {
            "uniform random"
        }
    );

    let start = Instant::now();
    let mut index = HnswIndex::new(DIM, 16, 200);
    for v in &data {
        index.insert(v.clone()).expect("insert");
    }
    let build = start.elapsed().as_secs_f64();
    println!(
        "insert: {vectors} vectors in {build:.1}s ({:.0} vectors/s)",
        vectors as f64 / build
    );

    let query_ids: Vec<usize> = (0..queries).map(|_| rng.gen_range(0..vectors)).collect();
    let mut latencies_us: Vec<f64> = Vec::with_capacity(queries);
    for &qi in &query_ids {
        let t = Instant::now();
        let hits = index.search(&data[qi], TOP_K, ef_search).expect("search");
        latencies_us.push(t.elapsed().as_secs_f64() * 1e6);
        assert!(!hits.is_empty());
    }
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "query (top {TOP_K}, ef {ef_search}, {queries} queries): p50 {:.3} ms, p95 {:.3} ms, p99 {:.3} ms",
        percentile(&latencies_us, 0.50) / 1000.0,
        percentile(&latencies_us, 0.95) / 1000.0,
        percentile(&latencies_us, 0.99) / 1000.0,
    );

    // Two recall measures against brute force. Set recall counts exact id matches. Tie-aware
    // recall counts any hit at least as similar as the true 10th neighbor, so near-equal
    // neighbors that swap places are not penalized.
    let mut set_found = 0usize;
    let mut tie_found = 0usize;
    let recall_ids = &query_ids[..RECALL_QUERIES.min(queries)];
    for &qi in recall_ids {
        let mut exact: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(&data[qi], v)))
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let truth: std::collections::HashSet<usize> =
            exact.iter().take(TOP_K).map(|(i, _)| *i).collect();
        let threshold = exact[TOP_K - 1].1 - 1e-6;
        let approx = index.search(&data[qi], TOP_K, ef_search).expect("search");
        set_found += approx.iter().filter(|(i, _)| truth.contains(i)).count();
        tie_found += approx.iter().filter(|(_, s)| *s >= threshold).count();
    }
    let slots = (recall_ids.len() * TOP_K) as f64;
    println!(
        "recall@{TOP_K} over {} queries: set {:.3}, tie-aware {:.3}",
        recall_ids.len(),
        set_found as f64 / slots,
        tie_found as f64 / slots
    );

    let path = std::env::temp_dir().join("tpt_rag_bench_hnsw.bin");
    let t = Instant::now();
    index.flush(Path::new(&path)).expect("flush");
    let flush_s = t.elapsed().as_secs_f64();
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let t = Instant::now();
    let loaded = HnswIndex::load_from_disk(Path::new(&path), DIM, 16, 200).expect("load");
    let load_s = t.elapsed().as_secs_f64();
    println!(
        "flush: {flush_s:.2}s ({:.1} MB), load: {load_s:.2}s, loaded {} vectors",
        bytes as f64 / 1e6,
        loaded.len()
    );
    let _ = std::fs::remove_file(&path);
}
