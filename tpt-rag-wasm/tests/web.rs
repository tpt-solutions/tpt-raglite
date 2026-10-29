//! Runs the bindings as JavaScript sees them. `wasm-pack test --node` runs these in Node.

use tpt_rag_wasm::TptRag;
use wasm_bindgen_test::*;

fn embed(chunks_json: &str, dimensions: usize) -> Vec<f32> {
    // A stand-in for the host's model: one fixed vector per chunk, so the test needs no model.
    let count = chunks_json.matches("\"text\"").count();
    let mut out = Vec::with_capacity(count * dimensions);
    for i in 0..count {
        for d in 0..dimensions {
            out.push(if d == i % dimensions { 1.0 } else { 0.0 });
        }
    }
    out
}

#[wasm_bindgen_test]
fn plan_add_query_and_keyword_round_trip() {
    let mut rag = TptRag::new(3).unwrap();
    let plan = rag
        .plan(
            "markdown",
            "# Setup\n\nBackups run every night.\n\n# Garden\n\nTomatoes need sun.\n",
        )
        .unwrap();
    let vectors = embed(&plan, 3);
    let stored = rag
        .add("notes", vec!["ops".into()], &plan, &vectors)
        .unwrap();
    assert_eq!(stored, plan.matches("\"text\"").count());

    let keyword: serde_json::Value =
        serde_json::from_str(&rag.keyword("tomatoes", 1).unwrap()).unwrap();
    assert_eq!(keyword[0]["source"], "notes");
    assert_eq!(keyword[0]["section"], "Garden");

    let nearest: serde_json::Value =
        serde_json::from_str(&rag.query(&[1.0, 0.0, 0.0], 1).unwrap()).unwrap();
    assert_eq!(nearest[0]["section"], "Setup");
}

#[wasm_bindgen_test]
fn export_and_import_keep_the_database() {
    let mut rag = TptRag::new(3).unwrap();
    let plan = rag.plan("text", "Jupiter is the largest planet.").unwrap();
    rag.add("space", vec![], &plan, &embed(&plan, 3)).unwrap();

    let bytes = rag.export().unwrap();
    let loaded = TptRag::import(&bytes).unwrap();
    let stats: serde_json::Value = serde_json::from_str(&loaded.stats().unwrap()).unwrap();
    assert_eq!(stats["chunks"], 1);
    assert_eq!(stats["dimensions"], 3);
    assert!(loaded.keyword("jupiter", 1).unwrap().contains("space"));
}

#[wasm_bindgen_test]
fn bad_input_throws_instead_of_panicking() {
    let mut rag = TptRag::new(3).unwrap();
    assert!(rag.plan("pdf", "x").is_err());
    let plan = rag.plan("text", "One sentence about gardens.").unwrap();
    assert!(rag.add("x", vec![], &plan, &[1.0]).is_err());
    assert!(TptRag::import(b"nope").is_err());
}

#[wasm_bindgen_test]
fn hybrid_search_returns_fused_results() {
    let mut rag = TptRag::new(3).unwrap();
    let plan = rag.plan("text", "Jupiter is the largest planet.").unwrap();
    rag.add("space", vec![], &plan, &embed(&plan, 3)).unwrap();
    let plan = rag.plan("text", "Tomato plants need sun.").unwrap();
    rag.add("garden", vec![], &plan, &embed(&plan, 3)).unwrap();

    let hits: serde_json::Value =
        serde_json::from_str(&rag.hybrid(&[1.0, 0.0, 0.0], "tomato", 2, 0.5).unwrap()).unwrap();
    assert_eq!(hits.as_array().unwrap().len(), 2);
    assert!(hits[0]["score"].as_f64().unwrap() <= 1.0);
}
