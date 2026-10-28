//! End-to-end tests against the real all-MiniLM-L6-v2 model.
//!
//! They need `models/all-MiniLM-L6-v2.onnx` and `models/vocab.txt` in the repository root.
//! Run `scripts/fetch_model.sh` (or `.ps1`) to download them. Without the model, each test
//! prints a notice and passes without checking anything.

use std::path::{Path, PathBuf};

use tpt_rag_core::{ChunkConfig, Metadata, Parser, QueryFilter, RAGLite, RagConfig, RagError};

fn model_path() -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/all-MiniLM-L6-v2.onnx");
    if path.exists() {
        Some(path)
    } else {
        eprintln!(
            "skipping: {} not found, run scripts/fetch_model",
            path.display()
        );
        None
    }
}

macro_rules! require_model {
    () => {
        match model_path() {
            Some(path) => path,
            None => return,
        }
    };
}

fn open(db: &Path, model: &Path) -> RAGLite {
    RAGLite::open_with_model(db, model).expect("open database")
}

fn write_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, content).expect("write fixture");
    path
}

const COOKING: &str = "To make sourdough bread, mix flour, water and salt, then let the dough \
rise overnight. Bake the loaf in a very hot oven until the crust turns deep brown.";
const SPACE: &str = "Jupiter is the largest planet in the solar system. It orbits the sun \
once every twelve years, far beyond the orbit of Mars.";
const GARDEN: &str = "Tomato plants need full sun and regular watering. Pinch out the side \
shoots so the plant puts its energy into fruit.";

#[test]
fn three_documents_top_result_is_the_right_one() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);

    let cooking = write_file(dir.path(), "cooking.txt", COOKING);
    let space = write_file(dir.path(), "space.txt", SPACE);
    let garden = write_file(dir.path(), "garden.txt", GARDEN);
    for file in [&cooking, &space, &garden] {
        rag.add_document(file, &["test"]).unwrap();
    }

    let results = rag
        .query("how long does it take a planet to orbit the sun", 3)
        .unwrap();
    assert_eq!(results.len(), 3);
    assert!(
        results[0].source.ends_with("space.txt"),
        "got {}",
        results[0].source
    );
    assert_eq!(results[0].tags, vec!["test"]);
}

#[test]
fn data_survives_reopen() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db");

    {
        let rag = open(&db, &model);
        rag.add_text(
            SPACE,
            &Metadata {
                source: Some("space".into()),
                tags: vec![],
            },
        )
        .unwrap();
        rag.add_text(
            GARDEN,
            &Metadata {
                source: Some("garden".into()),
                tags: vec![],
            },
        )
        .unwrap();
    }

    let rag = open(&db, &model);
    let stats = rag.stats().unwrap();
    assert_eq!(stats.chunks, 2);
    assert_eq!(stats.vectors, 2);
    assert!(stats.index_bytes > 0);
    let results = rag.query("tomato plants and watering", 1).unwrap();
    assert_eq!(results[0].source, "garden");
}

#[test]
fn ten_thousand_chunks_survive_restart() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db");
    // Windows of 8 words with 2 words of overlap give one chunk per 6 words.
    let config = || RagConfig {
        chunk: ChunkConfig {
            window_size: 8,
            overlap: 2,
            min_chunk_size: 1,
        },
        model_path: Some(model.clone()),
        ..RagConfig::default()
    };
    let text: String = (0..60_000).map(|i| format!("word{} ", i % 997)).collect();

    let before = {
        let rag = RAGLite::open_with_config(&db, config()).unwrap();
        rag.add_text(&text, &Metadata::default()).unwrap();
        let stats = rag.stats().unwrap();
        assert!(stats.chunks >= 9_990, "only {} chunks", stats.chunks);
        assert_eq!(stats.chunks, stats.vectors);
        stats.chunks
    };

    let rag = RAGLite::open_with_config(&db, config()).unwrap();
    let stats = rag.stats().unwrap();
    assert_eq!(stats.chunks, before);
    assert_eq!(stats.vectors, before);
    assert_eq!(rag.query("word5 word6", 5).unwrap().len(), 5);
}

#[test]
fn deleted_documents_disappear_and_stay_deleted() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db");
    let space = write_file(dir.path(), "space.txt", SPACE);
    let garden = write_file(dir.path(), "garden.txt", GARDEN);

    {
        let rag = open(&db, &model);
        rag.add_document(&space, &[]).unwrap();
        rag.add_document(&garden, &[]).unwrap();
        let removed = rag.delete_document(&space.to_string_lossy()).unwrap();
        assert_eq!(removed, 1);
        let results = rag.query("jupiter the largest planet", 5).unwrap();
        assert!(results.iter().all(|r| !r.source.ends_with("space.txt")));
    }

    let rag = open(&db, &model);
    assert_eq!(rag.chunk_count().unwrap(), 1);
    let results = rag.query("jupiter the largest planet", 5).unwrap();
    assert!(results.iter().all(|r| !r.source.ends_with("space.txt")));
}

#[test]
fn readding_a_file_replaces_its_chunks() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    let space = write_file(dir.path(), "space.txt", SPACE);

    rag.add_document(&space, &[]).unwrap();
    rag.add_document(&space, &[]).unwrap();
    assert_eq!(rag.chunk_count().unwrap(), 1);
}

#[test]
fn index_file_missing_is_rebuilt_from_metadata() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db");
    {
        let rag = open(&db, &model);
        rag.add_text(SPACE, &Metadata::default()).unwrap();
        rag.add_text(GARDEN, &Metadata::default()).unwrap();
    }
    std::fs::remove_file(db.join("hnsw.bin")).unwrap();

    let rag = open(&db, &model);
    assert_eq!(rag.vector_count(), 2);
    let results = rag.query("tomato plants", 1).unwrap();
    assert!(results[0].text.contains("Tomato"));
}

#[test]
fn corrupt_index_file_is_rebuilt_from_metadata() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db");
    {
        let rag = open(&db, &model);
        rag.add_text(SPACE, &Metadata::default()).unwrap();
        rag.add_text(GARDEN, &Metadata::default()).unwrap();
    }
    let index_path = db.join("hnsw.bin");
    let mut bytes = std::fs::read(&index_path).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&index_path, bytes).unwrap();

    let rag = open(&db, &model);
    assert_eq!(rag.vector_count(), 2);
    assert_eq!(rag.query("jupiter", 1).unwrap()[0].source, "inline");
}

#[test]
fn index_from_before_a_rebuild_is_detected_by_epoch() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db");
    let index_path = db.join("hnsw.bin");
    let stale;
    {
        let rag = open(&db, &model);
        rag.add_text(SPACE, &Metadata::default()).unwrap();
        rag.flush().unwrap();
        stale = std::fs::read(&index_path).unwrap();
        rag.rebuild_index().unwrap();
    }
    // Simulate a crash after an old index file was restored but before the metadata
    // could be updated: the file is valid, but its epoch no longer matches.
    std::fs::write(&index_path, stale).unwrap();

    let rag = open(&db, &model);
    assert_eq!(rag.query("jupiter", 1).unwrap().len(), 1);
    assert_eq!(rag.vector_count(), 1);
}

#[test]
fn different_model_is_refused_unless_reembedding_is_requested() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db");
    {
        let rag = open(&db, &model);
        rag.add_text(SPACE, &Metadata::default()).unwrap();
    }

    // The same weights under another file name count as another model.
    let other_dir = dir.path().join("other");
    std::fs::create_dir_all(&other_dir).unwrap();
    let other_model = other_dir.join("other-model.onnx");
    std::fs::copy(&model, &other_model).unwrap();
    std::fs::copy(
        model.with_file_name("vocab.txt"),
        other_dir.join("vocab.txt"),
    )
    .unwrap();

    let refused = RAGLite::open_with_model(&db, &other_model);
    assert!(matches!(refused, Err(RagError::Model(_))));

    let config = RagConfig {
        model_path: Some(other_model),
        reembed_on_model_change: true,
        ..RagConfig::default()
    };
    let rag = RAGLite::open_with_config(&db, config).unwrap();
    assert_eq!(rag.query("jupiter", 1).unwrap()[0].source, "inline");
    assert_eq!(rag.stats().unwrap().model, "other-model.onnx");
}

#[test]
fn hybrid_scores_are_normalized_and_keyword_matches_rank() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    rag.add_text(
        SPACE,
        &Metadata {
            source: Some("space".into()),
            tags: vec![],
        },
    )
    .unwrap();
    rag.add_text(
        GARDEN,
        &Metadata {
            source: Some("garden".into()),
            tags: vec![],
        },
    )
    .unwrap();

    let results = rag.query_hybrid("Jupiter", 2, 0.5).unwrap();
    assert_eq!(results[0].source, "space");
    assert!(results.iter().all(|r| r.score > 0.0 && r.score <= 1.0));
    assert!(rag.query_hybrid("anything", 0, 2.0).unwrap().is_empty());
}

#[test]
fn directory_ingestion_reports_failures_and_keeps_going() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir_all(docs.join("nested")).unwrap();
    write_file(&docs, "space.txt", SPACE);
    write_file(&docs.join("nested"), "garden.md", GARDEN);
    write_file(&docs.join("nested"), "empty.md", "   \n");
    write_file(&docs, "broken.json", "{ not json");
    write_file(&docs, "ignored.bin", "binary");

    let rag = open(&dir.path().join("db"), &model);
    let mut progress = Vec::new();
    let report = rag
        .add_directory(&docs, &[], &mut |i, total, _| progress.push((i, total)))
        .unwrap();

    assert_eq!(report.added.len(), 2, "added: {:?}", report.added);
    assert_eq!(report.failed.len(), 2, "failed: {:?}", report.failed);
    assert_eq!(progress.len(), 4);
    assert_eq!(progress.last(), Some(&(4, 4)));
}

#[test]
fn registered_parser_overrides_builtin_for_its_extension() {
    struct Shout;
    impl Parser for Shout {
        fn parse(&self, _path: &Path) -> tpt_rag_core::error::Result<String> {
            Ok("custom parser output".to_string())
        }
        fn supported_extensions(&self) -> &[&str] {
            &["txt"]
        }
    }

    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let mut rag = open(&dir.path().join("db"), &model);
    rag.register_parser(Box::new(Shout));
    let file = write_file(dir.path(), "notes.txt", SPACE);
    rag.add_document(&file, &[]).unwrap();

    let results = rag.query("custom", 1).unwrap();
    assert_eq!(results[0].text, "custom parser output");
}

#[test]
fn reranking_orders_by_cross_encoder_relevance() {
    let model = require_model!();
    let reranker =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/ms-marco-MiniLM-L-6-v2.onnx");
    if !reranker.exists() {
        eprintln!("skipping: {} not found", reranker.display());
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let config = RagConfig {
        model_path: Some(model),
        reranker_path: Some(reranker),
        ..RagConfig::default()
    };
    let rag = RAGLite::open_with_config(&dir.path().join("db"), config).unwrap();
    rag.add_text(
        SPACE,
        &Metadata {
            source: Some("space".into()),
            tags: vec![],
        },
    )
    .unwrap();
    rag.add_text(
        GARDEN,
        &Metadata {
            source: Some("garden".into()),
            tags: vec![],
        },
    )
    .unwrap();
    rag.add_text(
        COOKING,
        &Metadata {
            source: Some("cooking".into()),
            tags: vec![],
        },
    )
    .unwrap();

    let results = rag
        .query_reranked("how big is Jupiter compared to other planets", 2, 3)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].source, "space");
    assert!(results[0].score >= results[1].score);
    assert!(results.iter().all(|r| r.score > 0.0 && r.score < 1.0));
}

#[test]
fn reranking_without_a_reranker_model_is_an_error() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    let err = rag.query_reranked("anything", 1, 5).unwrap_err();
    assert!(err.to_string().contains("reranker"), "{err}");
}

#[cfg(feature = "embedded-model")]
#[test]
fn embedded_model_needs_no_model_files() {
    let dir = tempfile::tempdir().unwrap();
    let rag = RAGLite::open_with_config(
        &dir.path().join("db"),
        RagConfig {
            model_path: None,
            ..RagConfig::default()
        },
    );
    // The embedded default is used only when no model file is configured through
    // RAGLITE_MODEL. In that case the test is not meaningful, so it passes.
    if std::env::var_os("RAGLITE_MODEL").is_some() {
        return;
    }
    let rag = rag.unwrap();
    rag.add_text(SPACE, &Metadata::default()).unwrap();
    assert_eq!(rag.query("jupiter", 1).unwrap()[0].text, SPACE);
    assert_eq!(rag.stats().unwrap().model, "all-MiniLM-L6-v2 (embedded)");
}

#[test]
fn glob_ingestion_takes_only_matching_supported_files() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let notes = dir.path().join("notes").join("deep");
    std::fs::create_dir_all(&notes).unwrap();
    write_file(&notes, "space.md", SPACE);
    write_file(&dir.path().join("notes"), "garden.txt", GARDEN);
    write_file(&dir.path().join("notes"), "skip.bin", "binary");
    write_file(dir.path(), "outside.md", COOKING);

    let rag = open(&dir.path().join("db"), &model);
    // Glob patterns use forward slashes on every platform.
    let pattern = format!("{}/notes/**/*", dir.path().display()).replace('\\', "/");
    let report = rag
        .add_glob(&pattern, &["notes"], &mut |_, _, _| {})
        .unwrap();

    assert_eq!(report.added.len(), 2, "added: {:?}", report.added);
    assert!(report.failed.is_empty());
    assert!(report.added.iter().all(|p| !p.ends_with("outside.md")));
    assert_eq!(rag.chunk_count().unwrap(), 2);
}

#[test]
fn filtered_query_keeps_only_matching_tags_and_sources() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    rag.add_text(
        SPACE,
        &Metadata {
            source: Some("astro/space".into()),
            tags: vec!["science".into(), "public".into()],
        },
    )
    .unwrap();
    rag.add_text(
        "Jupiter is also a gas giant with dozens of moons.",
        &Metadata {
            source: Some("notes/jupiter".into()),
            tags: vec!["personal".into()],
        },
    )
    .unwrap();

    let by_tag = QueryFilter {
        tags: vec!["science".into()],
        source_prefix: None,
        created_since: None,
    };
    let hits = rag.query_filtered("Jupiter", 5, &by_tag).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].source, "astro/space");

    let by_prefix = QueryFilter {
        tags: vec![],
        source_prefix: Some("notes/".into()),
        created_since: None,
    };
    let hits = rag.query_filtered("Jupiter", 5, &by_prefix).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].source, "notes/jupiter");

    let none = QueryFilter {
        tags: vec!["science".into(), "personal".into()],
        source_prefix: None,
        created_since: None,
    };
    assert!(rag.query_filtered("Jupiter", 5, &none).unwrap().is_empty());
}

#[test]
fn check_reports_a_healthy_database_as_ok() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    rag.add_text(SPACE, &Metadata::default()).unwrap();
    rag.add_text(GARDEN, &Metadata::default()).unwrap();
    rag.flush().unwrap();

    let checks = rag.check().unwrap();
    for check in &checks {
        assert!(check.ok, "{}: {}", check.name, check.detail);
    }
    assert!(checks.iter().any(|c| c.name == "full-text index"));
}

#[test]
fn database_from_a_newer_schema_is_refused() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("db");
    drop(open(&db, &model));

    let store = tpt_rag_core::metadata::MetadataStore::open(&db).unwrap();
    store.set_meta("schema_version", "99").unwrap();
    drop(store);

    match RAGLite::open_with_model(&db, &model) {
        Err(RagError::SchemaTooNew { stored: 99, .. }) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("opened a database from a newer schema"),
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn pdf_with_a_text_layer_is_searchable() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);

    rag.add_document(&fixture("sample.pdf"), &[]).unwrap();
    let results = rag
        .query("which planet is known as the red planet", 1)
        .unwrap();
    assert!(results[0].source.ends_with("sample.pdf"));
    // The fixture has one page, so its chunks report page 1.
    assert_eq!(results[0].section.as_deref(), Some("page 1"));
    assert!(results[0].text.contains("Mars"), "{}", results[0].text);
}

#[test]
fn pdf_without_text_is_reported_as_empty() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);

    let err = rag.add_document(&fixture("blank.pdf"), &[]).unwrap_err();
    assert!(matches!(err, RagError::EmptyDocument(_)), "{err}");
    assert_eq!(rag.chunk_count().unwrap(), 0);
}

#[test]
fn csv_and_json_rows_are_searchable_with_their_field_names() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    let csv = write_file(
        dir.path(),
        "planets.csv",
        "name,diameter_km\nJupiter,139820\nMars,6779\n",
    );
    let json = write_file(
        dir.path(),
        "orbit.json",
        r#"{"planet": "Venus", "orbital_period_days": 225}"#,
    );
    rag.add_document(&csv, &[]).unwrap();
    rag.add_document(&json, &[]).unwrap();

    let results = rag.query("Venus orbital period", 1).unwrap();
    assert!(results[0].source.ends_with("orbit.json"));
    assert!(results[0].text.contains("orbital_period_days: 225"));
}

#[test]
fn invalid_config_is_rejected_before_loading_anything() {
    let dir = tempfile::tempdir().unwrap();
    let config = RagConfig {
        m: 0,
        ..RagConfig::default()
    };
    assert!(matches!(
        RAGLite::open_with_config(&dir.path().join("db"), config),
        Err(RagError::InvalidConfig(_))
    ));
}

#[test]
fn diverse_query_avoids_near_duplicates() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    let texts = [
        ("a", "Jupiter is the largest planet in the solar system."),
        ("b", "Jupiter is the largest planet in our solar system."),
        ("c", "Bananas are a good source of potassium."),
    ];
    for (source, text) in texts {
        rag.add_text(
            text,
            &Metadata {
                source: Some(source.into()),
                tags: vec![],
            },
        )
        .unwrap();
    }
    let question = "largest planet in the solar system";

    // Weight 1.0 is plain relevance order, so it matches the ordinary query.
    let plain: Vec<u64> = rag
        .query(question, 2)
        .unwrap()
        .iter()
        .map(|r| r.chunk_id)
        .collect();
    let relevant: Vec<u64> = rag
        .query_diverse(question, 2, 1.0)
        .unwrap()
        .iter()
        .map(|r| r.chunk_id)
        .collect();
    assert_eq!(relevant, plain);

    // Weight 0.0 keeps one Jupiter sentence and then the unrelated one.
    let varied = rag.query_diverse(question, 2, 0.0).unwrap();
    assert_eq!(varied.len(), 2);
    assert!(
        varied[1].text.contains("Bananas"),
        "got {:?}",
        varied[1].text
    );
}

#[test]
fn offsets_locate_each_result_in_the_ingested_text() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    let text = vec!["Backups run every night at two in the morning."; 150].join(" ");
    rag.add_text(
        &text,
        &Metadata {
            source: Some("long".into()),
            tags: vec![],
        },
    )
    .unwrap();

    let hits = rag.query("when do backups run", 10).unwrap();
    assert!(!hits.is_empty());
    for hit in &hits {
        let start = hit.start_offset.expect("start offset") as usize;
        let end = hit.end_offset.expect("end offset") as usize;
        assert_eq!(&text[start..end], hit.text);
    }
}

#[test]
fn neighbors_surround_a_hit_in_document_order() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    let text = vec!["Backups run every night at two in the morning."; 150].join(" ");
    rag.add_text(
        &text,
        &Metadata {
            source: Some("long".into()),
            tags: vec![],
        },
    )
    .unwrap();

    let hit = rag.query("when do backups run", 1).unwrap().remove(0);
    let around = rag.neighbors(hit.chunk_id, 1).unwrap();
    assert!(
        !around.is_empty() && around.len() <= 2,
        "got {} neighbours",
        around.len()
    );
    for n in &around {
        let start = n.start_offset.expect("start offset") as usize;
        let end = n.end_offset.expect("end offset") as usize;
        assert_eq!(&text[start..end], n.text);
        assert_eq!(n.score, 0.0);
        assert_ne!(n.chunk_id, hit.chunk_id);
    }
    assert!(around
        .windows(2)
        .all(|w| w[0].start_offset < w[1].start_offset));
}

#[test]
fn markdown_results_report_their_section() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    let doc = format!(
        "# Setup\n\n{}\n\n# Lounge\n\n{}",
        "Backups run every night at two in the morning. ".repeat(60),
        "Tea is served warm in the lounge every afternoon. ".repeat(60),
    );
    let guide = write_file(dir.path(), "guide.md", &doc);
    rag.add_document(&guide, &[]).unwrap();

    let backups = rag.query("when do backups run", 1).unwrap().remove(0);
    assert_eq!(backups.section.as_deref(), Some("Setup"));
    let tea = rag.query("where is tea served", 1).unwrap().remove(0);
    assert_eq!(tea.section.as_deref(), Some("Lounge"));
    // No chunk spans the boundary between the two sections.
    for hit in rag.query("backups tea lounge", 10).unwrap() {
        assert!(
            !(hit.text.contains("Backups") && hit.text.contains("Tea")),
            "{}",
            hit.text
        );
    }
}

#[test]
fn plain_text_results_have_no_section() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    rag.add_text(
        "Jupiter is a gas giant.",
        &Metadata {
            source: Some("notes".into()),
            tags: vec![],
        },
    )
    .unwrap();
    assert_eq!(rag.query("gas giant", 1).unwrap()[0].section, None);
}

#[test]
fn sync_indexes_changes_and_removes_deleted_files() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let notes = dir.path().join("notes");
    std::fs::create_dir(&notes).unwrap();
    let rag = open(&dir.path().join("db"), &model);
    write_file(&notes, "space.txt", SPACE);
    write_file(&notes, "garden.txt", GARDEN);

    let first = rag.sync_directory(&notes, &[]).unwrap();
    assert_eq!(first.indexed.len(), 2);
    assert!(first.failed.is_empty());

    let again = rag.sync_directory(&notes, &[]).unwrap();
    assert!(again.indexed.is_empty());
    assert_eq!(again.unchanged, 2);

    write_file(&notes, "space.txt", COOKING);
    let changed = rag.sync_directory(&notes, &[]).unwrap();
    assert_eq!(changed.indexed.len(), 1);
    assert_eq!(changed.unchanged, 1);
    assert!(rag.query("sourdough bread", 1).unwrap()[0]
        .text
        .contains("sourdough"));

    std::fs::remove_file(notes.join("garden.txt")).unwrap();
    let removed = rag.sync_directory(&notes, &[]).unwrap();
    assert_eq!(removed.removed.len(), 1);
    assert!(removed.removed[0].ends_with("garden.txt"));
    assert!(rag
        .query("tomato plants", 5)
        .unwrap()
        .iter()
        .all(|r| !r.source.ends_with("garden.txt")));

    // A removed file that comes back is indexed again.
    write_file(&notes, "garden.txt", GARDEN);
    let back = rag.sync_directory(&notes, &[]).unwrap();
    assert_eq!(back.indexed.len(), 1);
}

#[test]
fn sync_ingests_a_file_again_after_add_document() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let notes = dir.path().join("notes");
    std::fs::create_dir(&notes).unwrap();
    let rag = open(&dir.path().join("db"), &model);
    let space = write_file(&notes, "space.txt", SPACE);

    assert_eq!(rag.sync_directory(&notes, &[]).unwrap().indexed.len(), 1);
    // Re-adding by hand replaces the chunks, so the next sync must not skip the file.
    rag.add_document(&space, &[]).unwrap();
    assert_eq!(rag.sync_directory(&notes, &[]).unwrap().indexed.len(), 1);
}

#[test]
fn generated_cases_find_their_own_source() {
    let model = require_model!();
    let dir = tempfile::tempdir().unwrap();
    let rag = open(&dir.path().join("db"), &model);
    for (source, text) in [("space", SPACE), ("cooking", COOKING), ("garden", GARDEN)] {
        rag.add_text(
            text,
            &Metadata {
                source: Some(source.into()),
                tags: vec![],
            },
        )
        .unwrap();
    }

    let cases = rag.eval_cases(1).unwrap();
    assert_eq!(cases.len(), 3);
    for case in &cases {
        let found = rag.query(&case.query, 3).unwrap();
        assert!(
            found.iter().any(|r| r.source == case.expected_source),
            "{:?} did not find {}",
            case.query,
            case.expected_source
        );
    }
}
