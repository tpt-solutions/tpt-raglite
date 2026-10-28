"""Integration tests for the Python bindings.

They need the model files in models/ (run scripts/fetch_model.sh) and a built extension
(`maturin develop` in tpt-rag-py/). Run with `pytest tpt-rag-py/tests`.
"""

import os
from pathlib import Path

import pytest

from tpt_raglite import (
    EmptyDocumentError,
    ModelError,
    RAGLite,
    RAGLiteError,
    SearchResult,
    UnsupportedFormatError,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("RAGLITE_MODEL", ROOT / "models" / "all-MiniLM-L6-v2.onnx"))
RERANKER = ROOT / "models" / "ms-marco-MiniLM-L-6-v2.onnx"
FIXTURES = ROOT / "tpt-rag-core" / "tests" / "fixtures"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason=f"{MODEL} not found; run scripts/fetch_model.sh"
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "db")


@pytest.fixture
def db(db_path):
    return RAGLite(db_path, model_path=str(MODEL))


def test_exceptions_share_a_base_class():
    for error in (ModelError, UnsupportedFormatError, EmptyDocumentError):
        assert issubclass(error, RAGLiteError)


def test_add_text_and_query_returns_search_results(db):
    db.add_text("Jupiter is the largest planet in the solar system.", source="space")
    db.add_text("Tomato plants need full sun and regular watering.", source="garden")

    results = db.query("how big is Jupiter", top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].source == "space"
    assert -1.0 <= results[0].score <= 1.0
    assert "SearchResult(" in repr(results[0])


def test_add_files_keeps_tags_and_reads_pdfs(db):
    db.add_files([str(FIXTURES / "sample.pdf")], tags=["docs"])

    results = db.query("which planet is known as the red planet", top_k=1)
    assert "Mars" in results[0].text
    assert results[0].tags == ["docs"]


def test_add_directory_reports_added_and_failed(db, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("# Notes\n\nBake sourdough with flour and water.")
    (docs / "broken.json").write_text("{ not json")
    (docs / "ignored.bin").write_bytes(b"\x00\x01")

    report = db.add_directory(str(docs))

    assert [Path(p).name for p in report["added"]] == ["notes.md"]
    assert len(report["failed"]) == 1
    assert Path(report["failed"][0][0]).name == "broken.json"
    assert db.stats()["chunks"] == 1


def test_unsupported_extension_raises_typed_error(db, tmp_path):
    odd = tmp_path / "data.xyz"
    odd.write_text("content")
    with pytest.raises(UnsupportedFormatError):
        db.add_files([str(odd)])


def test_pdf_without_text_raises_empty_document(db):
    with pytest.raises(EmptyDocumentError):
        db.add_files([str(FIXTURES / "blank.pdf")])


def test_missing_model_raises_model_error(db_path, tmp_path):
    with pytest.raises(ModelError):
        RAGLite(db_path, model_path=str(tmp_path / "missing.onnx"))


def test_delete_document_and_stats(db):
    db.add_text("Mars is the red planet.", source="mars")
    db.add_text("Venus is the hottest planet.", source="venus")

    assert db.delete_document("mars") == 1
    stats = db.stats()
    assert stats["chunks"] == 1
    assert stats["model"] == MODEL.name
    assert all(r.source != "mars" for r in db.query("red planet", top_k=5))


def test_hybrid_query_scores_are_normalized(db):
    db.add_text("Jupiter is the largest planet.", source="space")
    db.add_text("Tomato plants need sun.", source="garden")

    results = db.query_hybrid("Jupiter", top_k=2, alpha=0.5)
    assert results[0].source == "space"
    assert all(0.0 < r.score <= 1.0 for r in results)


def test_data_survives_reopen_after_flush(db_path):
    first = RAGLite(db_path, model_path=str(MODEL))
    first.add_text("Jupiter is the largest planet.", source="space")
    first.flush()
    del first

    second = RAGLite(db_path, model_path=str(MODEL))
    assert second.query("largest planet", top_k=1)[0].source == "space"


def test_context_manager_flushes_on_exit(db_path):
    with RAGLite(db_path, model_path=str(MODEL)) as db:
        db.add_text("Saturn has rings.", source="saturn")

    with RAGLite(db_path, model_path=str(MODEL)) as db:
        assert db.stats()["chunks"] == 1


def test_rebuild_index_keeps_results(db):
    db.add_text("Saturn has rings.", source="saturn")
    assert db.rebuild_index() == 1
    assert db.query("rings", top_k=1)[0].source == "saturn"


@pytest.mark.skipif(not RERANKER.exists(), reason="reranker model not found")
def test_reranking_orders_by_relevance(db_path):
    db = RAGLite(db_path, model_path=str(MODEL), reranker_path=str(RERANKER))
    db.add_text("Jupiter is the largest planet in the solar system.", source="space")
    db.add_text("Tomato plants need full sun and regular watering.", source="garden")
    db.add_text("Sourdough needs a long overnight rise.", source="bread")

    results = db.query_reranked("how big is Jupiter", top_k=2, rerank_top_n=3)
    assert [r.source for r in results][0] == "space"
    assert results[0].score >= results[1].score
    assert all(0.0 < r.score < 1.0 for r in results)


def test_query_filters_by_tag_and_source_prefix(db):
    db.add_text("Jupiter is the largest planet.", source="astro/space", tags=["science"])
    db.add_text("Jupiter has many moons too.", source="notes/moons", tags=["personal"])

    by_tag = db.query("Jupiter", top_k=5, tags=["science"])
    assert [r.source for r in by_tag] == ["astro/space"]

    by_prefix = db.query("Jupiter", top_k=5, source_prefix="notes/")
    assert [r.source for r in by_prefix] == ["notes/moons"]


def test_add_glob_takes_matching_files(db, tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nSourdough needs an overnight rise.")
    (tmp_path / "b.txt").write_text("Tomatoes like full sun.")
    (tmp_path / "c.bin").write_bytes(b"\x00")

    report = db.add_glob(str(tmp_path / "*.md").replace("\\", "/"), tags=["kb"])

    assert [Path(p).name for p in report["added"]] == ["a.md"]
    assert report["failed"] == []


def test_check_reports_healthy_database(db):
    db.add_text("Saturn has rings.", source="saturn")
    checks = db.check()
    assert checks and all(c["ok"] for c in checks), checks
    assert {"name", "ok", "detail"} <= set(checks[0])


def test_reranking_without_a_model_is_an_error(db):
    db.add_text("Saturn has rings.", source="saturn")
    with pytest.raises(ModelError, match="reranker"):
        db.query_reranked("rings", top_k=1)


def test_diverse_query_and_since_filter(db):
    db.add_text("Jupiter is the largest planet in the solar system.", source="a")
    db.add_text("Jupiter is the largest planet in our solar system.", source="b")
    db.add_text("Bananas are a good source of potassium.", source="c")
    question = "largest planet in the solar system"

    plain = [r.chunk_id for r in db.query(question, top_k=2)]
    relevant = [r.chunk_id for r in db.query_diverse(question, top_k=2, relevance_weight=1.0)]
    assert relevant == plain

    varied = db.query_diverse(question, top_k=2, relevance_weight=0.0)
    assert "Bananas" in varied[1].text

    assert db.query(question, top_k=5, since=2**40) == []
    assert len(db.query(question, top_k=5, since=0)) == 3


def test_results_carry_offsets_into_the_source_text(db):
    # ASCII only: Python string indexes are characters, while the offsets are bytes.
    text = " ".join(["Backups run every night at two in the morning."] * 150)
    db.add_text(text, source="long")
    hits = db.query("when do backups run", top_k=5)
    assert hits
    for hit in hits:
        assert text[hit.start_offset : hit.end_offset] == hit.text


def test_neighbors_return_the_chunks_around_a_hit(db):
    text = " ".join(["Backups run every night at two in the morning."] * 150)
    db.add_text(text, source="long")
    hit = db.query("when do backups run", top_k=1)[0]
    around = db.neighbors(hit.chunk_id, radius=1)
    assert 0 < len(around) <= 2
    assert all(n.source == "long" and n.chunk_id != hit.chunk_id for n in around)


def test_markdown_results_report_their_section(db, tmp_path):
    body_a = "Backups run every night at two in the morning. " * 60
    body_b = "Tea is served warm in the lounge every afternoon. " * 60
    guide = tmp_path / "guide.md"
    guide.write_text(f"# Setup\n\n{body_a}\n\n# Lounge\n\n{body_b}", encoding="utf-8")
    db.add_files([str(guide)])
    assert db.query("when do backups run", top_k=1)[0].section == "Setup"
    assert db.query("where is tea served", top_k=1)[0].section == "Lounge"


def test_pdf_results_report_their_page(db):
    db.add_files([str(FIXTURES / "sample.pdf")])
    hit = db.query("which planet is known as the red planet", top_k=1)[0]
    assert hit.section == "page 1"


def test_sync_directory_indexes_changes_and_removes(db, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("Backups run every night at two.", encoding="utf-8")
    (notes / "b.txt").write_text("Tomato plants need full sun.", encoding="utf-8")

    first = db.sync_directory(str(notes))
    assert len(first["indexed"]) == 2
    again = db.sync_directory(str(notes))
    assert again["indexed"] == [] and again["unchanged"] == 2

    (notes / "b.txt").unlink()
    gone = db.sync_directory(str(notes))
    assert len(gone["removed"]) == 1 and gone["removed"][0].endswith("b.txt")
    assert all(not r.source.endswith("b.txt") for r in db.query("tomato plants", top_k=5))


def test_chunks_do_not_cross_headings(db, tmp_path):
    body_a = "Backups run every night at two in the morning. " * 60
    body_b = "Tea is served warm in the lounge every afternoon. " * 60
    guide = tmp_path / "guide.md"
    guide.write_text(f"# Setup\n\n{body_a}\n\n# Lounge\n\n{body_b}", encoding="utf-8")
    db.add_files([str(guide)])
    for hit in db.query("backups tea lounge", top_k=10):
        assert not ("Backups" in hit.text and "Tea" in hit.text), hit.text
