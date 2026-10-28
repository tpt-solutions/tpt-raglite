"""Tests for the sync logic. They use a fake database, so no model is needed.

Run with: python -m pytest templates/local-notes-search
"""

import os
from pathlib import Path

from notes_search import scan, search, sync


class FakeDB:
    def __init__(self, fail_on=()):
        self.indexed = {}
        self.deleted = []
        self.flushed = 0
        self.fail_on = set(fail_on)

    def add_files(self, paths, tags=None):
        for path in paths:
            if path in self.fail_on:
                raise RuntimeError("unreadable")
            self.indexed[path] = self.indexed.get(path, 0) + 1

    def delete_document(self, source):
        self.deleted.append(source)
        return 1 if self.indexed.pop(source, None) else 0

    def query(self, text, top_k=5):
        return []

    def flush(self):
        self.flushed += 1


def write(path: Path, text: str, mtime: int) -> str:
    """Writes a note with a whole-second mtime. Windows keeps no finer resolution here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return str(path)


def mtime_ns(source: str) -> int:
    return Path(source).stat().st_mtime_ns


def test_scan_keeps_only_note_formats(tmp_path):
    keep = write(tmp_path / "a.md", "x", 1)
    write(tmp_path / "image.png", "x", 1)
    nested = write(tmp_path / "sub" / "b.txt", "x", 1)

    assert scan(tmp_path) == {keep: mtime_ns(keep), nested: mtime_ns(nested)}


def test_first_sync_indexes_every_note(tmp_path):
    a = write(tmp_path / "a.md", "alpha", 10)
    b = write(tmp_path / "b.md", "beta", 20)
    db = FakeDB()

    manifest, failures = sync(db, tmp_path, {})

    assert manifest == {a: mtime_ns(a), b: mtime_ns(b)}
    assert failures == {}
    assert set(db.indexed) == {a, b}
    assert db.flushed == 1


def test_unchanged_notes_are_not_read_again(tmp_path):
    a = write(tmp_path / "a.md", "alpha", 10)
    db = FakeDB()

    sync(db, tmp_path, {a: mtime_ns(a)})

    assert db.indexed == {}


def test_changed_note_is_reingested(tmp_path):
    a = write(tmp_path / "a.md", "alpha v2", 99)
    db = FakeDB()

    manifest, _ = sync(db, tmp_path, {a: mtime_ns(a) - 1})

    assert db.indexed == {a: 1}
    assert manifest == {a: mtime_ns(a)}


def test_deleted_note_is_removed_from_index(tmp_path):
    gone = str(tmp_path / "gone.md")
    db = FakeDB()
    db.indexed[gone] = 1

    manifest, _ = sync(db, tmp_path, {gone: 5})

    assert db.deleted == [gone]
    assert manifest == {}


def test_failed_note_is_reported_and_retried_next_time(tmp_path):
    bad = write(tmp_path / "bad.pdf", "not really a pdf", 7)
    db = FakeDB(fail_on={bad})

    manifest, failures = sync(db, tmp_path, {})

    assert failures == {bad: "unreadable"}
    assert bad not in manifest

    db.fail_on.clear()
    manifest, failures = sync(db, tmp_path, manifest)

    assert failures == {}
    assert manifest == {bad: mtime_ns(bad)}


def test_search_returns_source_score_and_text(tmp_path):
    class Hit:
        source = "a.md"
        score = 0.8
        text = "backups run nightly"

    class DB(FakeDB):
        def query(self, text, top_k=5):
            assert top_k == 3
            return [Hit()]

    assert search(DB(), "backups", top_k=3) == [("a.md", 0.8, "backups run nightly")]
