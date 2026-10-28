"""Local notes search: index a folder of notes and find them by meaning.

Each run compares the notes folder with the last sync. New and changed notes are ingested,
and notes that were deleted are removed from the index. Nothing leaves the machine.

    python notes_search.py ./notes "how did I set up the backups?"

The sync state lives in `<db>/sync.json`, so a second run only re-reads what changed.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Protocol, Sequence

NOTE_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}
MANIFEST_NAME = "sync.json"


class Notes(Protocol):
    """The part of RAGLite this module uses. Tests can pass a fake."""

    def add_files(self, paths: list[str], tags: Optional[list[str]] = None) -> None: ...

    def delete_document(self, source: str) -> int: ...

    def query(self, text: str, top_k: int = 5) -> Sequence[object]: ...

    def flush(self) -> None: ...


def scan(folder: Path) -> dict[str, int]:
    """Maps each note's path to its modification time, in nanoseconds."""
    return {
        str(path): path.stat().st_mtime_ns
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in NOTE_SUFFIXES
    }


def sync(db: Notes, folder: Path, manifest: dict[str, int]) -> tuple[dict[str, int], dict[str, str]]:
    """Brings the index in line with the folder.

    Returns the new manifest and a map of notes that failed to ingest to the reason. A failed
    note is left out of the manifest, so the next sync tries it again.
    """
    current = scan(folder)

    for source in sorted(set(manifest) - set(current)):
        db.delete_document(source)

    failures: dict[str, str] = {}
    updated: dict[str, int] = {}
    for source, mtime in current.items():
        if manifest.get(source) == mtime:
            updated[source] = mtime
            continue
        # Re-adding a note replaces its earlier chunks, so no delete is needed first.
        try:
            db.add_files([source])
            updated[source] = mtime
        except Exception as error:  # One bad note must not stop the rest of the sync.
            failures[source] = str(error)

    db.flush()
    return updated, failures


def search(db: Notes, question: str, top_k: int = 5) -> list[tuple[str, float, str]]:
    """Returns (source, score, text) for the notes that best match the question."""
    return [(r.source, r.score, r.text) for r in db.query(question, top_k=top_k)]


def load_manifest(path: Path) -> dict[str, int]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search a folder of notes by meaning.")
    parser.add_argument("notes", type=Path, help="folder with your notes")
    parser.add_argument("question", nargs="?", help="what to look for; omit to only sync")
    parser.add_argument("--db", type=Path, default=Path("./notes_db"), help="index folder")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="results to show")
    args = parser.parse_args(argv)

    from tpt_raglite import RAGLite

    args.db.mkdir(parents=True, exist_ok=True)
    manifest_path = args.db / MANIFEST_NAME
    db = RAGLite(str(args.db))

    manifest, failures = sync(db, args.notes, load_manifest(manifest_path))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Indexed {len(manifest)} note(s).")
    for source, reason in failures.items():
        print(f"  skipped {source}: {reason}")

    if args.question:
        for source, score, text in search(db, args.question, args.top_k):
            snippet = " ".join(text.split())[:200]
            print(f"\n{score:.3f}  {source}\n  {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
