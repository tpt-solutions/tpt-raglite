"""Minimal tpt-raglite example: ingest a few files and ask a question.

Run from the repository root after `maturin develop` in tpt-rag-py/:

    RAGLITE_MODEL=models/all-MiniLM-L6-v2.onnx python examples/python_quickstart.py
"""

from pathlib import Path

from tpt_raglite import RAGLite

db = RAGLite("./quickstart_db")

notes = Path("./quickstart_notes.md")
notes.write_text(
    "# Backups\n\nThe nightly backup runs at 02:00 and keeps seven days of history.\n"
)
db.add_files([str(notes)], tags=["ops"])
db.add_text("Jupiter is the largest planet in the solar system.", source="space")

for result in db.query("when do backups run", top_k=3):
    print(f"[{result.score:.2f}] {result.source}: {result.text}")

db.flush()
