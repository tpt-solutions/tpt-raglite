"""Answers questions about a folder of PDFs, showing the passage each answer came from.

Scanned PDFs have no text layer. They are reported as failures, not silently skipped, so
run OCR on them first.

    python examples/python_pdf_qa.py ./manuals "How do I reset the device?"
"""

import sys

from tpt_raglite import RAGLite


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python_pdf_qa.py <pdf folder> <question>")
    folder, question = sys.argv[1], sys.argv[2]

    db = RAGLite("./pdf_qa_db")
    report = db.add_directory(folder, tags=["pdf"])
    print(f"ingested {len(report['added'])} file(s)")
    for path, reason in report["failed"]:
        print(f"  skipped {path}: {reason}")

    results = db.query(question, top_k=3)
    if not results:
        print("no matching passages")
        return
    for rank, result in enumerate(results, start=1):
        print(f"\n#{rank} score {result.score:.2f} from {result.source}")
        print(result.text)

    db.flush()


if __name__ == "__main__":
    main()
