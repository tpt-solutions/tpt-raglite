"""Writes the PDF fixtures used by the tests. Run from this folder: python make_pdf_fixtures.py

sample.pdf  one page with a text layer (the query target).
blank.pdf   one page with no text layer, like a scanned page without OCR.
"""

from pathlib import Path


def build_pdf(page_text: str) -> bytes:
    if page_text:
        escaped = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content = f"BT /F1 18 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    else:
        content = b""

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


if __name__ == "__main__":
    here = Path(__file__).parent
    (here / "sample.pdf").write_bytes(
        build_pdf("Mars is the fourth planet from the sun and is known as the red planet.")
    )
    (here / "blank.pdf").write_bytes(build_pdf(""))
    print("wrote sample.pdf and blank.pdf")
