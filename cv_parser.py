"""Extract text from CV PDFs (path, Path, BytesIO, or Streamlit UploadedFile-like)."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union

import pdfplumber

FileInput = Union[str, Path, BinaryIO]


def extract_cv_text(file: FileInput) -> str:
    """
    Extract all text from a PDF CV.
    Accepts: filesystem path, pathlib.Path, BytesIO, or any object with .read().
    """
    pdf_source: str | BinaryIO
    if isinstance(file, Path):
        pdf_source = str(file)
    elif isinstance(file, str):
        pdf_source = file
    else:
        data = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
        pdf_source = BytesIO(data)

    text_parts: list[str] = []
    with pdfplumber.open(pdf_source) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text_parts.append(f"--- Page {page_num} ---\n{text}")
    return "\n\n".join(text_parts)


def extract_cv_structured(pdf_path: str | Path) -> dict:
    """Extract text and table data from a PDF CV."""
    path = str(pdf_path)
    result: dict = {"pages": []}

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_data = {
                "page": page_num,
                "text": page.extract_text() or "",
                "tables": page.extract_tables() or [],
            }
            result["pages"].append(page_data)

    return result


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Extract text from a CV PDF.")
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Path to PDF file (required unless piping).",
    )
    args = parser.parse_args()
    if not args.pdf:
        print("Usage: python cv_parser.py <path/to/cv.pdf>", file=sys.stderr)
        raise SystemExit(1)
    print(extract_cv_text(args.pdf))
