"""Unit tests for cv_parser.py"""
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from cv_parser import extract_cv_text


def _make_pdf_mock(page_texts: list[str | None]):
    """Build a mock pdfplumber PDF with the given per-page text values."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)

    pdf_mock = MagicMock()
    pdf_mock.pages = pages
    pdf_mock.__enter__ = lambda s: s
    pdf_mock.__exit__ = MagicMock(return_value=False)
    return pdf_mock


@patch("cv_parser.pdfplumber.open")
def test_single_page_returns_text(mock_open):
    mock_open.return_value = _make_pdf_mock(["John Doe\nData Engineer"])

    result = extract_cv_text("cv.pdf")

    assert "John Doe" in result
    assert "Data Engineer" in result


@patch("cv_parser.pdfplumber.open")
def test_multiple_pages_joined(mock_open):
    mock_open.return_value = _make_pdf_mock(["Page one content", "Page two content"])

    result = extract_cv_text("cv.pdf")

    assert "Page one content" in result
    assert "Page two content" in result
    assert "Page 1" in result
    assert "Page 2" in result


@patch("cv_parser.pdfplumber.open")
def test_none_pages_skipped(mock_open):
    """Pages returning None from extract_text() should be silently skipped."""
    mock_open.return_value = _make_pdf_mock([None, "Second page text", None])

    result = extract_cv_text("cv.pdf")

    assert "Second page text" in result
    assert "Page 1" not in result  # skipped — no text
    assert "Page 2" in result


@patch("cv_parser.pdfplumber.open")
def test_all_empty_pages_returns_empty_string(mock_open):
    mock_open.return_value = _make_pdf_mock([None, None])

    result = extract_cv_text("cv.pdf")

    assert result == ""


@patch("cv_parser.pdfplumber.open")
def test_accepts_bytesio(mock_open):
    """extract_cv_text should accept a BytesIO object, not just a path string."""
    mock_open.return_value = _make_pdf_mock(["BytesIO content"])

    result = extract_cv_text(BytesIO(b"fake-pdf-bytes"))

    assert "BytesIO content" in result
    # pdfplumber.open was called with the BytesIO object
    mock_open.assert_called_once()
