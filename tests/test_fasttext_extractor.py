import json
from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from strategies.FastTextExtractor import FastTextExtractor


class FakePage:
    def __init__(
        self,
        text: str,
        width: float = 612,
        height: float = 792,
        images=None,
        chars=None,
        words=None,
    ):
        self.width = width
        self.height = height
        self._text = text
        self.images = images or []
        self.chars = chars or []
        self._words = words or []

    def extract_text(self):
        return self._text

    def extract_words(self):
        return self._words


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_high_confidence_text_pdf(monkeypatch, tmp_path):
    words = [
        {"text": "hello", "x0": 50, "x1": 90, "top": 700, "bottom": 720},
        {"text": "world", "x0": 95, "x1": 140, "top": 700, "bottom": 720},
    ]
    page = FakePage(
        text="Hello world " * 200,
        images=[],
        chars=[{"fontname": "Helvetica"} for _ in range(10)],
        words=words,
    )
    module = sys.modules[FastTextExtractor.__module__]
    monkeypatch.setattr(module.pdfplumber, "open", lambda _: FakePDF([page]))

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    extractor = FastTextExtractor()
    doc = extractor.extract(pdf_path)

    assert doc.doc_id == "doc"
    assert doc.pages[0].confidence == "high_confidence"
    assert doc.pages[0].needs_layout_escalation is False
    assert len(doc.pages[0].text_blocks) == len(words)
    assert doc.total_time_ms >= doc.pages[0].extraction_time_ms


def test_low_confidence_image_heavy(monkeypatch, tmp_path):
    page = FakePage(
        text="",
        images=[{"width": 600, "height": 750}],
        chars=[],
        words=[],
    )
    module = sys.modules[FastTextExtractor.__module__]
    monkeypatch.setattr(module.pdfplumber, "open", lambda _: FakePDF([page]))

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    extractor = FastTextExtractor()
    doc = extractor.extract(pdf_path)

    assert doc.pages[0].confidence == "low_confidence"
    assert doc.pages[0].needs_layout_escalation is True
    assert doc.pages[0].image_area_ratio > 0.3
