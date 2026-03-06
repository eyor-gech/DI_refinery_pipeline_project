import sys
from pathlib import Path

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from strategies.LayoutExtractor import DoclingDocumentAdapter, LayoutExtractor


def test_docling_adapter_to_extracteddocument():
    adapter = DoclingDocumentAdapter()
    fake_doc = {
        "pages": [
            {
                "number": 1,
                "size": (612, 792),
                "blocks": [
                    {"text": "Column 1 text", "bbox": [0, 0, 200, 100], "reading_order": 0},
                    {"text": "Column 2 text", "bbox": [300, 0, 500, 100], "reading_order": 1},
                ],
                "tables": [{"bbox": [50, 150, 300, 300], "rows": [["h1", "h2"], ["v1", "v2"]], "caption": "Table 1"}],
                "figures": [{"bbox": [100, 320, 250, 420], "caption": "Figure 1"}],
                "elapsed_ms": 5.0,
            }
        ]
    }

    doc = adapter.adapt(fake_doc, pdf_path="sample.pdf", total_time_ms=5.0)

    assert doc.extractor == "layout"
    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert len(page.text_blocks) == 2
    assert len(page.tables) == 1
    assert page.tables[0].rows[1][0] == "v1"
    assert len(page.figures) == 1
    assert page.confidence in {"high_confidence", "medium_confidence"}


class FakeTable:
    def __init__(self, bbox, rows):
        self.bbox = bbox
        self._rows = rows

    def extract(self):
        return self._rows


class FakePage:
    def __init__(self, text, width=612, height=792, words=None, images=None, tables=None, chars=None):
        self.width = width
        self.height = height
        self._text = text
        self._words = words or []
        self.images = images or []
        self._tables = tables or []
        self.chars = chars or []

    def extract_text(self):
        return self._text

    def extract_words(self):
        return self._words

    def find_tables(self):
        return self._tables


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_layout_extractor_fallback_pdfplumber(monkeypatch, tmp_path):
    words = [{"text": "multi", "x0": 10, "x1": 30, "top": 700, "bottom": 720}]
    tables = [FakeTable([0, 0, 100, 100], [["a", "b"]])]
    page = FakePage(
        text="Some text",
        words=words,
        images=[{"width": 300, "height": 300}],
        tables=tables,
        chars=[{"fontname": "Helvetica"}],
    )

    module = sys.modules[LayoutExtractor.__module__]
    monkeypatch.setattr(module.LayoutExtractor, "_run_docling", lambda self, path: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(module.pdfplumber, "open", lambda _: FakePDF([page]))

    pdf_path = tmp_path / "complex.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    extractor = LayoutExtractor()
    doc = extractor.extract(pdf_path)

    assert doc.extractor == "layout"
    assert len(doc.pages) == 1
    page_out = doc.pages[0]
    assert page_out.full_text.startswith("Some text") or page_out.full_text.startswith("multi")
    assert len(page_out.tables) == 1
    assert page_out.tables[0].rows[0][0] == "a"
