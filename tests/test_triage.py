import json
from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


import pytest

from agents.triage import TriageAgent


class FakePage:
    def __init__(
        self,
        text: str,
        width: float = 612,
        height: float = 792,
        images=None,
        chars=None,
        words=None,
        annots=None,
        tables=False,
    ):
        self.width = width
        self.height = height
        self._text = text
        self.images = images or []
        self.chars = chars or []
        self._words = words or []
        self.annots = annots or []
        self._tables = tables

    def extract_text(self):
        return self._text

    def extract_words(self, **_):
        return self._words

    def find_tables(self):
        return ["table"] if self._tables else []


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_native_digital_detection(monkeypatch, tmp_path):
    page = FakePage(
        text="Hello world " * 500,
        images=[],
        chars=[{"fontname": "Helvetica"} for _ in range(20)],
        words=[{"x0": 50, "x1": 120} for _ in range(100)],
    )
    monkeypatch.setattr("agents.triage.pdfplumber.open", lambda _: FakePDF([page]))

    pdf_path = tmp_path / "digital.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    agent = TriageAgent(profiles_dir=tmp_path / "profiles")

    profile = agent.profile(pdf_path)

    assert profile.origin_type == "native_digital"
    assert profile.estimated_extraction_cost == "fast_text_sufficient"
    saved = tmp_path / "profiles" / "digital.json"
    assert saved.exists()
    data = json.loads(saved.read_text())
    assert data["origin_type"] == "native_digital"


def test_scanned_detection(monkeypatch, tmp_path):
    page = FakePage(
        text="",
        images=[{"width": 600, "height": 750}],
        chars=[],
        words=[],
    )
    monkeypatch.setattr("agents.triage.pdfplumber.open", lambda _: FakePDF([page]))

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    agent = TriageAgent(profiles_dir=tmp_path / "profiles")

    profile = agent.profile(pdf_path)

    assert profile.origin_type == "scanned_image"
    assert profile.estimated_extraction_cost == "needs_vision_model"
    assert profile.image_area_ratio > 0.3


def test_form_fillable(monkeypatch, tmp_path):
    page = FakePage(
        text="Form content with fields",
        images=[],
        chars=[{"fontname": "Helvetica"}],
        words=[{"x0": 60, "x1": 110}],
        annots=[{"Subtype": "/Widget"}],
    )
    monkeypatch.setattr("agents.triage.pdfplumber.open", lambda _: FakePDF([page]))

    pdf_path = tmp_path / "form.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    agent = TriageAgent(profiles_dir=tmp_path / "profiles")

    profile = agent.profile(pdf_path)

    assert profile.origin_type == "form_fillable"
    assert profile.layout_complexity == "single_column"
    assert profile.estimated_extraction_cost == "fast_text_sufficient"


def test_table_heavy_layout(monkeypatch, tmp_path):
    page = FakePage(
        text="Row1 Cell1 Row1 Cell2",
        images=[],
        chars=[{"fontname": "Helvetica"}],
        words=[{"x0": 40, "x1": 100}],
        tables=True,
    )
    monkeypatch.setattr("agents.triage.pdfplumber.open", lambda _: FakePDF([page]))

    pdf_path = tmp_path / "table.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    agent = TriageAgent(profiles_dir=tmp_path / "profiles")

    profile = agent.profile(pdf_path)

    assert profile.layout_complexity == "table_heavy"
    assert profile.estimated_extraction_cost == "needs_layout_model"
