import sys
from pathlib import Path

import pytest

from src.strategies.VisionExtractor import CostTracker, VisionExtractor


class FakeImage:
    def save(self, buf, format=None):
        buf.write(b"fakepng")


class FakeRendered:
    @property
    def original(self):
        return FakeImage()


class FakePage:
    def __init__(self, width=612, height=792, images=None):
        self.width = width
        self.height = height
        self.images = images or []
        self.chars = []
        self._text = ""

    def to_image(self, resolution=200):
        return FakeRendered()


class FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_ocr_only_page(monkeypatch, tmp_path):
    page = FakePage()
    module = sys.modules[VisionExtractor.__module__]
    monkeypatch.setattr(module.pdfplumber, "open", lambda _: FakePDF([page]))
    monkeypatch.setattr(module.pytesseract, "image_to_string", lambda *args, **kwargs: "Hello OCR text" * 10)

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    extractor = VisionExtractor(cost_cap_usd=0.01)
    doc = extractor.extract(pdf_path)

    assert doc.extractor == "vision"
    assert doc.pages[0].confidence == "high_confidence"
    assert "Hello OCR" in doc.pages[0].full_text


def test_vlm_fallback_when_ocr_weak(monkeypatch, tmp_path):
    page = FakePage(images=[{"width": 600, "height": 700}])
    module = sys.modules[VisionExtractor.__module__]
    monkeypatch.setattr(module.pdfplumber, "open", lambda _: FakePDF([page]))
    monkeypatch.setattr(module.pytesseract, "image_to_string", lambda *args, **kwargs: "")

    # Force VLM call
    def fake_call_vlm(image_b64, model):
        return ("VLM transcription text", 500)

    monkeypatch.setattr(module.VisionExtractor, "_call_vlm", staticmethod(fake_call_vlm))

    pdf_path = tmp_path / "vlm.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    extractor = VisionExtractor(cost_cap_usd=1.0)
    doc = extractor.extract(pdf_path)

    assert "VLM transcription" in doc.pages[0].full_text
    assert doc.pages[0].confidence == "high_confidence"


def test_budget_guard_blocks_expensive_calls(monkeypatch, tmp_path):
    page = FakePage()
    module = sys.modules[VisionExtractor.__module__]
    monkeypatch.setattr(module.pdfplumber, "open", lambda _: FakePDF([page]))
    monkeypatch.setattr(module.pytesseract, "image_to_string", lambda *args, **kwargs: "")

    # Expensive model
    def fake_call_vlm(image_b64, model):
        return ("Should not be used", 10_000_000)

    monkeypatch.setattr(module.VisionExtractor, "_call_vlm", staticmethod(fake_call_vlm))

    pdf_path = tmp_path / "budget.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    extractor = VisionExtractor(cost_cap_usd=0.0001, vlm_models={"expensive": 0.00001})
    doc = extractor.extract(pdf_path)

    # With tiny cap, fallback should be skipped
    assert doc.pages[0].full_text == ""
    assert doc.pages[0].confidence == "low_confidence"


def test_mixed_pages_track_tokens(monkeypatch, tmp_path):
    pages = [FakePage(), FakePage()]
    module = sys.modules[VisionExtractor.__module__]
    monkeypatch.setattr(module.pdfplumber, "open", lambda _: FakePDF(pages))

    # First page OCR succeeds, second fails -> VLM
    ocr_calls = {"count": 0}

    def fake_image_to_string(*args, **kwargs):
        ocr_calls["count"] += 1
        return "text" * 20 if ocr_calls["count"] == 1 else ""

    monkeypatch.setattr(module.pytesseract, "image_to_string", fake_image_to_string)

    def fake_call_vlm(image_b64, model):
        return ("fallback text", 1200)

    monkeypatch.setattr(module.VisionExtractor, "_call_vlm", staticmethod(fake_call_vlm))

    pdf_path = tmp_path / "mixed.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    extractor = VisionExtractor(cost_cap_usd=1.0)
    doc = extractor.extract(pdf_path)

    assert doc.pages[0].full_text.startswith("text")
    assert doc.pages[1].full_text.startswith("fallback")
