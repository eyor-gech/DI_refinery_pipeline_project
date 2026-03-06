from pathlib import Path

import json

from src.agents.extractor import ExtractionRouter
from src.agents.triage import DocumentProfile
from src.models.extracted import BoundingBox, ExtractedDocument, PageExtractionResult, TextBlock


def make_page(page_number: int, confidence: str, text: str = "text") -> PageExtractionResult:
    bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10)
    block = TextBlock(text=text, bbox=bbox, reading_order=0)
    return PageExtractionResult(
        page_number=page_number,
        text_blocks=[block],
        full_text=text,
        confidence=confidence,
        needs_layout_escalation=False,
        character_count=len(text),
        character_density=0.001,
        image_area_ratio=0.0,
        font_metadata_presence=True,
        extraction_time_ms=5.0,
        tables=[],
        figures=[],
    )


class FakeExtractor:
    def __init__(self, name: str, page_confidences):
        self.name = name
        self.page_confidences = page_confidences

    def extract(self, pdf_path):
        pages = [make_page(i + 1, conf) for i, conf in enumerate(self.page_confidences)]
        for p in pages:
            p.strategy_used = self.name
        return ExtractedDocument(
            doc_id=Path(pdf_path).stem, extractor=self.name, pages=pages, total_time_ms=10.0
        )


def test_single_level_routing(monkeypatch, tmp_path):
    strategies = {
        "fast_text": FakeExtractor("fast_text", ["high_confidence"]),
        "layout": FakeExtractor("layout", ["high_confidence"]),
        "vision": FakeExtractor("vision", ["high_confidence"]),
    }
    router = ExtractionRouter(
        strategies=strategies,
        thresholds={"fast_text": 0.7},
        ledger_path=tmp_path / "ledger.jsonl",
    )

    profile = DocumentProfile(
        doc_id="doc",
        origin_type="native_digital",
        layout_complexity="single_column",
        language="en",
        language_confidence=1.0,
        domain_hint="general",
        estimated_extraction_cost="fast_text_sufficient",
        character_count=0,
        character_density=0.0,
        whitespace_ratio=0.0,
        image_area_ratio=0.0,
        font_metadata_presence=True,
    )

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"")

    result = router.route(pdf_path, profile)

    assert result.pages[0].strategy_used == "fast_text"
    assert result.pages[0].escalated is False
    ledger_lines = (tmp_path / "ledger.jsonl").read_text().strip().splitlines()
    assert len(ledger_lines) == 1
    entry = json.loads(ledger_lines[0])
    assert entry["strategy_used"] == "fast_text"
    assert entry["escalation_occurred"] is False


def test_multi_level_escalation(tmp_path):
    strategies = {
        "layout": FakeExtractor("layout", ["low_confidence"]),
        "vision": FakeExtractor("vision", ["high_confidence"]),
    }
    router = ExtractionRouter(
        strategies=strategies,
        thresholds={"fast_text": 0.7, "layout": 0.7},
        ledger_path=tmp_path / "ledger.jsonl",
    )
    profile = DocumentProfile(
        doc_id="doc",
        origin_type="native_digital",
        layout_complexity="multi_column",
        language="en",
        language_confidence=1.0,
        domain_hint="general",
        estimated_extraction_cost="needs_layout_model",
        character_count=0,
        character_density=0.0,
        whitespace_ratio=0.0,
        image_area_ratio=0.0,
        font_metadata_presence=True,
    )
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"")

    result = router.route(pdf_path, profile)

    assert result.pages[0].strategy_used == "vision"
    assert result.pages[0].escalated is True
    ledger_lines = (tmp_path / "ledger.jsonl").read_text().strip().splitlines()
    entry = json.loads(ledger_lines[0])
    assert entry["escalation_occurred"] is True


def test_low_confidence_final_strategy_flags_review(tmp_path):
    strategies = {"vision": FakeExtractor("vision", ["low_confidence"])}
    router = ExtractionRouter(
        strategies=strategies,
        thresholds={"vision": 0.8},
        ledger_path=tmp_path / "ledger.jsonl",
    )
    profile = DocumentProfile(
        doc_id="doc",
        origin_type="scanned_image",
        layout_complexity="single_column",
        language="en",
        language_confidence=1.0,
        domain_hint="general",
        estimated_extraction_cost="needs_vision_model",
        character_count=0,
        character_density=0.0,
        whitespace_ratio=0.0,
        image_area_ratio=0.0,
        font_metadata_presence=False,
    )
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"")

    result = router.route(pdf_path, profile)

    assert result.pages[0].flagged_for_review is True
    ledger_lines = (tmp_path / "ledger.jsonl").read_text().strip().splitlines()
    entry = json.loads(ledger_lines[0])
    assert entry["flagged_for_review"] is True
