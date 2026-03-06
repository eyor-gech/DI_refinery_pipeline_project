import json
from pathlib import Path

from src.agents.indexer import PageIndexBuilder
from src.models.extracted import (
    BoundingBox,
    ExtractedDocument,
    FigureBlock,
    PageExtractionResult,
    TableBlock,
    TextBlock,
)


def build_doc():
    bbox = BoundingBox(x0=0, y0=0, x1=100, y1=100)
    text_blocks = [
        TextBlock(text="INTRODUCTION", bbox=bbox, reading_order=0),
        TextBlock(text="This document describes Refinery Pipeline", bbox=bbox, reading_order=1),
    ]
    page = PageExtractionResult(
        page_number=1,
        text_blocks=text_blocks,
        full_text="INTRODUCTION\nThis document describes Refinery Pipeline 2026.",
        confidence="high_confidence",
        needs_layout_escalation=False,
        character_count=10,
        character_density=0.1,
        image_area_ratio=0.0,
        font_metadata_presence=True,
        extraction_time_ms=1.0,
        tables=[TableBlock(bbox=bbox, rows=[["H1", "H2"], ["A", "B"]])],
        figures=[FigureBlock(bbox=bbox, caption="Architecture diagram")],
    )
    return ExtractedDocument(doc_id="doc", extractor="fast_text", pages=[page], total_time_ms=1.0)


def test_pageindex_builder_creates_file(tmp_path):
    builder = PageIndexBuilder(output_dir=tmp_path)
    doc = build_doc()

    index = builder.build_index(doc)

    saved = tmp_path / "doc.json"
    assert saved.exists()
    data = json.loads(saved.read_text())
    assert data["doc_id"] == "doc"
    assert data["sections"][0]["title"] == "Page 1"
    assert "table" in data["sections"][0]["data_types_present"]


def test_child_sections_and_summary(tmp_path):
    builder = PageIndexBuilder(output_dir=tmp_path)
    doc = build_doc()
    index = builder.build_index(doc)

    page_section = index.sections[0]
    assert page_section.child_sections
    child = page_section.child_sections[0]
    assert child.title == "INTRODUCTION"
    assert child.summary.startswith("INTRODUCTION")
