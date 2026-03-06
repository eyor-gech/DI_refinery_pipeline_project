from src.agents.chunker import ChunkingEngine
from src.models.extracted import (
    BoundingBox,
    ExtractedDocument,
    FigureBlock,
    PageExtractionResult,
    TableBlock,
    TextBlock,
)


def make_page():
    bbox = BoundingBox(x0=0, y0=0, x1=100, y1=100)
    text_blocks = [
        TextBlock(text="SECTION INTRO", bbox=bbox, reading_order=0),
        TextBlock(text="1. Item one", bbox=bbox, reading_order=1),
        TextBlock(text="2. Item two", bbox=bbox, reading_order=2),
        TextBlock(text="See xref: Table 1", bbox=bbox, reading_order=3),
    ]
    table = TableBlock(bbox=bbox, rows=[["H1", "H2"], ["v1", "v2"]])
    figure = FigureBlock(bbox=bbox, caption="Figure caption text")
    page = PageExtractionResult(
        page_number=1,
        text_blocks=text_blocks,
        full_text="SECTION INTRO\n1. Item one\n2. Item two\nSee xref: Table 1",
        confidence="high_confidence",
        needs_layout_escalation=False,
        character_count=10,
        character_density=0.1,
        image_area_ratio=0.0,
        font_metadata_presence=True,
        extraction_time_ms=1.0,
        tables=[table],
        figures=[figure],
    )
    return page


def test_chunking_rules_enforced():
    doc = ExtractedDocument(doc_id="doc", extractor="fast_text", pages=[make_page()], total_time_ms=1.0)
    engine = ChunkingEngine(token_limit=5)  # small to force list split metadata

    chunks = engine.chunk(doc)

    # Table chunk present and intact
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert table_chunks and "table" in table_chunks[0].content

    # Figure caption attached
    figure_chunks = [c for c in chunks if c.chunk_type == "figure"]
    assert figure_chunks and "figure caption" in figure_chunks[0].content.lower()

    # Numbered list split flagged when over limit
    list_chunks = [c for c in chunks if c.chunk_type == "list"]
    assert list_chunks
    assert any(c.metadata.get("split") == "true" for c in list_chunks)

    # Section metadata carried
    assert any(c.metadata.get("parent_section") == "SECTION INTRO" for c in chunks if c.chunk_type == "list")

    # Cross references captured as relationships
    xref_chunks = [c for c in chunks if "xref:" in c.content.lower()]
    assert xref_chunks and xref_chunks[0].relationships
