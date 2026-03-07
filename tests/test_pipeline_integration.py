import json
from pathlib import Path

from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent, ProvenanceChain
from src.models.extracted import (
    BoundingBox,
    ExtractedDocument,
    FigureBlock,
    PageExtractionResult,
    TableBlock,
    TextBlock,
)
from run_pipeline import ingest_vector_store, persist_chunks, write_full_text


def make_extracted_doc():
    bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10)
    text_blocks = [
        TextBlock(text="HEADER:", bbox=bbox, reading_order=0),
        TextBlock(text="1. Item one", bbox=bbox, reading_order=1),
        TextBlock(text="See xref: Table 1", bbox=bbox, reading_order=2),
    ]
    table = TableBlock(bbox=bbox, rows=[["H1", "H2"], ["A", "B"]])
    figure = FigureBlock(bbox=bbox, caption="Figure caption text")
    page = PageExtractionResult(
        page_number=1,
        text_blocks=text_blocks,
        full_text="HEADER:\n1. Item one\nSee xref: Table 1",
        confidence="high_confidence",
        needs_layout_escalation=False,
        character_count=10,
        character_density=0.1,
        image_area_ratio=0.0,
        font_metadata_presence=True,
        extraction_time_ms=1.0,
        tables=[table],
        figures=[figure],
        strategy_used="fast_text",
        escalated=False,
        flagged_for_review=False,
        cost_estimate=0.0,
    )
    return ExtractedDocument(doc_id="doc", extractor="fast_text", pages=[page], total_time_ms=1.0)


def test_full_pipeline_execution(tmp_path):
    doc = make_extracted_doc()

    # chunking
    chunker = ChunkingEngine(token_limit=50)
    chunks = chunker.chunk(doc)
    assert chunks

    base_dir = tmp_path / ".refinery"
    write_full_text(doc, base_dir / "fulltext")
    persist_chunks(chunks, base_dir / "chunks", doc.doc_id)
    ingest_vector_store(chunks, base_dir / "vector_store", doc.doc_id)

    # indexing
    indexer = PageIndexBuilder(output_dir=base_dir / "pageindex")
    index = indexer.build_index(doc)
    assert (base_dir / "pageindex" / "doc.json").exists()

    # fact table
    qa = QueryAgent(pageindex_dir=base_dir / "pageindex", fact_db_path=str(base_dir / "facts.db"))
    qa.ingest_facts(doc)
    assert (base_dir / "facts.db").exists()

    # vector payload exists
    assert (base_dir / "vector_store" / "doc.json").exists()


class DummyVectorStore:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k=3):
        return self.hits[:top_k]


def test_query_with_provenance(tmp_path):
    hits = [
        {
            "content": "Answer content",
            "doc_id": "doc",
            "page_number": 1,
            "bounding_box": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            "content_hash": "abc",
        }
    ]
    agent = QueryAgent(vector_store=DummyVectorStore(hits), pageindex_dir=tmp_path, fact_db_path=str(tmp_path / "facts.db"))
    result = agent.answer("What is the answer?", audit=True)
    assert result.answer
    assert result.provenance
    assert isinstance(result.provenance[0], ProvenanceChain)


def test_audit_claim_verification(tmp_path):
    agent = QueryAgent(vector_store=DummyVectorStore([]), pageindex_dir=tmp_path, fact_db_path=str(tmp_path / "facts.db"))
    agent.fact_table.add_fact("doc", 1, "Revenue", 4.2, content_hash="hash123")

    result = agent.audit_claim("Revenue was 4.2")
    assert result.audited is True
    assert result.provenance
