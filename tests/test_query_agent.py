import json
from pathlib import Path

from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent, ProvenanceChain
from src.models.extracted import (
    BoundingBox,
    ExtractedDocument,
    PageExtractionResult,
    TextBlock,
)


class DummyVectorStore:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k=3):
        return self.hits[:top_k]


def make_doc():
    bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10)
    tb = TextBlock(text="Revenue 1234", bbox=bbox, reading_order=0)
    page = PageExtractionResult(
        page_number=1,
        text_blocks=[tb],
        full_text="Revenue 1234",
        confidence="high_confidence",
        needs_layout_escalation=False,
        character_count=10,
        character_density=0.1,
        image_area_ratio=0.0,
        font_metadata_presence=True,
        extraction_time_ms=1.0,
        tables=[],
        figures=[],
        strategy_used="fast_text",
        escalated=False,
        flagged_for_review=False,
        cost_estimate=0.0,
    )
    return ExtractedDocument(doc_id="doc", extractor="fast_text", pages=[page], total_time_ms=1.0)


def test_semantic_search_with_provenance(tmp_path):
    hits = [
        {
            "content": "Answer content",
            "doc_id": "doc",
            "page_number": 1,
            "bounding_box": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            "content_hash": "abc",
        }
    ]
    agent = QueryAgent(vector_store=DummyVectorStore(hits), pageindex_dir=tmp_path)
    result = agent.answer("question")
    assert result.answer == "Answer content"
    assert isinstance(result.provenance[0], ProvenanceChain)
    assert result.provenance[0].content_hash == "abc"


def test_audit_mode_requires_provenance(tmp_path):
    agent = QueryAgent(vector_store=DummyVectorStore([]), pageindex_dir=tmp_path)
    result = agent.answer("question", audit=True)
    assert result.audited is False


def test_fact_table_query(tmp_path):
    doc = make_doc()
    agent = QueryAgent(vector_store=DummyVectorStore([]), pageindex_dir=tmp_path, fact_db_path=str(tmp_path / "facts.db"))
    agent.ingest_facts(doc)
    result = agent.answer("What is Revenue?")
    assert "Revenue" in result.answer
    assert result.provenance[0].document_name == "doc"


def test_pageindex_navigation(tmp_path):
    builder = PageIndexBuilder(output_dir=tmp_path)
    index = builder.build_index(make_doc())
    agent = QueryAgent(vector_store=DummyVectorStore([]), pageindex_dir=tmp_path)
    sections = agent.pageindex_navigate("doc")
    assert sections
    assert sections[0]["title"] == "Page 1"
