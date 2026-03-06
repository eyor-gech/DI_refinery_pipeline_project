"""
Query Agent
-----------
LangGraph-style query orchestrator that can:
- navigate page indexes
- run semantic search over a vector store
- answer structured numeric queries using a FactTable (SQLite)
- return answers with provenance chains
- support audit mode to verify claims against citations
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

from pydantic import BaseModel, Field

from src.models.extracted import ExtractedDocument


class ProvenanceChain(BaseModel):
    document_name: str
    page_number: int
    bounding_box: Dict[str, float]
    content_hash: str


class QueryResult(BaseModel):
    answer: str
    provenance: List[ProvenanceChain]
    audited: bool = False
    facts_used: List[Dict[str, Any]] = Field(default_factory=list)


class VectorStore(Protocol):
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        ...


@dataclass
class FactTable:
    db_path: str = ":memory:"

    def __post_init__(self):
        self.conn = sqlite3.connect(self.db_path)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                doc_id TEXT,
                page_number INTEGER,
                fact_key TEXT,
                fact_value REAL,
                unit TEXT
            )
            """
        )
        self.conn.commit()

    def add_fact(self, doc_id: str, page: int, key: str, value: float, unit: str = ""):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO facts (doc_id, page_number, fact_key, fact_value, unit) VALUES (?, ?, ?, ?, ?)",
            (doc_id, page, key, value, unit),
        )
        self.conn.commit()

    def query(self, key: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT doc_id, page_number, fact_key, fact_value, unit FROM facts WHERE fact_key = ?", (key,))
        rows = cur.fetchall()
        return [
            {
                "doc_id": r[0],
                "page_number": r[1],
                "fact_key": r[2],
                "fact_value": r[3],
                "unit": r[4],
            }
            for r in rows
        ]


class QueryAgent:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        pageindex_dir: Path | str = ".refinery/pageindex",
        fact_db_path: str = ":memory:",
    ):
        self.vector_store = vector_store
        self.pageindex_dir = Path(pageindex_dir)
        self.fact_table = FactTable(fact_db_path)

    # Public API ------------------------------------------------------------
    def answer(self, question: str, audit: bool = False) -> QueryResult:
        # Try fact lookup first for numeric queries
        fact_hits = self.structured_query(question)
        if fact_hits:
            answer = self._format_fact_answer(fact_hits)
            provenance = [
                ProvenanceChain(
                    document_name=f["doc_id"],
                    page_number=f["page_number"],
                    bounding_box={},
                    content_hash="",
                )
                for f in fact_hits
            ]
            return QueryResult(answer=answer, provenance=provenance, audited=bool(provenance), facts_used=fact_hits)

        search_hits = self.semantic_search(question)
        if not search_hits:
            return QueryResult(answer="No relevant content found.", provenance=[], audited=False)

        top = search_hits[0]
        prov = [
            ProvenanceChain(
                document_name=top.get("doc_id", ""),
                page_number=top.get("page_number", 0),
                bounding_box=top.get("bounding_box", {}),
                content_hash=top.get("content_hash", ""),
            )
        ]
        audited = True if (not audit or prov) else False
        return QueryResult(answer=top.get("content", ""), provenance=prov, audited=audited)

    def semantic_search(self, query: str) -> List[Dict[str, Any]]:
        if not self.vector_store:
            return []
        return self.vector_store.search(query, top_k=3)

    def structured_query(self, query: str) -> List[Dict[str, Any]]:
        # naive key extraction: last word
        key = query.strip().split()[-1].strip("?.")
        return self.fact_table.query(key)

    def pageindex_navigate(self, doc_id: str) -> List[Dict[str, Any]]:
        path = self.pageindex_dir / f"{doc_id}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("sections", [])

    def ingest_facts(self, doc: ExtractedDocument) -> None:
        num_pattern = re.compile(r"([A-Za-z][A-Za-z0-9_]+)\s*[:=]?\s*([\d,]+(?:\.\d+)?)")
        for page in doc.pages:
            for match in num_pattern.finditer(page.full_text or ""):
                key = match.group(1)
                val = float(match.group(2).replace(",", ""))
                self.fact_table.add_fact(doc.doc_id, page.page_number, key, val)

    def _format_fact_answer(self, facts: List[Dict[str, Any]]) -> str:
        parts = [f"{f['fact_key']} = {f['fact_value']}" + (f" {f['unit']}" if f.get('unit') else "") for f in facts]
        return "; ".join(parts)


__all__ = ["QueryAgent", "ProvenanceChain", "QueryResult", "FactTable"]
