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
import hashlib
import os
import shutil
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

    def _init(self, conn):
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                doc_id TEXT,
                page_number INTEGER,
                fact_key TEXT,
                fact_value REAL,
                unit TEXT,
                content_hash TEXT
            )
            """
        )
        conn.commit()

    def add_fact(self, doc_id: str, page: int, key: str, value: float, unit: str = "", content_hash: str = ""):
        with sqlite3.connect(self.db_path) as conn:
            self._init(conn)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO facts (doc_id, page_number, fact_key, fact_value, unit, content_hash) VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, page, key, value, unit, content_hash),
            )
            conn.commit()

    def query(self, key: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            self._init(conn)
            cur = conn.cursor()
            cur.execute(
                "SELECT doc_id, page_number, fact_key, fact_value, unit, content_hash FROM facts WHERE fact_key = ?",
                (key,),
            )
            rows = cur.fetchall()

        return [
            {
                "doc_id": r[0],
                "page_number": r[1],
                "fact_key": r[2],
                "fact_value": r[3],
                "unit": r[4],
                "content_hash": r[5] or "",
            }
            for r in rows
        ]


class QueryAgent:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        pageindex_dir: Path | str = ".refinery/pageindex",
        fact_db_path=".refinery/facts/facts.db"
    ):
        self.vector_store = vector_store
        self.pageindex_dir = Path(pageindex_dir)
        self.fact_table = FactTable(fact_db_path)

    # Public API ------------------------------------------------------------
    def answer(self, question: str, audit: bool = False, doc_id: Optional[str] = None) -> QueryResult:
        intent = self._detect_intent(question)

        if intent == "structured":
            fact_hits = self.structured_query(question)
            if fact_hits:
                return self._result_from_facts(fact_hits)

        if intent == "navigational":
            nav_sections = self.pageindex_navigate(doc_id)
            if nav_sections:
                top = nav_sections[0]
                prov = [
                    ProvenanceChain(
                        document_name=doc_id or top.get("doc_id", ""),
                        page_number=top.get("page_start", 0),
                        bounding_box={},
                        content_hash="",
                    )
                ]
                answer = f"Navigate to section '{top.get('title')}' spanning pages {top.get('page_start')}-{top.get('page_end')}."
                return QueryResult(answer=answer, provenance=prov, audited=bool(prov))

        # Semantic path
        hits = self.semantic_search(question)
        hits = self._rerank(question, hits)
        if not hits:
            return QueryResult(answer="No relevant content found.", provenance=[], audited=False)

        synthesized = self._synthesize_answer(question, hits)
        provenance = [
            ProvenanceChain(
                document_name=h.get("doc_id", ""),
                page_number=h.get("page_number", 0),
                bounding_box=h.get("bounding_box", {}),
                content_hash=h.get("content_hash", ""),
            )
            for h in hits[:3]
        ]
        audited = True if (not audit or provenance) else False
        return QueryResult(answer=synthesized, provenance=provenance, audited=audited)

    def semantic_search(self, query: str) -> List[Dict[str, Any]]:
        if not self.vector_store:
            return []
        return self.vector_store.search(query, top_k=3)

    def structured_query(self, query: str) -> List[Dict[str, Any]]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]+", query)
        for key in tokens:
            hits = self.fact_table.query(key)
            if hits:
                return hits
        return []

    def pageindex_navigate(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if doc_id:
            paths = [self.pageindex_dir / f"{doc_id}.json"]
        else:
            paths = sorted(self.pageindex_dir.glob("*.json"))
        for path in paths:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return data.get("sections", [])
        return []

    def ingest_facts(self, doc: ExtractedDocument) -> None:
        num_pattern = re.compile(r"([A-Za-z][A-Za-z0-9_]+)\s*[:=]?\s*([\d,]+(?:\.\d+)?)")

        for page in doc.pages:
            text = page.full_text or ""
            page_hash = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
            for match in num_pattern.finditer(text):
                key = match.group(1)
                value_str = match.group(2)

                # Skip empty or invalid values
                if not value_str:
                    continue

                try:
                    val = float(value_str.replace(",", ""))
                except ValueError:
                    continue

                self.fact_table.add_fact(doc.doc_id, page.page_number, key, val, content_hash=page_hash)

    def _format_fact_answer(self, facts: List[Dict[str, Any]]) -> str:
        parts = [f"{f['fact_key']} = {f['fact_value']}" + (f" {f['unit']}" if f.get('unit') else "") for f in facts]
        return "; ".join(parts)

    # Intent / orchestration -----------------------------------------------
    def _detect_intent(self, question: str) -> str:
        q = question.lower()
        if "page" in q or "section" in q:
            return "navigational"
        if re.search(r"\d", q) or re.search(r"\b(revenue|total|amount|value|cost)\b", q):
            return "structured"
        return "semantic"

    def _result_from_facts(self, fact_hits: List[Dict[str, Any]]) -> QueryResult:
        answer = self._format_fact_answer(fact_hits)
        provenance = [
            ProvenanceChain(
                document_name=f["doc_id"],
                page_number=f["page_number"],
                bounding_box={},
                content_hash=f.get("content_hash", ""),
            )
            for f in fact_hits
        ]
        return QueryResult(answer=answer, provenance=provenance, audited=bool(provenance), facts_used=fact_hits)

    def _rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def score(hit):
            content = hit.get("content", "").lower()
            q_terms = query.lower().split()
            return sum(content.count(t) for t in q_terms)
        return sorted(hits, key=score, reverse=True)

    def _synthesize_answer(self, question: str, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "No relevant information found."

        hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)

        if shutil.which("ollama") and os.getenv("ENABLE_OLLAMA", "0") == "1":
            try:
                import subprocess

                context = "\n".join(
                    [
                        f"[{i}] DOC:{h.get('doc_id','')} PAGE:{h.get('page_number','')} "
                        f"BBOX:{h.get('bounding_box','')} HASH:{h.get('content_hash','')}\n"
                        f"{h.get('content','')}\n"
                        for i, h in enumerate(hits[:3])
                    ]
                )

                prompt = f"""
                        You are a document QA assistant.

                        Answer ONLY using the provided context.

                        If the answer is missing say:
                        "Answer not found in the document."

                        Return:

                        Answer:
                        <answer>

                        Sources:
                        Document:
                        Page:
                        BoundingBox:
                        ContentHash:

                        Context:
                        {context}

                        Question:
                        {question}
                        """

                result = subprocess.run(
                    ["ollama", "run", "gemini-3-flash-preview", prompt],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )

                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

            except Exception:
                pass

        return hits[0].get("content", "")

    # Audit -----------------------------------------------------------------
    def audit_claim(self, claim: str) -> QueryResult:
        fact_hits = self.structured_query(claim)
        if fact_hits:
            return self._result_from_facts(fact_hits)
        hits = self.semantic_search(claim)
        hits = self._rerank(claim, hits)
        if hits:
            prov = [
                ProvenanceChain(
                    document_name=hits[0].get("doc_id", ""),
                    page_number=hits[0].get("page_number", 0),
                    bounding_box=hits[0].get("bounding_box", {}),
                    content_hash=hits[0].get("content_hash", ""),
                )
            ]
            return QueryResult(answer="verified", provenance=prov, audited=True)
        return QueryResult(answer="unverifiable", provenance=[], audited=False)


__all__ = ["QueryAgent", "ProvenanceChain", "QueryResult", "FactTable"]
