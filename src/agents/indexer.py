"""
PageIndexBuilder
----------------
Builds a hierarchical page/section index to support navigation and pre-vector
retrieval. Operates on ExtractedDocument and persists JSON to
.refinery/pageindex/{doc_id}.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List
import subprocess

from pydantic import BaseModel, Field, ConfigDict

from src.models.extracted import ExtractedDocument, PageExtractionResult


class SectionNode(BaseModel):
    title: str
    page_start: int
    page_end: int
    child_sections: List["SectionNode"] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    summary: str = ""
    data_types_present: List[str] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)


SectionNode.model_rebuild()


class PageIndex(BaseModel):
    doc_id: str
    sections: List[SectionNode]


class PageIndexBuilder:
    def __init__(self, output_dir: Path | str = ".refinery/pageindex"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, doc: ExtractedDocument) -> PageIndex:
        sections: List[SectionNode] = []
        for page in doc.pages:
            page_node = self._build_page_node(page)
            sections.append(page_node)
        index = PageIndex(doc_id=doc.doc_id, sections=sections)
        self._save(index)
        return index

    # Navigation helpers ---------------------------------------------------
    def topic_traversal(self, index: PageIndex) -> List[str]:
        titles: List[str] = []
        def walk(nodes: List[SectionNode]):
            for n in nodes:
                titles.append(n.title)
                walk(n.child_sections)
        walk(index.sections)
        return titles

    def top_sections(self, index: PageIndex, query: str, n: int = 3) -> List[SectionNode]:
        scored = []
        q = query.lower()
        for sec in index.sections:
            score = sec.title.lower().count(q) + sec.summary.lower().count(q)
            scored.append((score, sec))
            for child in sec.child_sections:
                score_child = child.title.lower().count(q) + child.summary.lower().count(q)
                scored.append((score_child, child))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:n] if _ > 0] or [sec for _, sec in scored[:n]]

    # Internal helpers ------------------------------------------------------
    def _build_page_node(self, page: PageExtractionResult) -> SectionNode:
        child_sections = self._detect_child_sections(page)
        key_entities = self._extract_entities(page.full_text)
        data_types = self._detect_data_types(page)
        summary = self._llm_summarize(page.full_text)
        title = f"Page {page.page_number}"
        return SectionNode(
            title=title,
            page_start=page.page_number,
            page_end=page.page_number,
            child_sections=child_sections,
            key_entities=key_entities,
            summary=summary,
            data_types_present=data_types,
        )

    def _detect_child_sections(self, page: PageExtractionResult) -> List[SectionNode]:
        sections: List[SectionNode] = []
        lines = (page.full_text or "").splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.isupper() or stripped.endswith(":"):
                sections.append(
                    SectionNode(
                        title=stripped.rstrip(":"),
                        page_start=page.page_number,
                        page_end=page.page_number,
                        child_sections=[],
                        key_entities=self._extract_entities(stripped),
                        summary=self._llm_summarize(stripped),
                        data_types_present=self._detect_data_types(page),
                    )
                )
        return sections

    def _extract_entities(self, text: str) -> List[str]:
        # Very lightweight entity heuristic: capitalized words 4+ letters
        candidates = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text or "")
        # de-duplicate preserving order
        seen = set()
        entities = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                entities.append(c)
        return entities[:10]

    def _detect_data_types(self, page: PageExtractionResult) -> List[str]:
        types = set()
        if page.tables:
            types.add("table")
        if page.figures:
            types.add("figure")
        if any(ch.isdigit() for ch in page.full_text or ""):
            types.add("numeric")
        if "=" in (page.full_text or ""):
            types.add("equation")
        if page.full_text:
            types.add("text")
        return sorted(types)

    def _summarize(self, text: str, max_words: int = 40) -> str:
        words = (text or "").split()
        return " ".join(words[:max_words])

    def _llm_summarize(self, text: str) -> str:
        """Attempt a fast LLM summary via Ollama gemini-flash, fall back to heuristic."""
        if not text:
            return ""
        try:
            import shutil
            import os

            if shutil.which("ollama") and os.getenv("ENABLE_OLLAMA", "0") == "1":
                prompt = f"Summarize briefly: {text[:500]}"
                result = subprocess.run(
                    ["ollama", "run", "gemini-3-flash-preview", prompt],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip().splitlines()[0][:400]
        except Exception:
            pass
        return self._summarize(text)

    def _save(self, index: PageIndex) -> None:
        path = self.output_dir / f"{index.doc_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(index.model_dump(), f, ensure_ascii=False, indent=2)


__all__ = ["PageIndexBuilder", "PageIndex", "SectionNode"]
