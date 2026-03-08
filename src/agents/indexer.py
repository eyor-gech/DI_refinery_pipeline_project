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
from typing import List, Iterable, Tuple, Optional
import subprocess
from dataclasses import dataclass

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

@dataclass
class HeadingCandidate:
    title: str
    level: int
    page: int


class PageIndexBuilder:
    def __init__(self, output_dir: Path | str = ".refinery/pageindex"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, doc: ExtractedDocument, output_path: Path | str | None = None) -> PageIndex:
        """
        Build a hierarchical section tree for a document and persist to JSON.
        output_path overrides the builder's default output directory when provided.
        """
        candidates = self._collect_heading_candidates(doc)
        sections = self._build_hierarchy(candidates, doc)

        if not sections:
            # Fallback: one section per page if no headings are detected.
            sections = [self._page_fallback_node(page) for page in doc.pages]

        self._populate_metadata(sections, doc)
        index = PageIndex(doc_id=doc.doc_id, sections=sections)
        self._save(index, output_path=output_path)
        return index

    # Navigation helpers ---------------------------------------------------
    def traverse(self, index: PageIndex, topic_or_query: str, top_k: int = 5, use_embeddings: bool = True) -> List[SectionNode]:
        """
        Return the most relevant sections for a topic or query.
        Prefers fast embedding search via fastembed; falls back to lexical scoring.
        """
        flat = list(self._flatten(index.sections))

        if use_embeddings:
            try:
                from src.utils.vector_store import SimpleVectorStore

                store = SimpleVectorStore()
                for node in flat:
                    store.add(f"{node.title}\n{node.summary}", {"node": node})
                results = store.search(topic_or_query, top_k=top_k)
                return [r["node"] for r in results]
            except Exception:
                # Embedding backend unavailable; fall back to lexical scoring.
                pass

        q = topic_or_query.lower()
        scored: List[Tuple[float, SectionNode]] = []
        for node in flat:
            text = f"{node.title} {node.summary} {' '.join(node.key_entities)}".lower()
            # simple token overlap scoring
            score = text.count(q) + sum(1 for token in q.split() if token in text)
            scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def topic_traversal(self, index: PageIndex) -> List[str]:
        """Compatibility helper: returns titles in traversal order."""
        return [n.title for n in self._flatten(index.sections)]

    # Internal helpers ------------------------------------------------------
    def _collect_heading_candidates(self, doc: ExtractedDocument) -> List[HeadingCandidate]:
        """
        Lightweight heading detector using numbering, uppercase, and delimiter heuristics.
        Produces ordered candidates across the entire document.
        """
        candidates: List[HeadingCandidate] = []
        heading_keywords = {"introduction", "scope", "summary", "conclusion", "abstract", "method", "methods", "results"}

        for page in doc.pages:
            for raw_line in (page.full_text or "").splitlines():
                line = raw_line.strip()
                if not line or len(line) > 140:
                    continue

                numeric_match = re.match(r"^(?P<num>\d+(?:\.\d+)*)(?:[\s\-–\)])+(?P<title>.+)$", line)
                if numeric_match:
                    level = numeric_match.group("num").count(".") + 1
                    title = numeric_match.group("title").strip(" :-")
                    candidates.append(HeadingCandidate(title=title, level=level, page=page.page_number))
                    continue

                if line.isupper() and len(line.split()) <= 10:
                    candidates.append(HeadingCandidate(title=line.title(), level=1, page=page.page_number))
                    continue

                if line.endswith(":") and len(line.split()) <= 12:
                    candidates.append(HeadingCandidate(title=line.rstrip(":"), level=2, page=page.page_number))
                    continue

                lowered = line.lower()
                if any(k in lowered for k in heading_keywords) and len(line.split()) <= 12:
                    candidates.append(HeadingCandidate(title=line.rstrip(":"), level=1, page=page.page_number))

        return candidates

    def _build_hierarchy(self, candidates: List[HeadingCandidate], doc: ExtractedDocument) -> List[SectionNode]:
        sections: List[SectionNode] = []
        stack: List[Tuple[SectionNode, int]] = []
        last_page = doc.pages[-1].page_number if doc.pages else 0

        for cand in candidates:
            while stack and cand.level <= stack[-1][1]:
                node, _ = stack.pop()
                close_page = max(node.page_start, cand.page - 1)
                node.page_end = max(node.page_end, close_page)

            new_node = self._blank_section(cand.title, cand.page)
            if stack:
                stack[-1][0].child_sections.append(new_node)
            else:
                sections.append(new_node)

            stack.append((new_node, cand.level))

        # Close any remaining open sections
        while stack:
            node, _ = stack.pop()
            node.page_end = max(node.page_end, last_page)

        return sections

    def _page_fallback_node(self, page: PageExtractionResult) -> SectionNode:
        return SectionNode(
            title=f"Page {page.page_number}",
            page_start=page.page_number,
            page_end=page.page_number,
            child_sections=[],
            key_entities=self._extract_entities(page.full_text),
            summary=self._fast_summarize(page.full_text, f"Page {page.page_number}"),
            data_types_present=self._detect_data_types(page),
        )

    def _blank_section(self, title: str, page_number: int) -> SectionNode:
        return SectionNode(
            title=title,
            page_start=page_number,
            page_end=page_number,
            child_sections=[],
            key_entities=[],
            summary="",
            data_types_present=[],
        )

    def _populate_metadata(self, sections: List[SectionNode], doc: ExtractedDocument) -> None:
        for sec in sections:
            self._populate_section(sec, doc)

    def _populate_section(self, node: SectionNode, doc: ExtractedDocument) -> None:
        # ensure children populated first to derive end page
        for child in node.child_sections:
            self._populate_section(child, doc)
            node.page_end = max(node.page_end, child.page_end)

        text = self._collect_text(doc, node.page_start, node.page_end)
        node.summary = self._fast_summarize(text, node.title)
        node.key_entities = self._extract_entities(text)
        node.data_types_present = self._aggregate_data_types(doc, node.page_start, node.page_end)

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

    def _aggregate_data_types(self, doc: ExtractedDocument, start_page: int, end_page: int) -> List[str]:
        types = set()
        for page in doc.pages:
            if start_page <= page.page_number <= end_page:
                types.update(self._detect_data_types(page))
        return sorted(types)

    def _fast_summarize(self, text: str, title: str, max_words: int = 60) -> str:
        """
        Cheap, section-aware summarization.
        Attempts a tiny local LLM when enabled; otherwise uses extractive first-sentences heuristic.
        """
        if not text:
            return ""

        llm_summary = self._try_local_llm(text, title)
        if llm_summary:
            return llm_summary

        # Extractive fallback: first two sentences capped at max_words
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        summary = " ".join(sentences[:2])
        words = summary.split()
        return " ".join(words[:max_words])

    def _try_local_llm(self, text: str, title: str) -> str:
        """
        Attempt a fast, cheap model via Ollama (e.g., gemini-3-flash-preview) when enabled.
        Guarded to avoid blocking if the model is unavailable.
        """
        try:
            import shutil
            import os

            if shutil.which("ollama") and os.getenv("ENABLE_OLLAMA", "0") == "1":
                prompt = (
                    f"Provide a concise summary (<=60 words) of the section titled '{title}'. "
                    f"Focus on main facts and entities only.\n\n{text[:1200]}"
                )
                result = subprocess.run(
                    ["ollama", "run", "gemini-3-flash-preview", prompt],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip().splitlines()[0][:400]
        except Exception:
            return ""
        return ""

    def _collect_text(self, doc: ExtractedDocument, start_page: int, end_page: int) -> str:
        pages_text = []
        for page in doc.pages:
            if start_page <= page.page_number <= end_page:
                pages_text.append(page.full_text or "")
        return "\n".join(pages_text)

    def _flatten(self, nodes: Iterable[SectionNode]) -> Iterable[SectionNode]:
        for n in nodes:
            yield n
            yield from self._flatten(n.child_sections)

    def _save(self, index: PageIndex, output_path: Optional[Path | str] = None) -> None:
        target_dir = Path(output_path) if output_path else self.output_dir
        target_dir = target_dir if isinstance(target_dir, Path) else Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{index.doc_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(index.model_dump(), f, ensure_ascii=False, indent=2)


__all__ = ["PageIndexBuilder", "PageIndex", "SectionNode"]
