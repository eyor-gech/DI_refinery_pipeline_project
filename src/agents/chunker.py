"""
Chunking Engine
---------------
Converts ExtractedDocument objects into LDUs (logical document units) with
chunk-type aware rules and validation.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.extracted import ExtractedDocument, PageExtractionResult, BoundingBox


class LDU(BaseModel):
    content: str
    chunk_type: str
    page_refs: List[int]
    bounding_box: Dict[str, float]
    parent_section: Optional[str] = None
    token_count: int
    content_hash: str
    relationships: List[str] = Field(default_factory=list)
    items: Optional[List[str]] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class ChunkValidator:
    """Validates chunking rules."""

    def __init__(self, token_limit: int):
        self.token_limit = token_limit

    def validate(self, chunks: List[LDU]) -> None:
        for chunk in chunks:
            # Rule 1: tables stay intact
            if chunk.chunk_type == "table":
                if "table" not in chunk.content.lower():
                    raise ValueError("Table chunk missing table content marker")
                if chunk.metadata.get("has_header") not in ("True", "true", True):
                    raise ValueError("Table chunk missing header binding")
            # Rule 2: figure captions attached
            if chunk.chunk_type == "figure":
                if "caption" not in chunk.content.lower() or chunk.metadata.get("caption_attached") != "true":
                    raise ValueError("Figure caption not attached to figure chunk")
            # Rule 3: numbered lists intact unless exceeding token limit
            if chunk.chunk_type == "list":
                if chunk.token_count > self.token_limit and chunk.metadata.get("split") != "true":
                    raise ValueError("Long list must be marked as split")
                if chunk.token_count <= self.token_limit and chunk.metadata.get("split") == "true":
                    raise ValueError("Short list should not be split")
            # Rule 4: section headers stored as metadata
            if chunk.parent_section:
                if chunk.metadata.get("parent_section") != chunk.parent_section:
                    raise ValueError("Parent section missing from metadata")
            # Rule 5: cross references stored as relationships (optional list)
            if "xref:" in chunk.content.lower():
                if not chunk.relationships:
                    raise ValueError("Cross references must be captured in relationships")


class ChunkingEngine:
    def __init__(self, token_limit: int = 400):
        self.token_limit = token_limit
        self.validator = ChunkValidator(token_limit=token_limit)

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        chunks: List[LDU] = []
        for page in doc.pages:
            chunks.extend(self._chunk_page(page))
        self.validator.validate(chunks)
        return chunks

    # Page-level processing -------------------------------------------------
    def _chunk_page(self, page: PageExtractionResult) -> List[LDU]:
        chunks: List[LDU] = []
        current_section = None

        # 1) tables
        for tbl in page.tables:
            content = "table: " + " | ".join([" ; ".join(row) for row in tbl.rows])
            chunks.append(
                self._make_ldu(
                    content,
                    "table",
                    page,
                    tbl.bbox,
                    current_section,
                    extra_metadata={"has_header": True, "table_rows": len(tbl.rows)},
                )
            )

        # 2) figures
        for fig in page.figures:
            caption = fig.caption or ""
            content = f"figure: caption: {caption}".strip()
            chunks.append(self._make_ldu(content, "figure", page, fig.bbox, current_section))

        # 3) text & lists
        lines = (page.full_text or "").splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if self._is_section_header(line):
                current_section = line
                idx += 1
                continue

            # numbered list detection
            if self._is_numbered_item(line):
                items = []
                while idx < len(lines) and self._is_numbered_item(lines[idx].strip()):
                    items.append(lines[idx].strip())
                    idx += 1
                chunks.extend(self._emit_list_chunks(items, page, current_section))
                continue

            # generic text chunk
            if line:
                bbox = self._aggregate_bbox(page)
                chunks.append(self._make_ldu(line, "text", page, bbox, current_section))
            idx += 1

        return chunks

    # Helpers --------------------------------------------------------------
    def _emit_list_chunks(self, items: List[str], page: PageExtractionResult, section: Optional[str]) -> List[LDU]:
        result: List[LDU] = []
        buffer: List[str] = []
        for item in items:
            tokens = len(" ".join(buffer + [item]).split())
            if tokens > self.token_limit and buffer:
                content = "\n".join(buffer)
                ldu = self._make_ldu(content, "list", page, self._aggregate_bbox(page), section, split=True, items=buffer)
                result.append(ldu)
                buffer = [item]
            else:
                buffer.append(item)
        if buffer:
            split_flag = len(" ".join(buffer).split()) > self.token_limit
            ldu = self._make_ldu(
                "\n".join(buffer), "list", page, self._aggregate_bbox(page), section, split=split_flag, items=buffer
            )
            result.append(ldu)
        return result

    def _make_ldu(
        self,
        content: str,
        chunk_type: str,
        page: PageExtractionResult,
        bbox_obj,
        section: Optional[str],
        split: bool = False,
        items: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, str]] = None,
    ) -> LDU:
        bbox = {
            "x0": float(getattr(bbox_obj, "x0", 0.0)),
            "y0": float(getattr(bbox_obj, "y0", 0.0)),
            "x1": float(getattr(bbox_obj, "x1", 0.0)),
            "y1": float(getattr(bbox_obj, "y1", 0.0)),
        }
        token_count = len(content.split())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        metadata = {}
        if section:
            metadata["parent_section"] = section
        if split:
            metadata["split"] = "true"
        if chunk_type == "figure":
            metadata["caption_attached"] = "true"
        if extra_metadata:
            metadata.update({k: str(v) for k, v in extra_metadata.items()})
        relationships: List[str] = []
        if "xref:" in content.lower():
            relationships.append(content)

        return LDU(
            content=content,
            chunk_type=chunk_type,
            page_refs=[page.page_number],
            bounding_box=bbox,
            parent_section=section,
            token_count=token_count,
            content_hash=content_hash,
            metadata=metadata,
            items=items,
            relationships=relationships,
        )

    def _aggregate_bbox(self, page: PageExtractionResult) -> BoundingBox:
        if not page.text_blocks:
            return BoundingBox(x0=0, y0=0, x1=0, y1=0)
        x0 = min(tb.bbox.x0 for tb in page.text_blocks)
        y0 = min(tb.bbox.y0 for tb in page.text_blocks)
        x1 = max(tb.bbox.x1 for tb in page.text_blocks)
        y1 = max(tb.bbox.y1 for tb in page.text_blocks)
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def _is_section_header(self, line: str) -> bool:
        return bool(line) and (line.isupper() or line.endswith(":"))

    def _is_numbered_item(self, line: str) -> bool:
        return bool(re.match(r"^\d+[\.\)]\s", line))


__all__ = ["ChunkingEngine", "ChunkValidator", "LDU"]
