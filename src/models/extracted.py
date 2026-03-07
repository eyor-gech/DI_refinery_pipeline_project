"""
Data models for extraction outputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box."""

    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(BaseModel):
    """Extracted text snippet with spatial metadata."""

    text: str
    bbox: BoundingBox
    reading_order: int


class TableBlock(BaseModel):
    """Structured table representation."""

    bbox: BoundingBox
    rows: List[List[str]]
    caption: Optional[str] = None


class FigureBlock(BaseModel):
    """Detected figure with optional caption."""

    bbox: BoundingBox
    caption: Optional[str] = None


class PageExtractionResult(BaseModel):
    page_number: int
    text_blocks: List[TextBlock]
    full_text: str
    confidence: str
    needs_layout_escalation: bool = False
    character_count: int
    character_density: float
    image_area_ratio: float
    font_metadata_presence: bool
    extraction_time_ms: float
    tables: List[TableBlock] = Field(default_factory=list)
    figures: List[FigureBlock] = Field(default_factory=list)
    strategy_used: str = ""
    escalated: bool = False
    flagged_for_review: bool = False
    cost_estimate: float = 0.0


class ExtractedDocument(BaseModel):
    doc_id: str
    extractor: str = "fast_text"
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pages: List[PageExtractionResult]
    total_time_ms: float
    #tool_used: str = ""
