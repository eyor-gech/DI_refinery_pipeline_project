"""
FastTextExtractor
-----------------
Low-cost extraction path for native digital PDFs using pdfplumber.
Produces an ExtractedDocument with per-page confidence and escalation hints.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import pdfplumber
from pydantic import BaseModel, ConfigDict

from src.models.extracted import (
    BoundingBox,
    ExtractedDocument,
    PageExtractionResult,
    TextBlock,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FastTextExtractor(BaseModel):
    """
    Strategy A: fast text extraction for native digital PDFs.
    Uses pdfplumber for text/word extraction and derives confidence heuristics.
    """

    # Thresholds tuned for typical A4 / Letter PDFs.
    density_high_threshold: float = 5e-5
    image_high_threshold: float = 0.35
    image_low_threshold: float = 0.2
    char_low_threshold: int = 50
    char_high_threshold: int = 200

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def extract(self, pdf_path: Path | str) -> ExtractedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        start = time.perf_counter()
        pages: List[PageExtractionResult] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_start = time.perf_counter()
                result = self._extract_page(page, page_number)
                page_elapsed = (time.perf_counter() - page_start) * 1000
                result.extraction_time_ms = page_elapsed
                pages.append(result)

        total_time_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "FastTextExtractor processed %s in %.2f ms (%d pages)",
            pdf_path,
            total_time_ms,
            len(pages),
        )

        return ExtractedDocument(
            doc_id=pdf_path.stem,
            pages=pages,
            total_time_ms=total_time_ms,
        )

    # Internal helpers ---------------------------------------------------------
    def _extract_page(self, page, page_number: int) -> PageExtractionResult:
        page_area = float(page.width * page.height)
        text = page.extract_text() or ""
        char_count = len(text)
        char_density = char_count / page_area if page_area else 0.0

        # Image coverage
        image_area = 0.0
        for img in getattr(page, "images", []) or []:
            try:
                w = float(img.get("width", 0))
                h = float(img.get("height", 0))
                image_area += w * h
            except Exception:  # pragma: no cover - defensive
                continue
        image_area_ratio = image_area / page_area if page_area else 0.0

        # Font metadata presence
        chars = getattr(page, "chars", []) or []
        font_metadata_present = any(
            isinstance(ch, dict) and ("fontname" in ch or "font" in ch) for ch in chars
        )

        words = page.extract_words() or []
        text_blocks = self._build_text_blocks(words)
        reading_ordered_text = " ".join(block.text for block in text_blocks).strip()
        full_text = text if text else reading_ordered_text

        confidence, needs_escalation = self._confidence(
            char_count=char_count,
            char_density=char_density,
            font_metadata_present=font_metadata_present,
            image_area_ratio=image_area_ratio,
        )

        return PageExtractionResult(
            page_number=page_number,
            text_blocks=text_blocks,
            full_text=full_text,
            confidence=confidence,
            needs_layout_escalation=needs_escalation,
            character_count=char_count,
            character_density=char_density,
            image_area_ratio=image_area_ratio,
            font_metadata_presence=font_metadata_present,
            extraction_time_ms=0.0,  # filled by caller
        )

    def _build_text_blocks(self, words) -> List[TextBlock]:
        blocks: List[TextBlock] = []
        for idx, w in enumerate(words):
            try:
                bbox = BoundingBox(
                    x0=float(w["x0"]),
                    y0=float(w["top"]) if "top" in w else float(w.get("y0", 0)),
                    x1=float(w["x1"]),
                    y1=float(w["bottom"]) if "bottom" in w else float(w.get("y1", 0)),
                )
                blocks.append(TextBlock(text=w.get("text", ""), bbox=bbox, reading_order=idx))
            except Exception:  # pragma: no cover - defensive
                continue
        return blocks

    def _confidence(
        self,
        *,
        char_count: int,
        char_density: float,
        font_metadata_present: bool,
        image_area_ratio: float,
    ) -> tuple[str, bool]:
        """Return (confidence_label, needs_layout_escalation)."""
        if char_count < self.char_low_threshold or image_area_ratio > self.image_high_threshold:
            return "low_confidence", True

        score = 0
        if char_density > self.density_high_threshold:
            score += 1
        if font_metadata_present:
            score += 1
        if image_area_ratio < self.image_low_threshold:
            score += 1
        if char_count > self.char_high_threshold:
            score += 1

        if score >= 3:
            return "high_confidence", False
        return "medium_confidence", False


__all__ = ["FastTextExtractor"]
