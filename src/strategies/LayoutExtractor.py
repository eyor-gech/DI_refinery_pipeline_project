"""
LayoutExtractor
---------------
Strategy B for complex layouts (multi-column, tables, figures).
Primary path uses a Docling-style layout model, with a pdfplumber fallback.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pdfplumber
from pydantic import BaseModel, ConfigDict, Field

from src.models.extracted import (
    BoundingBox,
    ExtractedDocument,
    FigureBlock,
    PageExtractionResult,
    TableBlock,
    TextBlock,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DoclingDocumentAdapter:
    """
    Converts Docling-style output into the ExtractedDocument schema.
    The adapter expects a lightweight dict structure so it can be easily mocked in tests:
    {
        "pages": [
            {
                "number": 1,
                "size": (width, height),
                "blocks": [{"text": "...", "bbox": [x0, y0, x1, y1], "reading_order": 0}],
                "tables": [{"bbox": [...], "rows": [["a", "b"]], "caption": "..."}],
                "figures": [{"bbox": [...], "caption": "..."}],
                "elapsed_ms": 12.3
            },
            ...
        ]
    }
    """

    def adapt(self, docling_doc: Dict[str, Any], pdf_path: Path | str, total_time_ms: float) -> ExtractedDocument:
        pages: List[PageExtractionResult] = []

        for page_data in docling_doc.get("pages", []):
            width, height = self._page_size(page_data)
            page_area = max(width * height, 1.0)

            text_blocks = self._blocks(page_data.get("blocks", []))
            full_text = " ".join(tb.text for tb in text_blocks).strip()
            char_count = len(full_text)
            char_density = char_count / page_area

            tables = self._tables(page_data.get("tables", []))
            figures = self._figures(page_data.get("figures", []))

            page_time_ms = float(page_data.get("elapsed_ms", 0.0))

            pages.append(
                PageExtractionResult(
                    page_number=int(page_data.get("number", len(pages) + 1)),
                    text_blocks=text_blocks,
                    full_text=full_text,
                    confidence="high_confidence" if char_density > 1e-5 else "medium_confidence",
                    needs_layout_escalation=False,
                    character_count=char_count,
                    character_density=char_density,
                    image_area_ratio=page_data.get("image_area_ratio", 0.0),
                    font_metadata_presence=bool(page_data.get("font_metadata_presence", True)),
                    extraction_time_ms=page_time_ms,
                    tables=tables,
                    figures=figures,
                )
            )

        return ExtractedDocument(
            doc_id=Path(pdf_path).stem,
            extractor="layout",
            pages=pages,
            total_time_ms=total_time_ms,
        )

    def _page_size(self, page_data: Dict[str, Any]) -> Tuple[float, float]:
        size = page_data.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return float(size[0]), float(size[1])
        bbox = page_data.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
        return 612.0, 792.0  # sensible default (US Letter points)

    def _blocks(self, blocks: List[Dict[str, Any]]) -> List[TextBlock]:
        converted: List[TextBlock] = []
        for idx, b in enumerate(blocks):
            bbox = self._bbox_from(b.get("bbox"))
            text = b.get("text", "")
            order = b.get("reading_order", idx)
            converted.append(TextBlock(text=text, bbox=bbox, reading_order=int(order)))
        return converted

    def _tables(self, tables: List[Dict[str, Any]]) -> List[TableBlock]:
        converted: List[TableBlock] = []
        for t in tables:
            bbox = self._bbox_from(t.get("bbox"))
            rows = t.get("rows", [])
            caption = t.get("caption")
            converted.append(TableBlock(bbox=bbox, rows=rows, caption=caption))
        return converted

    def _figures(self, figures: List[Dict[str, Any]]) -> List[FigureBlock]:
        converted: List[FigureBlock] = []
        for f in figures:
            bbox = self._bbox_from(f.get("bbox"))
            converted.append(FigureBlock(bbox=bbox, caption=f.get("caption")))
        return converted

    def _bbox_from(self, bbox_like) -> BoundingBox:
        try:
            x0, y0, x1, y1 = bbox_like
            return BoundingBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))
        except Exception:  # pragma: no cover - defensive fallback
            return BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)


class LayoutExtractor(BaseModel):
    """
    Strategy B: layout-aware extraction with Docling-first, pdfplumber fallback.
    """

    adapter: DoclingDocumentAdapter = Field(default_factory=DoclingDocumentAdapter)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def extract(self, pdf_path: Path | str) -> ExtractedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        start = time.perf_counter()
        try:
            docling_doc = self._run_docling(pdf_path)
            total_time_ms = (time.perf_counter() - start) * 1000
            if docling_doc:
                logger.info("Docling extraction succeeded for %s", pdf_path)
                return self.adapter.adapt(docling_doc, pdf_path, total_time_ms)
        except Exception as exc:
            logger.warning("Docling extraction unavailable, falling back to pdfplumber: %s", exc)

        # Fallback
        return self._fallback_pdfplumber(pdf_path, start_time=start)

    # Implementation details -------------------------------------------------
    def _run_docling(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Placeholder hook for a Docling/minerU/DocTR pipeline.
        In production, connect the actual layout model here.
        """
        raise ImportError("Docling pipeline not configured")

    def _fallback_pdfplumber(self, pdf_path: Path, start_time: float) -> ExtractedDocument:
        pages: List[PageExtractionResult] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_start = time.perf_counter()
                page_area = float(page.width * page.height)

                text = page.extract_text() or ""
                char_count = len(text)
                char_density = char_count / page_area if page_area else 0.0

                image_area = 0.0
                for img in getattr(page, "images", []) or []:
                    try:
                        image_area += float(img.get("width", 0)) * float(img.get("height", 0))
                    except Exception:  # pragma: no cover
                        continue
                image_ratio = image_area / page_area if page_area else 0.0

                chars = getattr(page, "chars", []) or []
                font_metadata_present = any(
                    isinstance(ch, dict) and ("fontname" in ch or "font" in ch) for ch in chars
                )

                words = page.extract_words() or []
                text_blocks = self._build_text_blocks(words)
                full_text = text or " ".join(tb.text for tb in text_blocks)

                tables = self._extract_tables(page)

                page_time_ms = (time.perf_counter() - page_start) * 1000

                pages.append(
                    PageExtractionResult(
                        page_number=page_number,
                        text_blocks=text_blocks,
                        full_text=full_text,
                        confidence="medium_confidence",
                        needs_layout_escalation=image_ratio > 0.5,
                        character_count=char_count,
                        character_density=char_density,
                        image_area_ratio=image_ratio,
                        font_metadata_presence=font_metadata_present,
                        extraction_time_ms=page_time_ms,
                        tables=tables,
                        figures=[],
                    )
                )

        total_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Fallback layout extraction processed %s in %.2f ms", pdf_path, total_time_ms)
        return ExtractedDocument(
            doc_id=pdf_path.stem, extractor="layout", pages=pages, total_time_ms=total_time_ms
        )

    def _build_text_blocks(self, words) -> List[TextBlock]:
        blocks: List[TextBlock] = []
        for idx, w in enumerate(words):
            try:
                bbox = BoundingBox(
                    x0=float(w["x0"]),
                    y0=float(w.get("top", w.get("y0", 0))),
                    x1=float(w["x1"]),
                    y1=float(w.get("bottom", w.get("y1", 0))),
                )
                blocks.append(TextBlock(text=w.get("text", ""), bbox=bbox, reading_order=idx))
            except Exception:  # pragma: no cover
                continue
        return blocks

    def _extract_tables(self, page) -> List[TableBlock]:
        tables: List[TableBlock] = []
        if not hasattr(page, "find_tables"):
            return tables
        for tbl in page.find_tables() or []:
            try:
                rows = tbl.extract()
                x0, y0, x1, y1 = tbl.bbox
                bbox = BoundingBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))
                tables.append(TableBlock(bbox=bbox, rows=rows, caption=None))
            except Exception:  # pragma: no cover
                continue
        return tables


__all__ = ["LayoutExtractor", "DoclingDocumentAdapter"]
