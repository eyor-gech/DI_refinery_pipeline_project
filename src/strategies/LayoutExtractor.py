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

    def _page_size(self, page_data):

        size = page_data.get("size")

        if isinstance(size, (list, tuple)) and len(size) == 2:
            return float(size[0]), float(size[1])

        bbox = page_data.get("bbox")

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])

        return 612.0, 792.0

    def _blocks(self, blocks):

        converted = []

        for idx, b in enumerate(blocks):

            x0, y0, x1, y1 = b.get("bbox", [0, 0, 0, 0])

            converted.append(
                TextBlock(
                    text=b.get("text", ""),
                    bbox=BoundingBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)),
                    reading_order=int(b.get("reading_order", idx)),
                )
            )

        return converted

    def _tables(self, tables):

        converted = []

        for t in tables:

            x0, y0, x1, y1 = t.get("bbox", [0, 0, 0, 0])

            converted.append(
                TableBlock(
                    bbox=BoundingBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)),
                    rows=t.get("rows", []),
                    caption=t.get("caption"),
                )
            )

        return converted

    def _figures(self, figures):

        converted = []

        for f in figures:

            x0, y0, x1, y1 = f.get("bbox", [0, 0, 0, 0])

            converted.append(
                FigureBlock(
                    bbox=BoundingBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)),
                    caption=f.get("caption"),
                )
            )

        return converted
class LayoutExtractor(BaseModel):
    """
    Strategy B: layout-aware extraction with Docling-first, pdfplumber fallback.
    """

    adapter: Any = Field(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def extract(self, pdf_path: Path | str) -> ExtractedDocument:

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        start = time.perf_counter()

        try:
            docling_doc = self._run_docling(pdf_path)

            if docling_doc and self.adapter:
                total_time_ms = (time.perf_counter() - start) * 1000
                logger.info("Docling extraction succeeded")
                return self.adapter.adapt(docling_doc, pdf_path, total_time_ms)

        except Exception as exc:
            logger.warning("Docling unavailable → fallback: %s", exc)

        return self._fallback_pdfplumber(pdf_path, start)

    # ---------------------------------------------------------
    # DOCLING HOOK
    # ---------------------------------------------------------

    def _run_docling(self, pdf_path: Path) -> Dict[str, Any]:
        raise ImportError("Docling pipeline not configured")

    # ---------------------------------------------------------
    # FALLBACK IMPLEMENTATION
    # ---------------------------------------------------------

    def _fallback_pdfplumber(self, pdf_path: Path, start_time: float) -> ExtractedDocument:

        pages: List[PageExtractionResult] = []

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                page_start = time.perf_counter()

                page_area = float(page.width * page.height)

                # Better extraction parameters
                words = page.extract_words(
                    x_tolerance=2,
                    y_tolerance=2,
                    keep_blank_chars=False,
                    use_text_flow=True,
                ) or []

                columns = self._split_columns(words, page.width)

                text_blocks: List[TextBlock] = []
                order = 0

                for column in columns:

                    lines = self._cluster_lines(column)

                    for line in lines:

                        block = self._line_to_block(line, order)
                        text_blocks.append(block)

                        order += 1

                full_text = "\n".join(tb.text for tb in text_blocks)

                char_count = len(full_text)
                char_density = char_count / page_area if page_area else 0.0

                tables = self._extract_tables(page)

                image_ratio = self._image_ratio(page, page_area)

                font_metadata_present = any(
                    "fontname" in c or "font" in c for c in (page.chars or [])
                )

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

        return ExtractedDocument(
            doc_id=pdf_path.stem,
            extractor="layout",
            pages=pages,
            total_time_ms=total_time_ms,
        )

    # ---------------------------------------------------------
    # LAYOUT HELPERS
    # ---------------------------------------------------------

    def _split_columns(self, words: List[Dict], page_width: float):

        if not words:
            return []

        mid = page_width / 2

        left = []
        right = []

        for w in words:

            center = (w["x0"] + w["x1"]) / 2

            if center < mid:
                left.append(w)
            else:
                right.append(w)

        if left and right:
            return [sorted(left, key=lambda w: (w["top"], w["x0"])),
                    sorted(right, key=lambda w: (w["top"], w["x0"]))]

        return [sorted(words, key=lambda w: (w["top"], w["x0"]))]

    def _cluster_lines(self, words: List[Dict]):

        if not words:
            return []

        lines: List[List[Dict]] = []
        current = [words[0]]

        y_threshold = 3

        for w in words[1:]:

            prev = current[-1]

            if abs(w["top"] - prev["top"]) <= y_threshold:
                current.append(w)

            else:
                lines.append(current)
                current = [w]

        lines.append(current)

        return lines

    def _line_to_block(self, line: List[Dict], order: int) -> TextBlock:

        text = " ".join(w["text"] for w in line)

        x0 = min(w["x0"] for w in line)
        x1 = max(w["x1"] for w in line)
        y0 = min(w["top"] for w in line)
        y1 = max(w["bottom"] for w in line)

        return TextBlock(
            text=text,
            bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
            reading_order=order,
        )

    # ---------------------------------------------------------
    # TABLE EXTRACTION
    # ---------------------------------------------------------

    def _extract_tables(self, page) -> List[TableBlock]:

        tables: List[TableBlock] = []

        try:

            found = page.find_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                }
            )

            for tbl in found:

                rows = tbl.extract()

                x0, y0, x1, y1 = tbl.bbox

                tables.append(
                    TableBlock(
                        bbox=BoundingBox(
                            x0=float(x0),
                            y0=float(y0),
                            x1=float(x1),
                            y1=float(y1),
                        ),
                        rows=rows,
                        caption=None,
                    )
                )

        except Exception:
            pass

        return tables

    # ---------------------------------------------------------
    # IMAGE METRICS
    # ---------------------------------------------------------

    def _image_ratio(self, page, page_area):

        image_area = 0.0

        for img in getattr(page, "images", []) or []:

            try:
                image_area += float(img.get("width", 0)) * float(img.get("height", 0))
            except Exception:
                continue

        return image_area / page_area if page_area else 0.0


__all__ = ["LayoutExtractor", "DoclingDocumentAdapter"]