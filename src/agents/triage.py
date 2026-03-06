"""
Triage Agent
------------
Analyzes a PDF and emits a DocumentProfile describing origin, layout, and
processing cost hints. Designed to be light-weight and work with heterogeneous
documents prior to heavier extraction passes.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber
from pydantic import BaseModel, Field

# Configure a module-level logger. In production this should be configured by the host application.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DocumentProfile(BaseModel):
    """Structured profile for a document."""

    doc_id: str
    origin_type: str
    layout_complexity: str
    language: str
    language_confidence: float = Field(ge=0.0, le=1.0)
    domain_hint: str
    estimated_extraction_cost: str
    character_count: int
    character_density: float
    whitespace_ratio: float
    image_area_ratio: float
    font_metadata_presence: bool


@dataclass
class TextMetrics:
    character_count: int
    character_density: float
    whitespace_ratio: float
    image_area_ratio: float
    font_metadata_presence: bool
    text_sample: str
    column_counts: Counter
    table_pages: int
    widget_annotations: int


class TriageAgent:
    """Analyzes PDFs and produces DocumentProfile instances."""

    def __init__(self, profiles_dir: Path | str = ".refinery/profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    # Public API -----------------------------------------------------------------
    def profile(self, pdf_path: Path | str) -> DocumentProfile:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Starting triage for %s", pdf_path)
        metrics = self.analyze_text_density(pdf_path)
        origin = self.detect_origin_type(metrics)
        layout = self.detect_layout_complexity(metrics)
        lang, lang_conf = self.detect_language(metrics.text_sample)
        domain = self.detect_domain_hint(metrics.text_sample)
        cost = self.estimate_extraction_cost(origin, layout, metrics.image_area_ratio)

        profile = DocumentProfile(
            doc_id=pdf_path.stem,
            origin_type=origin,
            layout_complexity=layout,
            language=lang,
            language_confidence=lang_conf,
            domain_hint=domain,
            estimated_extraction_cost=cost,
            character_count=metrics.character_count,
            character_density=metrics.character_density,
            whitespace_ratio=metrics.whitespace_ratio,
            image_area_ratio=metrics.image_area_ratio,
            font_metadata_presence=metrics.font_metadata_presence,
        )

        self._persist_profile(profile)
        logger.info("Triage complete for %s", pdf_path)
        return profile

    # Core analysis --------------------------------------------------------------
    def analyze_text_density(self, pdf_path: Path | str) -> TextMetrics:
        """Compute document-level text and layout metrics using pdfplumber."""
        character_count = 0
        whitespace_count = 0
        page_areas: List[float] = []
        image_area = 0.0
        font_metadata_present = False
        text_fragments: List[str] = []
        column_counts: Counter = Counter()
        table_pages = 0
        widget_annotations = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_area = float(page.width * page.height)
                page_areas.append(page_area)

                text = page.extract_text() or ""
                text_fragments.append(text[:500])  # sample for language/domain
                character_count += len(text)
                whitespace_count += sum(1 for ch in text if ch.isspace())

                if getattr(page, "images", None):
                    for img in page.images:
                        try:
                            w = float(img.get("width", 0))
                            h = float(img.get("height", 0))
                            image_area += w * h
                        except Exception:  # pragma: no cover - defensive
                            continue

                chars = getattr(page, "chars", [])
                if chars:
                    font_metadata_present = any(
                        isinstance(ch, dict) and ("fontname" in ch or "font" in ch)
                        for ch in chars
                    ) or font_metadata_present

                # Column estimation using word x-coordinates
                words = getattr(page, "extract_words", lambda **_: [])()
                if words:
                    buckets = self._bucket_columns(words)
                    for bucket_idx, count in buckets.items():
                        if count > 0:
                            column_counts[bucket_idx] += 1

                # Table detection (cheap heuristic)
                if hasattr(page, "find_tables"):
                    tables = page.find_tables() or []
                    if tables:
                        table_pages += 1

                # Form fields hint
                annotations = getattr(page, "annots", []) or []
                widget_annotations += sum(
                    1
                    for ann in annotations
                    if isinstance(ann, dict) and ann.get("Subtype") == "/Widget"
                )

        total_area = sum(page_areas) or 1.0
        character_density = character_count / total_area
        total_chars = character_count or 1
        whitespace_ratio = whitespace_count / total_chars
        image_area_ratio = image_area / total_area
        text_sample = "\n".join(text_fragments)[:2000]

        logger.debug(
            "Metrics: chars=%d, density=%.6f, whitespace=%.4f, images=%.4f, fonts=%s",
            character_count,
            character_density,
            whitespace_ratio,
            image_area_ratio,
            font_metadata_present,
        )

        return TextMetrics(
            character_count=character_count,
            character_density=character_density,
            whitespace_ratio=whitespace_ratio,
            image_area_ratio=image_area_ratio,
            font_metadata_presence=font_metadata_present,
            text_sample=text_sample,
            column_counts=column_counts,
            table_pages=table_pages,
            widget_annotations=widget_annotations,
        )

    def detect_origin_type(self, metrics: TextMetrics) -> str:
        """Classify whether a PDF is digital, scanned, mixed, or form fillable."""
        image_heavy = metrics.image_area_ratio > 0.25
        text_dense = metrics.character_density > 0.00005  # tuned for typical A4 sizes
        has_fonts = metrics.font_metadata_presence
        has_forms = metrics.widget_annotations > 0

        if has_forms:
            logger.debug("Detected form widgets; classifying as form_fillable")
            return "form_fillable"

        if text_dense and has_fonts and not image_heavy:
            return "native_digital"

        if not text_dense and image_heavy:
            return "scanned_image"

        return "mixed"

    def detect_layout_complexity(self, metrics: TextMetrics) -> str:
        """Infer layout complexity (single/multi/table/figure/mixed)."""
        multi_col_pages = sum(1 for count in metrics.column_counts.values() if count >= 2)
        total_pages = max(len(metrics.column_counts), 1)
        multi_col_ratio = multi_col_pages / total_pages

        if metrics.table_pages >= 1 and metrics.table_pages >= total_pages / 2:
            return "table_heavy"
        if metrics.image_area_ratio > 0.3:
            return "figure_heavy"
        if multi_col_ratio > 0.4:
            return "multi_column"
        if metrics.table_pages > 0 and multi_col_ratio > 0.2:
            return "mixed"
        return "single_column"

    def detect_domain_hint(self, text_sample: str) -> str:
        """Lightweight keyword-driven domain inference."""
        lowered = text_sample.lower()
        domain_keywords: Dict[str, Iterable[str]] = {
            "financial": ["invoice", "balance sheet", "earnings", "debit", "credit", "account"],
            "legal": ["plaintiff", "defendant", "hereby", "agreement", "contract", "clause"],
            "technical": ["algorithm", "specification", "api", "architecture", "protocol"],
            "medical": ["patient", "diagnosis", "treatment", "clinical", "medication"],
        }
        scores = defaultdict(int)
        for domain, keywords in domain_keywords.items():
            for kw in keywords:
                if kw in lowered:
                    scores[domain] += 1
        if not scores:
            return "general"
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def detect_language(self, text_sample: str) -> Tuple[str, float]:
        """Detect language using langdetect if available; otherwise heuristic."""
        sample = text_sample.strip()
        if not sample:
            return "unknown", 0.0
        try:
            from langdetect import detect, DetectorFactory, detect_langs

            DetectorFactory.seed = 0
            langs = detect_langs(sample[:1000])
            best = langs[0]
            return best.lang, min(best.prob, 1.0)
        except Exception:  # pragma: no cover - dependency optional
            # Fallback: crude ASCII ratio heuristic for English
            ascii_ratio = sum(1 for ch in sample if ord(ch) < 128) / len(sample)
            return ("en" if ascii_ratio > 0.8 else "unknown", 0.5 if ascii_ratio > 0.8 else 0.2)

    def estimate_extraction_cost(self, origin: str, layout: str, image_ratio: float) -> str:
        """Map profile attributes to extraction cost class."""
        if origin == "scanned_image" or image_ratio > 0.35:
            return "needs_vision_model"
        if layout in {"multi_column", "table_heavy", "figure_heavy", "mixed"}:
            return "needs_layout_model"
        return "fast_text_sufficient"

    # Helpers --------------------------------------------------------------------
    def _bucket_columns(self, words: List[dict], bucket_width: float = 120.0) -> Counter:
        """
        Bucket words along x-axis to approximate column counts.
        Returns Counter keyed by bucket index with word counts.
        """
        buckets = Counter()
        for w in words:
            try:
                center = (float(w["x0"]) + float(w["x1"])) / 2.0
                bucket = int(center // bucket_width)
                buckets[bucket] += 1
            except Exception:  # pragma: no cover - defensive
                continue
        return buckets

    def _persist_profile(self, profile: DocumentProfile) -> None:
        """Persist profile JSON to the profiles directory."""
        output_path = self.profiles_dir / f"{profile.doc_id}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(profile.dict(), f, ensure_ascii=False, indent=2)
        logger.debug("Profile written to %s", output_path)


def main(pdf_path: str) -> None:
    """CLI entry point for ad-hoc runs."""
    logging.basicConfig(level=logging.INFO)
    agent = TriageAgent()
    profile = agent.profile(pdf_path)
    print(profile.json(indent=2))


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.agents.triage <path-to-pdf>")
    main(sys.argv[1])
