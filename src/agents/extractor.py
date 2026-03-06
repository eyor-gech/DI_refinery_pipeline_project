"""
Extraction Router
-----------------
Routes documents to extraction strategies based on profile, with confidence-gated escalation
and structured logging to .refinery/extraction_ledger.jsonl.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.agents.triage import DocumentProfile
from src.models.extracted import ExtractedDocument, PageExtractionResult
from src.strategies import FastTextExtractor, LayoutExtractor, VisionExtractor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ExtractionRouter:
    def __init__(
        self,
        strategies: Dict[str, object] | None = None,
        thresholds: Dict[str, float] | None = None,
        confidence_map: Dict[str, float] | None = None,
        ledger_path: Path | str = ".refinery/extraction_ledger.jsonl",
    ):
        self.strategies = strategies or {
            "fast_text": FastTextExtractor(),
            "layout": LayoutExtractor(),
            "vision": VisionExtractor(),
        }
        self.thresholds = thresholds or {"fast_text": 0.7, "layout": 0.7, "vision": 0.7}
        self.confidence_map = confidence_map or {
            "high_confidence": 1.0,
            "medium_confidence": 0.6,
            "low_confidence": 0.2,
        }
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    # Public API --------------------------------------------------------------
    def route(self, pdf_path: Path | str, profile: DocumentProfile) -> ExtractedDocument:
        plan = self._plan_for_profile(profile)
        logger.info("Routing %s with plan %s", pdf_path, plan)

        page_results: Dict[int, PageExtractionResult] = {}
        start_time = time.perf_counter()
        total_pages_expected = None

        for level, strategy_name in enumerate(plan):
            if strategy_name not in self.strategies:
                raise KeyError(f"Strategy '{strategy_name}' not configured in router")
            extractor = self.strategies[strategy_name]
            doc = extractor.extract(pdf_path)
            if total_pages_expected is None:
                total_pages_expected = len(doc.pages)

            for page in doc.pages:
                if page.page_number in page_results:
                    continue  # already accepted

                numeric_conf = self._confidence_score(page.confidence)
                threshold = self.thresholds.get(strategy_name, 0.7)
                meets = numeric_conf >= threshold

                page.strategy_used = strategy_name
                page.escalated = level > 0

                if meets or level == len(plan) - 1:
                    if not meets:
                        page.flagged_for_review = True
                    page_results[page.page_number] = page
                    self._log_entry(
                        document_id=doc.doc_id,
                        page=page,
                        strategy=strategy_name,
                        confidence=numeric_conf,
                        threshold=threshold,
                    )
                # else: defer page to next strategy

            if total_pages_expected is not None and len(page_results) >= total_pages_expected:
                break

        total_time_ms = (time.perf_counter() - start_time) * 1000
        pages_sorted = [page_results[k] for k in sorted(page_results.keys())]
        return ExtractedDocument(
            doc_id=Path(pdf_path).stem,
            extractor=plan[0],
            pages=pages_sorted,
            total_time_ms=total_time_ms,
        )

    # Helpers -----------------------------------------------------------------
    def _plan_for_profile(self, profile: DocumentProfile) -> Sequence[str]:
        origin = profile.origin_type
        layout = profile.layout_complexity

        if origin == "native_digital" and layout in {"single_column", "simple"}:
            return ("fast_text", "layout", "vision")
        if origin == "scanned_image":
            return ("vision",)
        return ("layout", "vision")

    def _confidence_score(self, label: str) -> float:
        return self.confidence_map.get(label, 0.0)

    def _log_entry(
        self,
        document_id: str,
        page: PageExtractionResult,
        strategy: str,
        confidence: float,
        threshold: float,
    ) -> None:
        entry = {
            "document_id": document_id,
            "page_number": page.page_number,
            "strategy_used": strategy,
            "confidence_score": confidence,
            "threshold": threshold,
            "processing_time": page.extraction_time_ms,
            "cost_estimate": page.cost_estimate,
            "escalation_occurred": page.escalated,
            "flagged_for_review": page.flagged_for_review,
        }
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


__all__ = ["ExtractionRouter"]
