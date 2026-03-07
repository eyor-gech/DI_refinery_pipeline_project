"""
VisionExtractor
---------------
Strategy C (vision-first): Gemini-3-Flash-Preview via Ollama is the primary extractor for
all page images (Amharic + English, multi-column, tables, figures). Budget-aware
fallbacks (qwen2-vl-72b-instruct, gemini-2.5-flash) remain available through the
existing CostTracker.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import pdfplumber
from pydantic import BaseModel, ConfigDict, Field

from src.models.extracted import BoundingBox, ExtractedDocument, PageExtractionResult, TextBlock

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class CostTracker:
    """Tracks token usage and cost per document."""

    cap_usd: float = 1.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    events: List[Tuple[str, int, float]] = field(default_factory=list)  # (model, tokens, cost)

    def can_spend(self, estimated_cost: float) -> bool:
        return (self.total_cost_usd + estimated_cost) <= self.cap_usd

    def record(self, model: str, tokens: int, cost: float) -> None:
        self.total_tokens += tokens
        self.total_cost_usd += cost
        self.events.append((model, tokens, cost))


class VisionExtractor(BaseModel):
    """
    Vision-first extractor with Gemini primary and VLM fallbacks.
    """

    cost_cap_usd: float = 1.0
    primary_model: str = "gemini-3-flash-preview"
    primary_cost_per_token: float = 0.0000025
    vlm_models: Dict[str, float] = Field(
        default_factory=lambda: {
            "qwen2-vl-72b-instruct": 0.0000025,
            "gemini-2.5-flash": 0.0000020,
        }
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def extract(self, pdf_path: Path | str) -> ExtractedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        cost_tracker = CostTracker(cap_usd=self.cost_cap_usd)
        pages: List[PageExtractionResult] = []
        start = time.perf_counter()

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_start = time.perf_counter()
                text = ""
                confidence = "high_confidence"

                try:
                    text, tokens = self._run_gemini_primary(page, cost_tracker)
                    cost_tracker.record(self.primary_model, tokens, tokens * self.primary_cost_per_token)
                except Exception as exc:
                    logger.warning("Primary Gemini extraction failed on page %s: %s", page_number, exc)
                    confidence = "low_confidence"

                if not text.strip():
                    fallback_text = self._try_vlm_fallback(page, cost_tracker)
                    if fallback_text:
                        text = fallback_text
                        confidence = "high_confidence"

                page_time_ms = (time.perf_counter() - page_start) * 1000
                page_result = self._page_result(page, page_number, text, confidence, page_time_ms)
                # attach content hash for provenance consumers
                page_result.__dict__["content_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
                pages.append(page_result)

        total_time_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "VisionExtractor processed %s pages=%d cost_usd=%.6f tokens=%d",
            pdf_path,
            len(pages),
            cost_tracker.total_cost_usd,
            cost_tracker.total_tokens,
        )

        return ExtractedDocument(
            doc_id=pdf_path.stem,
            extractor="vision",
            pages=pages,
            total_time_ms=total_time_ms,
        )

    # Gemini primary --------------------------------------------------------
    def _run_gemini_primary(self, page, cost_tracker: CostTracker) -> Tuple[str, int]:
        """Use Ollama Gemini-3-Flash-Preview as primary extractor."""
        image_b64 = self._page_image_b64(page)
        estimated_tokens = 1200
        estimated_cost = estimated_tokens * self.primary_cost_per_token
        if not cost_tracker.can_spend(estimated_cost):
            raise RuntimeError("Budget cap reached before Gemini primary call")

        # Ollama HTTP API (preferred for structured output)
        endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
        prompt = (
            "Extract all text from this page image. Preserve multi-column order, "
            "keep table headers with rows, keep figure captions attached, and keep layout separators. "
            "Return clean, readable structured text."
        )
        payload = {
            "model": self.primary_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        resp = requests.post(endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "") or data.get("data", "")
        tokens_used = len(text.split()) or estimated_tokens
        return text, tokens_used

    # Fallback VLMs --------------------------------------------------------
    def _try_vlm_fallback(self, page, cost_tracker: CostTracker) -> str:
        image_b64 = self._page_image_b64(page)
        for model, cost_per_token in sorted(self.vlm_models.items(), key=lambda kv: kv[1]):
            estimated_tokens = 800
            estimated_cost = estimated_tokens * cost_per_token
            if not cost_tracker.can_spend(estimated_cost):
                logger.info("Budget cap reached; skipping VLM fallback %s", model)
                continue
            try:
                text, tokens_used = self._call_vlm(image_b64, model)
                cost_tracker.record(model, tokens_used, tokens_used * cost_per_token)
                return text
            except Exception as exc:
                logger.warning("VLM fallback failed for %s: %s", model, exc)
                continue
        return ""

    def _call_vlm(self, image_b64: str, model: str) -> Tuple[str, int]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing for fallback VLM")

        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        prompt = (
            "You are a document transcription model. Return the page text with tables and captions preserved. "
            "Keep logical reading order."
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "image": image_b64}],
            "max_tokens": 2000,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tokens_used = len(text.split())
        return text, tokens_used

    # Helpers --------------------------------------------------------------
    def _page_image_b64(self, page) -> str:
        img = page.to_image(resolution=300).original
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _page_result(self, page, page_number: int, text: str, confidence: str, elapsed_ms: float) -> PageExtractionResult:
        page_area = float(page.width * page.height)
        char_count = len(text)
        char_density = char_count / page_area if page_area else 0.0

        bbox = BoundingBox(x0=0.0, y0=0.0, x1=float(page.width), y1=float(page.height))
        text_block = TextBlock(text=text, bbox=bbox, reading_order=0)

        image_area = 0.0
        for img in getattr(page, "images", []) or []:
            try:
                image_area += float(img.get("width", 0)) * float(img.get("height", 0))
            except Exception:
                continue
        image_ratio = image_area / page_area if page_area else 0.0

        chars_meta = getattr(page, "chars", []) or []
        font_meta = any(isinstance(ch, dict) and ("fontname" in ch or "font" in ch) for ch in chars_meta)

        result = PageExtractionResult(
            page_number=page_number,
            text_blocks=[text_block],
            full_text=text,
            confidence=confidence,
            needs_layout_escalation=False,
            character_count=char_count,
            character_density=char_density,
            image_area_ratio=image_ratio,
            font_metadata_presence=font_meta,
            extraction_time_ms=elapsed_ms,
            tables=[],
            figures=[],
        )
        return result


__all__ = ["VisionExtractor", "CostTracker"]
