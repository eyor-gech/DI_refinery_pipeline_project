"""
VisionExtractor
---------------
Strategy C: OCR + VLM hybrid extraction for scanned or low-quality pages.
Uses pytesseract first, then escalates to a multimodal VLM (OpenRouter) when OCR is insufficient.
Includes per-document budget guard (token/cost).
"""
from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from dotenv import load_dotenv

import pdfplumber
import pytesseract
from pydantic import BaseModel, ConfigDict, Field

from src.models.extracted import BoundingBox, ExtractedDocument, PageExtractionResult, TextBlock

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

load_dotenv() 
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
    Vision-based extractor with OCR-first, VLM-fallback strategy.
    """

    cost_cap_usd: float = 1.0
    vlm_models: Dict[str, float] = Field(
        default_factory=lambda: {
            "qwen2-vl-72b-instruct": 0.0000025,  # cost per token USD (example)
            "gemini-2.5-flash": 0.0000020,
        }
    )
    languages: Tuple[str, ...] = ("eng", "amh", "osd")
    ocr_min_chars: int = 50
    ocr_density_threshold: float = 1e-5
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
                text, confidence = self._run_ocr(page)

                # Heuristic: if OCR result is weak, escalate
                if len(text.strip()) < self.ocr_min_chars or confidence == "low_confidence":
                    fallback_text = self._try_vlm(page, cost_tracker)
                    if fallback_text:
                        text = fallback_text
                        confidence = "high_confidence"
                    else:
                        confidence = "low_confidence"

                page_time_ms = (time.perf_counter() - page_start) * 1000
                page_result = self._page_result(page, page_number, text, confidence, page_time_ms)
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

    # OCR path -----------------------------------------------------------------
    def _run_ocr(self, page) -> Tuple[str, str]:
        try:
            image = page.to_image(resolution=200).original
        except Exception as exc:  # pragma: no cover - pdf rendering fallback
            logger.warning("Falling back to rasterize page: %s", exc)
            image = None

        text = ""
        for lang in self.languages:
            try:
                text = pytesseract.image_to_string(image, lang=lang)
                if text and len(text.strip()) >= self.ocr_min_chars:
                    break
            except Exception:  # pragma: no cover - tesseract issues
                continue

        confidence = "high_confidence" if len(text.strip()) >= self.ocr_min_chars else "low_confidence"
        return text, confidence

    # VLM path -----------------------------------------------------------------
    def _try_vlm(self, page, cost_tracker: CostTracker) -> str:
        """Attempt VLM transcription respecting budget."""
        image_b64 = self._page_image_b64(page)

        # Prefer cheaper model first
        for model, cost_per_token in sorted(self.vlm_models.items(), key=lambda kv: kv[1]):
            estimated_tokens = 800  # rough default prompt+output guess
            estimated_cost = estimated_tokens * cost_per_token

            if not cost_tracker.can_spend(estimated_cost):
                logger.info("Budget cap reached; skipping VLM model %s", model)
                continue

            try:
                text, tokens_used = self._call_vlm(image_b64, model)
                cost = tokens_used * cost_per_token
                cost_tracker.record(model, tokens_used, cost)
                return text
            except Exception as exc:  # pragma: no cover - network / API errors
                logger.warning("VLM call failed for %s: %s", model, exc)
                continue

        return ""

    def _call_vlm(self, image_b64: str, model: str) -> Tuple[str, int]:
        """
        Call OpenRouter API with the given model and base64-encoded page image.
        Returns: (extracted_text, tokens_used)
        """
        # Choose API key based on model
        if "qwen" in model.lower():
            api_key = os.getenv("VLM_qwen")
        elif "gemini" in model.lower():
            api_key = os.getenv("VLM_gemini")
        else:
            raise RuntimeError(f"No API key configured for model {model}")

        if not api_key:
            raise RuntimeError(f"API key not found for model {model}")

        endpoint = "https://openrouter.ai/api/v1/chat/completions"

        # Build a prompt for image-to-text extraction
        prompt = (
            "Extract all readable text from this document page image. "
            "Keep the structure, tables, and figures in text form. "
            "Return only text output."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt, "image": image_b64}
            ],
            "max_tokens": 2000,
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Extract text from response structure
            text = data["choices"][0]["message"]["content"]
            # Estimate tokens (for tracking budget; simple placeholder)
            tokens_used = len(text.split())
            return text, tokens_used
        except Exception as e:
            raise RuntimeError(f"VLM API call failed for model {model}: {e}")

    # Helpers ------------------------------------------------------------------
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

        # One block covering whole page; finer segmentation can be added later.
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=float(page.width), y1=float(page.height))
        text_block = TextBlock(text=text, bbox=bbox, reading_order=0)

        # Image ratio heuristic
        image_area = 0.0
        for img in getattr(page, "images", []) or []:
            try:
                image_area += float(img.get("width", 0)) * float(img.get("height", 0))
            except Exception:
                continue
        image_ratio = image_area / page_area if page_area else 0.0

        chars_meta = getattr(page, "chars", []) or []
        font_meta = any(isinstance(ch, dict) and ("fontname" in ch or "font" in ch) for ch in chars_meta)

        return PageExtractionResult(
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


__all__ = ["VisionExtractor", "CostTracker"]
