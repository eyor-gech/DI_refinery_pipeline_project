"""Strategy implementations for document extraction."""

from .FastTextExtractor import FastTextExtractor
from .LayoutExtractor import LayoutExtractor, DoclingDocumentAdapter

__all__ = ["FastTextExtractor", "LayoutExtractor", "DoclingDocumentAdapter"]
