# Document Intelligence Refinery Pipeline
=================

Production-grade document intelligence pipeline for heterogeneous PDFs (digital, scanned, mixed, table-heavy) that preserves provenance end to end.

## Overview
- Stages: `Triage Agent` → `Extraction Router (multi-strategy)` → `Semantic Chunking Engine` → `PageIndex Builder` → `Query Agent with fact table` + `vector payloads`.
- Inputs: single PDF or directory; large PDFs auto-split, processed, then re-merged.
- Outputs (under `.refinery/`): profiles, extraction ledger, full text, chunks, vector payloads, page index JSON, fact table SQLite DB.

## Architecture
```mermaid
flowchart LR
    ingest[Ingest] --> triage[Triage Agent]
    triage --> router[Extraction Router]
    router -->|FastText/Layout/Vision| extraction[ExtractedDocument]
    extraction --> chunker[Chunking Engine]
    chunker --> indexer[PageIndex Builder]
    chunker --> vectors[Vector Store]
    chunker --> facts[FactTable (SQLite)]
    indexer --> query[Query Agent]
    vectors --> query
    facts --> query
    query --> audit[Audit Mode / Provenance]
```

## Prerequisites
- Python 3.10+ (tested with 3.11).
- System deps: `Tesseract OCR (required for scanned PDFs). Install via `apt-get install tesseract-ocr libgl1` or `brew install tesseract poppler`. On Windows, install Tesseract and add it to `PATH`.
- Optional: Poppler utilities for higher-quality rasterization when `pdf2image` is used.
- LLM: Tested by `Gemini 3 flash`

## Installation (local dev)
- From repo root e.g : `C:\Users\user\repo\refinery_pipeline_project`:
  - Create venv: `python -m venv .venv && .\.venv\Scripts\activate`
  - Upgrade toolchain: `pip install -U pip setuptools wheel`
  - Install runtime deps (mirrors `pyproject.toml`): `pip install fastembed gradio langdetect langgraph pdf2image pdfplumber pdfreader pillow pydantic pypdf pypdf2 pytesseract pytest python-dotenv requests streamlit tqdm`
  - (Optional) export API keys or model tokens in `.env`.

## Pipeline usage
- Full pipeline: `python run_pipeline.py --input data/test_documents`
  - Accepts a single PDF or a folder; large files are split automatically and merged after processing.
  - Artifacts land in `.refinery/` with the same document stem.
- Alternate QueryAgent variant: `python run_pipeline2.py --input path/to/file.pdf`
- Extraction only (no chunking/index/facts): `python run_extraction_pipeline.py` (uses strategies + ledger writing).
- Outputs you should expect:
  - `.refinery/profiles/<doc_id>.json`
  - `.refinery/extraction_ledger.jsonl`
  - `.refinery/fulltext/<doc_id>.txt`
  - `.refinery/chunks/<doc_id>.json`
  - `.refinery/vector_store/<doc_id>.json`
  - `.refinery/pageindex/<doc_id>.json`
  - `.refinery/facts/<doc_id>.db`

## CLI examples
- Process sample corpus: `python run_pipeline.py --input data/test_documents`
- Process one large contract (auto-split/merge): `python run_pipeline.py --input data/contracts/2024_master.pdf`
- Extraction debug pass only: `python run_extraction_pipeline.py`
- Launch RAG demo (Streamlit): `streamlit run streamlit_rag_app.py`
- Launch lightweight RAG UI (Gradio): `python gradio_rag_app.py`

## Docker
- Build image: `docker build -t refinery-pipeline .`
- Run on bundled samples (default CMD): `docker run --rm refinery-pipeline`
- Run on host PDFs and persist outputs:
  - `docker run --rm -v %CD%/data:/app/data -v %CD%/.refinery:/app/.refinery refinery-pipeline python run_pipeline.py --input data/your_pdfs`
- Override command for ad-hoc runs: `docker run --rm refinery-pipeline python run_pipeline2.py --input data/test_documents`
- Image includes Tesseract; GPU acceleration is not configured in this base image.
