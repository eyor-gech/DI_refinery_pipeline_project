"""
CLI entrypoint for Refinery Pipeline.

Steps:
1) Triage Agent -> DocumentProfile
2) ExtractionRouter -> ExtractedDocument with ledger logging
3) ChunkingEngine -> LDUs (persist)
4) PageIndexBuilder -> JSON index
5) Ingest chunks into a lightweight vector store placeholder
6) Build FactTable from numeric facts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from pypdf import PdfReader, PdfWriter

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent


# ------------------------------------------------------------------
# Utility writers
# ------------------------------------------------------------------

def write_full_text(doc, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    text_path = out_dir / f"{doc.doc_id}.txt"
    full_text = "\n\n".join(page.full_text for page in doc.pages)

    text_path.write_text(full_text, encoding="utf-8")


def persist_chunks(ldus, out_dir: Path, doc_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{doc_id}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            [c.model_dump() for c in ldus],
            f,
            ensure_ascii=False,
            indent=2,
        )


def ingest_vector_store(ldus, out_dir: Path, doc_id: str) -> None:
    """
    Placeholder vector store ingestion.
    Writes chunk payloads to JSON for later indexing by FAISS/Chroma/etc.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = []

    for ldu in ldus:
        payloads.append(
            {
                "doc_id": doc_id,
                "content": ldu.content,
                "chunk_type": ldu.chunk_type,
                "page_refs": ldu.page_refs,
                "bounding_box": ldu.bounding_box,
                "content_hash": ldu.content_hash,
                "parent_section": ldu.parent_section,
                "metadata": ldu.metadata,
            }
        )

    path = out_dir / f"{doc_id}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(payloads, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# PDF splitting
# ------------------------------------------------------------------

def split_large_pdf(pdf_path: Path) -> List[Path]:

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    if total_pages < 15:
        return [pdf_path]

    if total_pages < 200:
        parts = 3
    else:
        parts = 4

    pages_per_part = total_pages // parts + 1

    split_dir = pdf_path.parent / "split_parts"
    split_dir.mkdir(exist_ok=True)

    output_files = []

    for part in range(parts):

        start = part * pages_per_part
        end = min(start + pages_per_part, total_pages)

        if start >= total_pages:
            break

        writer = PdfWriter()

        for i in range(start, end):
            writer.add_page(reader.pages[i])

        part_path = split_dir / f"{pdf_path.stem}_part{part+1}.pdf"

        with open(part_path, "wb") as f:
            writer.write(f)

        output_files.append(part_path)

    print(f"Split {pdf_path.name} into {len(output_files)} parts")

    return output_files


# ------------------------------------------------------------------
# Main processing logic
# ------------------------------------------------------------------

def process_pdf(pdf_path: Path, args) -> None:

    triage = TriageAgent()
    profile = triage.profile(pdf_path)

    router = ExtractionRouter()
    extracted = router.route(pdf_path, profile)

    chunker = ChunkingEngine()
    ldus = chunker.chunk(extracted)

    indexer = PageIndexBuilder()
    indexer.build_index(extracted)

    base_dir = Path(".refinery")

    write_full_text(extracted, base_dir / "fulltext")
    persist_chunks(ldus, base_dir / "chunks", extracted.doc_id)
    ingest_vector_store(ldus, base_dir / "vector_store", extracted.doc_id)

    facts_dir = base_dir / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)

    fact_db_path = facts_dir / f"{extracted.doc_id}.db"

    qa = QueryAgent(
        pageindex_dir=base_dir / "pageindex",
        fact_db_path=str(fact_db_path),
    )

    qa.ingest_facts(extracted)

    print(f"Processed {pdf_path.name}")
    print(f"  Profile -> .refinery/profiles/{profile.doc_id}.json")
    print(f"  Ledger -> .refinery/extraction_ledger.jsonl")
    print(f"  Full text -> {base_dir/'fulltext'/(profile.doc_id+'.txt')}")
    print(f"  Chunks -> {base_dir/'chunks'/(profile.doc_id+'.json')}")
    print(f"  PageIndex -> {base_dir/'pageindex'/(profile.doc_id+'.json')}")
    print(f"  FactTable -> {fact_db_path}")
    print(f"  Vector payload -> {base_dir/'vector_store'/(profile.doc_id+'.json')}")


# ------------------------------------------------------------------
# Merge outputs when PDFs were split
# ------------------------------------------------------------------

def merge_pipeline_outputs(original_doc: str, part_ids: List[str]):

    base_dir = Path(".refinery")

    merged_chunks = []
    merged_vectors = []
    merged_text = []

    for pid in part_ids:

        chunk_file = base_dir / "chunks" / f"{pid}.json"
        if chunk_file.exists():
            with chunk_file.open(encoding="utf-8") as f:
                merged_chunks.extend(json.load(f))

        vector_file = base_dir / "vector_store" / f"{pid}.json"
        if vector_file.exists():
            with vector_file.open(encoding="utf-8") as f:
                merged_vectors.extend(json.load(f))

        text_file = base_dir / "fulltext" / f"{pid}.txt"
        if text_file.exists():
            merged_text.append(text_file.read_text(encoding="utf-8"))

    with (base_dir / "chunks" / f"{original_doc}.json").open("w", encoding="utf-8") as f:
        json.dump(merged_chunks, f, indent=2, ensure_ascii=False)

    with (base_dir / "vector_store" / f"{original_doc}.json").open("w", encoding="utf-8") as f:
        json.dump(merged_vectors, f, indent=2, ensure_ascii=False)

    (base_dir / "fulltext" / f"{original_doc}.txt").write_text(
        "\n\n".join(merged_text),
        encoding="utf-8",
    )

    print(f"Merged pipeline outputs for {original_doc}")


# ------------------------------------------------------------------
# PDF discovery
# ------------------------------------------------------------------

def find_pdfs(input_path: Path) -> List[Path]:

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]

    return list(input_path.rglob("*.pdf"))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Run Refinery Pipeline on PDFs.")

    parser.add_argument(
        "--input",
        required=True,
        help="Path to PDF file or directory containing PDFs",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    pdfs = find_pdfs(input_path)

    if not pdfs:
        raise SystemExit(f"No PDFs found at {input_path}")

    for pdf in pdfs:

        pdf_parts = split_large_pdf(pdf)

        part_ids = []

        for part in pdf_parts:
            process_pdf(part, args)
            part_ids.append(part.stem)

        if len(part_ids) > 1:
            merge_pipeline_outputs(pdf.stem, part_ids)


if __name__ == "__main__":
    main()