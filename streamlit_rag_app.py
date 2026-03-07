"""
Refinery Pipeline – Streamlit Demonstration Dashboard
Tabs: Documents | Pipeline | Query | Audit | Debug
This UI visualizes the end-to-end capabilities of the backend pipeline without
modifying backend logic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.agents.triage import TriageAgent
from src.utils.vector_store import SimpleVectorStore

# -----------------------------------------------------------------------------#
# Global configuration and session state
# -----------------------------------------------------------------------------#
st.set_page_config(page_title="Refinery Document Intelligence", layout="wide")

REFINERY_DIR = Path(".refinery")
UPLOADS_DIR = REFINERY_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
FACTS_DIR = REFINERY_DIR / "facts"
FACTS_DIR.mkdir(parents=True, exist_ok=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = SimpleVectorStore()

if "query_agent" not in st.session_state:
    st.session_state.query_agent = QueryAgent(
        vector_store=st.session_state.vector_store,
        pageindex_dir=REFINERY_DIR / "pageindex",
        fact_db_path=str(FACTS_DIR / "facts.db"),
    )

if "pipeline_outputs" not in st.session_state:
    st.session_state.pipeline_outputs = {}

if "last_query" not in st.session_state:
    st.session_state.last_query = {"question": "", "result": None, "retrieved": []}

# -----------------------------------------------------------------------------#
# Helper functions
# -----------------------------------------------------------------------------#
def upload_document():
    """Upload PDF and display basic file info."""
    st.subheader("Upload Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    info = {}
    if uploaded:
        uploaded.seek(0)
        info["name"] = uploaded.name
        info["size_mb"] = round(len(uploaded.getbuffer()) / (1024 * 1024), 2)
        try:
            reader = PdfReader(uploaded)
            info["pages"] = len(reader.pages)
        except Exception:
            info["pages"] = "Unknown"
        st.info(f"Selected: **{info['name']}** · {info['size_mb']} MB · {info['pages']} pages")
    return uploaded, info


def run_full_pipeline(uploaded_file):
    """Run triage → extraction → chunking → indexing → storage."""
    start = time.perf_counter()
    pdf_path = UPLOADS_DIR / uploaded_file.name
    uploaded_file.seek(0)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    triage_agent = TriageAgent()
    extractor = ExtractionRouter()
    chunker = ChunkingEngine()
    indexer = PageIndexBuilder()

    outputs = {"doc_path": pdf_path}

    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Triage classification")
        profile = triage_agent.profile(pdf_path)
        outputs["profile"] = profile

        st.write("Extraction (strategy routed)")
        extracted = extractor.route(pdf_path, profile)
        outputs["extracted"] = extracted

        st.write("Chunking")
        ldus = chunker.chunk(extracted)
        outputs["ldus"] = ldus

        st.write("Indexing & storage")
        page_index = indexer.build_index(extracted)
        outputs["page_index"] = page_index

        # Vector store ingestion with provenance metadata
        for ldu in ldus:
            st.session_state.vector_store.add(
                ldu.content,
                {
                    "doc_id": extracted.doc_id,
                    "page_number": ldu.page_refs[0] if ldu.page_refs else 0,
                    "bounding_box": ldu.bounding_box,
                    "content_hash": ldu.content_hash,
                    "parent_section": ldu.parent_section,
                    "content": ldu.content,
                },
            )

        vector_store_dir = REFINERY_DIR / "vector_store"
        vector_store_dir.mkdir(exist_ok=True)
        vector_path = vector_store_dir / f"{extracted.doc_id}.json"
        payloads = [
            {
                "doc_id": extracted.doc_id,
                "content": ldu.content,
                "page_refs": ldu.page_refs,
                "bounding_box": ldu.bounding_box,
                "content_hash": ldu.content_hash,
                "parent_section": ldu.parent_section,
            }
            for ldu in ldus
        ]
        vector_path.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")

        # Fact ingestion
        FACTS_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state.query_agent.ingest_facts(extracted)

        outputs["processing_time_s"] = round(time.perf_counter() - start, 2)
        outputs["vector_path"] = vector_path
        outputs["chunks_count"] = len(ldus)
        outputs["pages_count"] = len(extracted.pages)
        status.update(label="Pipeline complete", state="complete")

    st.session_state.pipeline_outputs = outputs
    return outputs


def pipeline_panel():
    """Pipeline status visualization."""
    outputs = st.session_state.pipeline_outputs
    if not outputs:
        st.info("Run the pipeline in the Documents tab first.")
        return

    stages = ["Triage", "Extraction", "Chunking", "Indexing", "Storage"]
    cols = st.columns(len(stages))
    for col, stage in zip(cols, stages):
        col.metric(stage, "Complete", delta="✓")

    st.progress(100, text="Pipeline finished")
    st.caption(
        f"Pages processed: {outputs.get('pages_count', 0)} · "
        f"Chunks generated: {outputs.get('chunks_count', 0)} · "
        f"Vector index: {outputs.get('vector_path', 'n/a')}"
    )


def query_interface():
    """Question answering interface."""
    st.subheader("Semantic Query")
    question = st.text_input("Ask a question about the uploaded document...", key="query_input")
    if st.button("Submit Query"):
        if not st.session_state.pipeline_outputs:
            st.warning("Please run the pipeline first.")
            return

        with st.spinner("Running QueryAgent..."):
            result = st.session_state.query_agent.answer(question)
            st.session_state.last_query = {"question": question, "result": result}
            # Capture retrieved chunks for debug/provenance view
            retrieved = st.session_state.vector_store.search(question, top_k=5)
            st.session_state.last_query["retrieved"] = retrieved

        st.success("Answer generated")
        st.markdown(f"### Answer\n{result.answer}")
        display_provenance(result)


def display_provenance(result):
    """Provenance display with expanders."""
    st.subheader("Provenance")
    if not result or not result.provenance:
        st.info("No provenance available.")
        return

    for i, p in enumerate(result.provenance, 1):
        with st.expander(f"Source {i}: {p.document_name} · Page {p.page_number}"):
            st.markdown(
                f"- **Document:** {p.document_name}\n"
                f"- **Page:** {p.page_number}\n"
                f"- **Bounding Box:** {p.bounding_box}\n"
                f"- **Content Hash:** `{p.content_hash}`"
            )
            # Locate matching chunk for preview
            preview = ""
            for chunk in st.session_state.last_query.get("retrieved", []):
                if p.content_hash and chunk.get("content_hash") == p.content_hash:
                    preview = chunk.get("content", "")
                    break
                if chunk.get("page_number") == p.page_number:
                    preview = chunk.get("content", "")
                    break
            if preview:
                st.write("**View Retrieved Chunk**")
                st.code(preview, language="markdown")


def audit_verification():
    """Audit mode: verify claims with provenance."""
    st.subheader("Audit Mode")
    claim = st.text_input("Enter a claim to verify", placeholder="Company revenue increased in 2023")
    if st.button("Verify Claim"):
        if not st.session_state.pipeline_outputs:
            st.warning("Run the pipeline first.")
            return
        with st.spinner("Auditing claim..."):
            result = st.session_state.query_agent.audit_claim(claim)
            verdict = "Verified with citation" if result.audited else "Unverifiable"
        st.markdown(f"### Result: {verdict}")
        display_provenance(result)


def debug_panel():
    """Expose debug information for grading or troubleshooting."""
    st.subheader("Debug / Trace")
    last = st.session_state.last_query
    retrieved = last.get("retrieved", [])
    st.write(f"Retrieved chunks: {len(retrieved)}")
    with st.expander("Retrieved Chunks"):
        if retrieved:
            st.dataframe(pd.DataFrame(retrieved))
        else:
            st.info("No chunks retrieved yet.")

    with st.expander("Tool / Agent Used"):
        st.write("Semantic search via QueryAgent + SimpleVectorStore")

    with st.expander("LLM Prompt Preview"):
        st.code("Prompt constructed inside QueryAgent._synthesize_answer", language="text")

    with st.expander("Response Metadata"):
        result = last.get("result")
        if result:
            st.json({"audited": result.audited, "facts_used": result.facts_used})
        else:
            st.info("No responses yet.")


def documents_tab():
    """Documents tab: upload + run pipeline."""
    uploaded, info = upload_document()
    if uploaded:
        st.divider()
        cols = st.columns(3)
        cols[0].metric("File Size (MB)", info.get("size_mb", 0))
        cols[1].metric("Pages", info.get("pages", "Unknown"))
        cols[2].metric("Status", "Ready for processing")

        if st.button("Run Pipeline", type="primary"):
            outputs = run_full_pipeline(uploaded)
            st.success(
                f"Processed {info.get('name')} in {outputs.get('processing_time_s', 0)}s "
                f"using strategy {outputs['extracted'].extractor}"
            )
            with st.expander("Triage Classification"):
                st.json(outputs["profile"].model_dump())


def pipeline_tab():
    st.subheader("Pipeline Visualization")
    pipeline_panel()


def query_tab():
    query_interface()


def audit_tab():
    audit_verification()


def debug_tab():
    debug_panel()


# -----------------------------------------------------------------------------#
# Layout
# -----------------------------------------------------------------------------#
st.title("Refinery Pipeline Dashboard")
tabs = st.tabs(["Documents", "Pipeline", "Query", "Audit", "Debug"])

with tabs[0]:
    documents_tab()
with tabs[1]:
    pipeline_tab()
with tabs[2]:
    query_tab()
with tabs[3]:
    audit_tab()
with tabs[4]:
    debug_tab()
