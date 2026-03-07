# streamlit_rag_demo.py
import json
import shutil
from pathlib import Path

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.utils.vector_store import SimpleVectorStore

# ---------------- Directories ----------------
REFINERY_DIR = Path(".refinery")
UPLOADS_DIR = REFINERY_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Session State ----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = SimpleVectorStore()

if "query_agent" not in st.session_state:
    st.session_state.query_agent = QueryAgent(
        vector_store=st.session_state.vector_store,
        pageindex_dir=REFINERY_DIR / "pageindex",
        fact_db_path=str(REFINERY_DIR / "facts" / "facts.db"),
    )
if "session_history" not in st.session_state:
    st.session_state.session_history = []

# ---------------- Helper Functions ----------------
def process_uploaded_pdf(pdf_file):
    if pdf_file is None:
        return "No file uploaded.", None, None, None

    # Save PDF
    pdf_path = UPLOADS_DIR / pdf_file.name
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    # Step 1: Triage
    triage = TriageAgent()
    profile = triage.profile(pdf_path)

    # Step 2: Extraction
    router = ExtractionRouter()
    extracted = router.route(pdf_path, profile)

    chunker = ChunkingEngine()
    ldus = chunker.chunk(extracted)
    for ldu in ldus:
        st.session_state.vector_store.add(
            ldu.content,
            {
                "doc_id": extracted.doc_id,
                "page_number": ldu.page_refs[0] if ldu.page_refs else 0,
                "content": ldu.content,
            },
    )
        
    indexer = PageIndexBuilder()
    page_index = indexer.build_index(extracted)

    # Persist vector store (JSON placeholder)
    vector_store_dir = REFINERY_DIR / "vector_store"
    vector_store_dir.mkdir(exist_ok=True)
    vector_path = vector_store_dir / f"{extracted.doc_id}.json"
    payloads = [{"doc_id": extracted.doc_id, "content": ldu.content, "page_refs": ldu.page_refs, "bounding_box": ldu.bounding_box, "content_hash": ldu.content_hash} for ldu in ldus]
    with vector_path.open("w", encoding="utf-8") as f:
        json.dump(payloads, f, ensure_ascii=False, indent=2)

    # Persist facts
    facts_dir = REFINERY_DIR / "facts"
    facts_dir.mkdir(exist_ok=True)
    fact_db_path = facts_dir / f"{extracted.doc_id}.db"
    st.session_state.query_agent.ingest_facts(extracted)

    return f"PDF processed: {pdf_file.name}", profile, extracted, page_index

def ask_question(query):
    if not query.strip():
        return "", [], st.session_state.session_history

    result = st.session_state.query_agent.answer(query)
    st.session_state.session_history.append({"question": query, "answer": result.answer})

    provenance_table = [
        [p.document_name, p.page_number, p.bounding_box, p.content_hash]
        for p in result.provenance
    ]
    return result.answer, provenance_table, st.session_state.session_history

def display_pdf(pdf_path):
    try:
        reader = PdfReader(str(pdf_path))
        text_preview = "\n\n".join(page.extract_text()[:500] for page in reader.pages)
        st.text_area("PDF Text Preview (first 500 chars per page)", text_preview, height=300)
    except Exception:
        st.warning("Cannot preview PDF content")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Refinery RAG Demo", layout="wide")
st.title("Refinery RAG Demo - 4 Step Pipeline")

tabs = st.tabs(["Step 1: Triage", "Step 2: Extraction", "Step 3: PageIndex", "Step 4: Query"])

# ---------------- Step 1: Triage ----------------
with tabs[0]:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    use_vlm_first = st.checkbox(
        "Skip OCR and use VLM directly (Qwen/Gemini)", value=False
    )
    if st.button("Process PDF for Triage"):
        status_msg, profile, extracted, page_index = process_uploaded_pdf(uploaded_file)
        st.success(status_msg)
        if profile:
            st.subheader("Document Profile (Triage)")
            st.json(profile.model_dump())
            st.subheader("Extraction Strategy Plan")
            st.text(extracted.extractor)

# ---------------- Step 2: Extraction ----------------
with tabs[1]:
    if 'extracted' in locals() and extracted:
        st.subheader("Original PDF")
        display_pdf(UPLOADS_DIR / uploaded_file.name)
        st.subheader("Extracted JSON Chunks")
        chunks_preview = [{"content": ldu.content, "chunk_type": ldu.chunk_type, "page_refs": ldu.page_refs} for ldu in ChunkingEngine().chunk(extracted)]
        st.json(chunks_preview)

        st.subheader("Extraction Ledger (last 5 entries)")
        ledger_path = REFINERY_DIR / "extraction_ledger.jsonl"
        if ledger_path.exists():
            with open(ledger_path) as f:
                lines = f.readlines()[-5:]
            st.json([json.loads(l) for l in lines])
    else:
        st.info("Upload and process a PDF in Step 1 first.")

# ---------------- Step 3: PageIndex ----------------
with tabs[2]:
    if 'page_index' in locals() and page_index:
        st.subheader("PageIndex Tree")
        st.code(json.dumps(page_index.model_dump(), indent=2))
    else:
        st.info("Process a PDF in Step 1 first.")

# ---------------- Step 4: Query ----------------
with tabs[3]:
    query_input = st.text_input("Enter your question:")
    if st.button("Ask Question"):
        answer, provenance, chat_history = ask_question(query_input)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Provenance")
        st.dataframe(pd.DataFrame(provenance, columns=["Document", "Page", "Bounding Box", "Content Hash"]))

        st.subheader("Chat History")
        for turn in chat_history:
            st.markdown(f"**Q:** {turn['question']}\n**A:** {turn['answer']}")