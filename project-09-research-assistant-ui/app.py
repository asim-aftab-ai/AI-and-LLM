"""AI Research Assistant - Streamlit Application.

A professional, minimal web UI built around the existing AI Research Assistant pipeline.
Reuses the underlying PDF extraction, cleaning, semantic chunking, dense vector retrieval,
and grounded LLM question-answering logic from Project 08 without duplicating backend code.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st
from dotenv import load_dotenv

# Set UTF-8 output encoding across terminal environments
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Path Resolution: Reuse Existing Research Assistant Backend Logic
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_08_DIR = CURRENT_DIR.parent / "project-08-ai-pdf-research-assistant"

# Ensure Project 08 is in sys.path to access its existing src modules
if str(PROJECT_08_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_08_DIR))

# Load environment variables from current project, project-08, or workspace root
load_dotenv(dotenv_path=CURRENT_DIR / ".env")
load_dotenv(dotenv_path=PROJECT_08_DIR / ".env")
load_dotenv(dotenv_path=CURRENT_DIR.parent / ".env")

# Import verified existing backend modules
from src.pdf_processor import validate_and_extract_pdf
from src.text_cleaner import clean_text, clean_pages_text
from src.document_processor import chunk_document, save_extracted_txt
from src.retriever import DocumentRetriever, format_context_for_prompt
from src.conversation import ConversationManager
from src.llm import get_llm_client, generate_grounded_answer, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling and Theme Configuration
# ---------------------------------------------------------------------------
def apply_professional_styles():
    """Apply clean, minimal, research-grade typography and UI styling."""
    st.markdown(
        """
        <style>
        /* Main Container layout */
        .block-container {
            max-width: 980px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }

        /* Typography & Header */
        .app-header {
            margin-bottom: 1.5rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
            padding-bottom: 1rem;
        }
        .app-title {
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 0.35rem;
            color: var(--text-color);
        }
        .app-subtitle {
            font-size: 1.0rem;
            color: #64748b;
            line-height: 1.5;
            margin-bottom: 0.8rem;
        }

        /* Status Badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.3rem 0.75rem;
            border-radius: 6px;
            margin-top: 0.3rem;
            margin-bottom: 1rem;
        }
        .status-ready {
            background-color: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .status-empty {
            background-color: rgba(100, 116, 139, 0.12);
            color: #94a3b8;
            border: 1px solid rgba(100, 116, 139, 0.25);
        }

        /* Source / Citation Card */
        .source-card {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-left: 3.5px solid #2563eb;
            background: rgba(30, 41, 59, 0.04);
            border-radius: 6px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.85rem;
            font-size: 0.88rem;
        }
        .source-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .source-badge {
            display: inline-block;
            background: rgba(37, 99, 235, 0.12);
            color: #2563eb;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.75rem;
        }
        .source-body {
            color: #334155;
            line-height: 1.6;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.83rem;
            white-space: pre-wrap;
            word-break: break-word;
            background: rgba(255, 255, 255, 0.6);
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            border: 1px solid rgba(148, 163, 184, 0.15);
        }
        @media (prefers-color-scheme: dark) {
            .app-subtitle { color: #94a3b8; }
            .source-card { background: rgba(15, 23, 42, 0.55); border-color: rgba(51, 65, 85, 0.7); border-left-color: #38bdf8; }
            .source-header { color: #94a3b8; }
            .source-badge { background: rgba(56, 189, 248, 0.18); color: #38bdf8; }
            .source-body { color: #cbd5e1; background: rgba(15, 23, 42, 0.8); border-color: rgba(51, 65, 85, 0.6); }
        }

        /* Sidebar Clean Layout */
        .sidebar-section-title {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-top: 1.2rem;
            margin-bottom: 0.6rem;
        }
        .doc-metric {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            font-size: 0.84rem;
            border-bottom: 1px dashed rgba(148, 163, 184, 0.2);
        }
        .doc-metric-label {
            color: #64748b;
        }
        .doc-metric-value {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Model & Retriever Initialization
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_retriever(model_name: str, api_key: str) -> DocumentRetriever:
    """Create and cache the document retriever to avoid reloading vector models."""
    return DocumentRetriever(model_name=model_name, api_key=api_key)


# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
def init_session_state():
    """Initialize persistent Streamlit session state."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationManager()
    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False
    if "current_filename" not in st.session_state:
        st.session_state.current_filename = ""
    if "cleaned_text" not in st.session_state:
        st.session_state.cleaned_text = ""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "doc_stats" not in st.session_state:
        st.session_state.doc_stats = {}


def resolve_api_key() -> str:
    """Obtain OpenRouter API key from secrets, environment, or user state."""
    # 1. Streamlit secrets (local .streamlit/secrets.toml)
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            val = str(st.secrets["OPENROUTER_API_KEY"]).strip()
            if val and val != "your_openrouter_api_key_here":
                return val
    except Exception:
        pass

    # 2. Environment variables
    env_val = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_val and env_val != "your_openrouter_api_key_here":
        return env_val

    # 3. User session override
    return st.session_state.get("custom_api_key", "").strip()


# ---------------------------------------------------------------------------
# Document Pipeline Execution (Using Existing Backend Logic)
# ---------------------------------------------------------------------------
def process_uploaded_document(uploaded_file, retriever: DocumentRetriever) -> bool:
    """Execute the existing research pipeline: validate, extract, clean, chunk, and index."""
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    # Step 1: Validate and extract text using existing pdf_processor
    extraction_result = validate_and_extract_pdf(file_bytes, filename)
    if not extraction_result["success"]:
        st.sidebar.error(extraction_result["error"])
        return False

    # Step 2: Clean and normalize text using existing text_cleaner
    raw_full_text = extraction_result["full_text"]
    cleaned_full_text = clean_text(raw_full_text)
    cleaned_pages = clean_pages_text(extraction_result["pages_text"])

    # Step 3: Chunk document using existing document_processor
    chunks = chunk_document(cleaned_pages, chunk_size_words=350, overlap_words=50)

    # Optional: Save extracted text to data/extracted for persistence
    output_dir = PROJECT_08_DIR / "data" / "extracted"
    try:
        txt_path = save_extracted_txt(cleaned_full_text, filename, output_dir=str(output_dir))
    except Exception:
        txt_path = ""

    # Step 4: Index chunks into semantic vector retriever
    retriever.index_chunks(chunks)

    # Step 5: Update session state
    st.session_state.document_loaded = True
    st.session_state.current_filename = filename
    st.session_state.cleaned_text = cleaned_full_text
    st.session_state.chunks = chunks
    st.session_state.doc_stats = {
        "filename": filename,
        "page_count": extraction_result["page_count"],
        "word_count": len(cleaned_full_text.split()),
        "character_count": len(cleaned_full_text),
        "chunk_count": len(chunks),
        "txt_path": txt_path,
    }

    # Reset conversation when a new document is loaded
    st.session_state.conversation.clear()
    return True


# ---------------------------------------------------------------------------
# Sidebar Rendering
# ---------------------------------------------------------------------------
def render_sidebar(api_key: str):
    """Render the sidebar containing file upload, document statistics, and settings."""
    st.sidebar.markdown("### Document Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload a research PDF",
        type=["pdf"],
        help="Upload a research paper, article, or report in PDF format.",
        label_visibility="collapsed",
    )

    # Document details card when active
    if st.session_state.document_loaded:
        stats = st.session_state.doc_stats
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 6px; padding: 0.75rem; margin-top: 0.5rem; margin-bottom: 0.8rem;">
                <div style="font-weight: 700; font-size: 0.85rem; color: #10b981; margin-bottom: 0.4rem;">
                    Active Document
                </div>
                <div class="doc-metric"><span class="doc-metric-label">Filename:</span><span class="doc-metric-value">{stats.get('filename')}</span></div>
                <div class="doc-metric"><span class="doc-metric-label">Pages:</span><span class="doc-metric-value">{stats.get('page_count')}</span></div>
                <div class="doc-metric"><span class="doc-metric-label">Word Count:</span><span class="doc-metric-value">{stats.get('word_count'):,}</span></div>
                <div class="doc-metric"><span class="doc-metric-label">Indexed Chunks:</span><span class="doc-metric-value">{stats.get('chunk_count')}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_dl, col_clr = st.sidebar.columns(2)
        with col_dl:
            st.download_button(
                label="Download Text",
                data=st.session_state.cleaned_text,
                file_name=f"{Path(stats['filename']).stem}_extracted.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download cleaned text extracted from this document.",
            )
        with col_clr:
            if st.button("Clear Chat", use_container_width=True, help="Clear message history."):
                st.session_state.conversation.clear()
                st.rerun()

        if st.sidebar.button("Reset Document", use_container_width=True, help="Unload document and reset workspace."):
            st.session_state.document_loaded = False
            st.session_state.current_filename = ""
            st.session_state.cleaned_text = ""
            st.session_state.chunks = []
            st.session_state.doc_stats = {}
            st.session_state.conversation.clear()
            st.rerun()

    else:
        st.sidebar.info("Upload a PDF above to begin research and analysis.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Assistant Settings")

    # LLM Model Selector (Options supported by existing backend)
    llm_options = [
        "google/gemini-2.5-flash",
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "nvidia/nemotron-3.5-lightning:free",
    ]
    selected_llm = st.sidebar.selectbox(
        "Language Model",
        options=llm_options,
        index=0,
        help="Select the foundation model used to formulate grounded answers.",
    )

    # Embedding Model Selector (Options supported by existing retriever)
    embedding_options = [
        "openai/text-embedding-3-small",
        "baai/bge-large-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    selected_embed = st.sidebar.selectbox(
        "Embedding Model",
        options=embedding_options,
        index=0,
        help="Model used to encode document chunks and queries into semantic vectors.",
    )

    # Retrieval Configuration (Directly supported by retriever.retrieve top_k)
    top_k = st.sidebar.slider(
        "Retrieved Sources (Top-K)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Number of relevant document excerpts to retrieve for each answer.",
    )

    # Multi-turn Context Depth
    max_history_turns = st.sidebar.slider(
        "Conversation History Depth",
        min_value=1,
        max_value=6,
        value=4,
        step=1,
        help="Number of previous question-answer exchanges passed as context.",
    )

    # API Key Configuration section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### API Configuration")
    has_api_key = bool(api_key)

    if has_api_key:
        st.sidebar.success("OpenRouter API key connected")
    else:
        st.sidebar.warning("No OpenRouter API key found")

    custom_key = st.sidebar.text_input(
        "Override OpenRouter API Key",
        type="password",
        value=st.session_state.get("custom_api_key", ""),
        placeholder="sk-or-v1-...",
        help="Enter an API key to override the environment or secrets configuration.",
    )
    if custom_key != st.session_state.get("custom_api_key", ""):
        st.session_state.custom_api_key = custom_key
        st.rerun()

    return uploaded_file, selected_llm, selected_embed, top_k, max_history_turns


# ---------------------------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------------------------
def main():
    apply_professional_styles()
    init_session_state()

    # 1. Resolve API key & sidebar configuration
    active_api_key = resolve_api_key()
    uploaded_file, selected_llm, selected_embed, top_k, max_history_turns = render_sidebar(active_api_key)

    # 2. Initialize cached DocumentRetriever
    retriever = get_cached_retriever(model_name=selected_embed, api_key=active_api_key)

    # 3. Handle document upload & indexing
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.current_filename:
            with st.spinner("Processing document: Extracting text, cleaning, and indexing chunks..."):
                success = process_uploaded_document(uploaded_file, retriever)
                if success:
                    st.session_state.indexed_embed_model = selected_embed
                    st.rerun()

    # 3b. Ensure retriever is always synchronized with active session chunks
    if st.session_state.get("document_loaded") and st.session_state.get("chunks"):
        needs_indexing = (
            not getattr(retriever, "chunks", None)
            or retriever.chunk_embeddings is None
            or len(retriever.chunks) != len(st.session_state.chunks)
            or st.session_state.get("indexed_embed_model") != selected_embed
        )
        if needs_indexing:
            with st.spinner("Indexing document chunks for semantic search..."):
                retriever.index_chunks(st.session_state.chunks)
                st.session_state.indexed_embed_model = selected_embed

    # 4. Main Header Area
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">Research Assistant</div>
            <div class="app-subtitle">
                Ask questions about your research documents and inspect the sources behind each answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Document Status Banner
    if st.session_state.document_loaded:
        stats = st.session_state.doc_stats
        st.markdown(
            f'<div class="status-badge status-ready">'
            f'<span>✓</span> Active Document: <strong>{stats["filename"]}</strong> '
            f'({stats["page_count"]} pages, {stats["chunk_count"]} chunks indexed)'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-empty">'
            '<span>ℹ</span> No document loaded. Upload a PDF from the sidebar to begin.'
            '</div>',
            unsafe_allow_html=True,
        )

    # 5. Conversation History Display
    messages = st.session_state.conversation.get_all_messages()
    for msg in messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

            # Render expandable sources if attached to the assistant message
            citations = msg.get("citations")
            if role == "assistant" and citations:
                citation_count = len(citations)
                with st.expander(f"Sources / Retrieved Context ({citation_count} excerpts)", expanded=False):
                    doc_name = st.session_state.get("current_filename", "Document")
                    for idx, cite in enumerate(citations, start=1):
                        page_num = cite.get("page_num", "Unknown")
                        chunk_id = cite.get("chunk_id", idx)
                        relevance = cite.get("score")
                        word_count = cite.get("word_count", len(cite.get("text", "").split()))

                        score_badge = f"{relevance:.2f}" if relevance is not None else "N/A"

                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-header">
                                    <span>Source Excerpt {idx} &bull; {doc_name} &bull; Page {page_num} (Chunk {chunk_id})</span>
                                    <span class="source-badge">Relevance: {score_badge} | {word_count} words</span>
                                </div>
                                <div class="source-body">{cite.get('text', '').strip()}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # 6. Chat Input Area
    input_placeholder = (
        "Ask a question about the document..."
        if st.session_state.document_loaded
        else "Please upload a research PDF in the sidebar first..."
    )
    user_input = st.chat_input(input_placeholder)

    if user_input:
        query_text = user_input.strip()
        if not query_text:
            return

        # Validation: Check if a document is loaded
        if not st.session_state.document_loaded:
            st.error("Please upload a PDF document before asking questions.")
            return

        # Validation: Check if an API key is available
        if not active_api_key:
            st.error(
                "An OpenRouter API key is required to formulate answers. "
                "Please configure OPENROUTER_API_KEY in `.streamlit/secrets.toml` or provide it in the sidebar."
            )
            return

        # Step A: Add user question to conversation history
        st.session_state.conversation.add_user_message(query_text)

        # Step B: Retrieve relevant context using existing retriever
        with st.spinner("Searching document for relevant context..."):
            if st.session_state.get("chunks") and (
                not getattr(retriever, "chunks", None)
                or retriever.chunk_embeddings is None
                or len(retriever.chunks) != len(st.session_state.chunks)
            ):
                retriever.index_chunks(st.session_state.chunks)
            retrieved_chunks = retriever.retrieve(query_text, top_k=top_k)
            context_text = format_context_for_prompt(retrieved_chunks)

        # Step C: Retrieve conversation history for multi-turn grounding
        history = st.session_state.conversation.get_messages_for_llm(max_turns=max_history_turns)

        # Step D: Generate grounded answer via existing LLM module
        with st.spinner("Researching and formulating answer..."):
            try:
                client = get_llm_client(active_api_key)
                answer = generate_grounded_answer(
                    client=client,
                    model=selected_llm,
                    question=query_text,
                    context_text=context_text,
                    conversation_history=history,
                )
                # Step E: Record assistant response along with real retrieved citations
                st.session_state.conversation.add_assistant_message(
                    answer,
                    citations=retrieved_chunks,
                )
            except Exception as e:
                error_message = f"An error occurred while communicating with the model: {str(e)}"
                st.session_state.conversation.add_assistant_message(error_message)

        st.rerun()


if __name__ == "__main__":
    main()
