"""AI PDF Research Assistant - Streamlit Application.

Combines PDF text extraction, cleaning, semantic retrieval, grounded LLM question answering,
and multi-turn conversational history in a single cohesive local application.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure local src module is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_processor import validate_and_extract_pdf
from src.text_cleaner import clean_text, clean_pages_text
from src.document_processor import chunk_document, save_extracted_txt
from src.retriever import DocumentRetriever, format_context_for_prompt
from src.conversation import ConversationManager
from src.llm import get_llm_client, generate_grounded_answer, DEFAULT_MODEL

# Load local environment variables if available
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI PDF Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure UTF-8 output encoding across environments
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


@st.cache_resource(show_spinner=False)
def load_cached_retriever(model_name: str = "openai/text-embedding-3-small", api_key: str = "") -> DocumentRetriever:
    """Initialize and cache the semantic retriever."""
    return DocumentRetriever(model_name=model_name, api_key=api_key)


def apply_custom_css():
    """Apply professional, minimalist styling with strong visual hierarchy."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1050px;
            padding-top: 2rem;
            padding-bottom: 3.5rem;
        }
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        .main-subtitle {
            font-size: 1rem;
            color: #94a3b8;
            margin-bottom: 1.5rem;
        }
        .status-badge {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.25rem 0.65rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        .status-ready {
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .status-empty {
            background: rgba(148, 163, 184, 0.12);
            color: #94a3b8;
            border: 1px solid #334155;
        }
        .citation-box {
            border-left: 3px solid #0284c7;
            background: rgba(15, 23, 42, 0.5);
            padding: 0.6rem 0.8rem;
            margin: 0.4rem 0;
            font-size: 0.82rem;
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """Set up initial Streamlit session state objects."""
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


def get_api_key() -> str:
    """Retrieve the OpenRouter API key from secrets, environment, or sidebar input."""
    # 1. Check Streamlit secrets
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            val = st.secrets["OPENROUTER_API_KEY"].strip()
            if val and val != "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
                return val
    except Exception:
        pass

    # 2. Check environment variables
    env_val = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_val and env_val != "your_openrouter_api_key_here":
        return env_val

    # 3. Check session state (from sidebar input)
    return st.session_state.get("user_provided_api_key", "").strip()


def process_uploaded_pdf(uploaded_file, retriever: DocumentRetriever):
    """Execute the full extraction, cleaning, saving, and indexing pipeline."""
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    # Step 1: Validate and extract text
    extraction_result = validate_and_extract_pdf(file_bytes, filename)
    if not extraction_result["success"]:
        st.error(extraction_result["error"])
        return False

    # Step 2: Clean and normalize text
    raw_text = extraction_result["full_text"]
    cleaned_full_text = clean_text(raw_text)
    cleaned_pages = clean_pages_text(extraction_result["pages_text"])

    # Step 3: Save clean text as .txt file on disk
    txt_path = save_extracted_txt(cleaned_full_text, filename, output_dir="data/extracted")

    # Step 4: Chunk document for retrieval
    chunks = chunk_document(cleaned_pages, chunk_size_words=350, overlap_words=50)

    # Step 5: Index chunks in semantic retriever
    retriever.index_chunks(chunks)

    # Update session state
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

    # Reset conversation history for the new document
    st.session_state.conversation.clear()
    return True


def render_sidebar(api_key_configured: bool):
    """Render the sidebar controls and document metadata."""
    st.sidebar.title("Document Manager")

    # Upload section inside sidebar
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Select a PDF file to process and chat with.",
    )

    st.sidebar.markdown("---")

    # Display active document statistics if loaded
    if st.session_state.document_loaded:
        stats = st.session_state.doc_stats
        st.sidebar.markdown("**Active Document**")
        st.sidebar.text(f"File: {stats.get('filename')}")
        st.sidebar.text(f"Pages: {stats.get('page_count')}")
        st.sidebar.text(f"Words: {stats.get('word_count'):,}")
        st.sidebar.text(f"Chunks: {stats.get('chunk_count')}")

        # Download extracted text button
        st.sidebar.download_button(
            label="Download Cleaned Text (.txt)",
            data=st.session_state.cleaned_text,
            file_name=f"{os.path.splitext(stats['filename'])[0]}_extracted.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Clear conversation button
        if st.sidebar.button("Clear Conversation", use_container_width=True):
            st.session_state.conversation.clear()
            st.rerun()

    else:
        st.sidebar.info("No document loaded. Upload a PDF above to begin.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Settings**")

    llm_options = [
        "google/gemini-2.5-flash",
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "nvidia/nemotron-3.5-lightning:free",
    ]
    selected_llm = st.sidebar.selectbox(
        "LLM Provider Model",
        options=llm_options,
        index=0,
    )

    embedding_options = [
        "openai/text-embedding-3-small",
        "baai/bge-large-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    selected_embed = st.sidebar.selectbox(
        "Embedding Model",
        options=embedding_options,
        index=0,
        help="Embedding model used for semantic document search.",
    )

    if not api_key_configured:
        st.sidebar.markdown("**API Configuration**")
        user_key = st.sidebar.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="sk-or-v1-...",
            help="Enter your OpenRouter API key to enable answers.",
        )
        if user_key:
            st.session_state.user_provided_api_key = user_key
            st.rerun()

    return uploaded_file, selected_llm, selected_embed


def main():
    apply_custom_css()
    initialize_session_state()

    # Determine API key availability
    active_api_key = get_api_key()
    api_key_configured = bool(active_api_key)

    # Render sidebar controls
    uploaded_file, selected_model, selected_embed = render_sidebar(api_key_configured)

    # Load cached semantic retriever with selected embedding model
    retriever = load_cached_retriever(model_name=selected_embed, api_key=active_api_key)

    # Process new PDF upload if detected
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.current_filename:
            with st.spinner("Processing PDF: Extracting, cleaning, and indexing document..."):
                success = process_uploaded_pdf(uploaded_file, retriever)
                if success:
                    st.rerun()

    # Main Area Header
    st.markdown('<div class="main-title">AI PDF Research Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Upload a PDF and ask questions grounded strictly in its contents.</div>',
        unsafe_allow_html=True,
    )

    # Document Status Banner
    if st.session_state.document_loaded:
        stats = st.session_state.doc_stats
        st.markdown(
            f'<div class="status-badge status-ready">Document Ready: {stats["filename"]} ({stats["page_count"]} pages, {stats["chunk_count"]} chunks)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-empty">No Document Loaded: Please upload a PDF in the sidebar</div>',
            unsafe_allow_html=True,
        )

    # Warning if API key is missing
    if not api_key_configured:
        st.warning(
            "OpenRouter API key not detected. Please configure OPENROUTER_API_KEY in "
            ".streamlit/secrets.toml, an environment variable, or enter it in the sidebar."
        )

    # Display Conversation History
    messages = st.session_state.conversation.get_all_messages()
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display source citations if attached to the assistant response
            if "citations" in msg and msg["citations"]:
                with st.expander(f"View Document Context ({len(msg['citations'])} excerpts retrieved)", expanded=False):
                    for cite in msg["citations"]:
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <strong>Page {cite.get('page_num', '?')} | Chunk {cite.get('chunk_id', '?')}</strong> (Relevance: {cite.get('score', 0.0):.2f})<br>
                                {cite.get('text', '')}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat Input Box
    user_question = st.chat_input(
        "Ask a question about the document..."
        if st.session_state.document_loaded
        else "Upload a PDF first to begin asking questions..."
    )

    if user_question:
        # Validation 1: Document must be loaded
        if not st.session_state.document_loaded:
            st.error("Please upload a PDF document before asking questions.")
            return

        # Validation 2: API key must be available
        if not api_key_configured:
            st.error("An OpenRouter API key is required to generate answers.")
            return

        # Validation 3: Question must not be empty
        cleaned_query = user_question.strip()
        if not cleaned_query:
            return

        # 1. Record user question
        st.session_state.conversation.add_user_message(cleaned_query)

        # 2. Retrieve relevant chunks
        with st.spinner("Finding relevant document context..."):
            retrieved_chunks = retriever.retrieve(cleaned_query, top_k=3)
            context_text = format_context_for_prompt(retrieved_chunks)

        # 3. Retrieve prior turns for multi-turn conversational context
        history = st.session_state.conversation.get_messages_for_llm(max_turns=4)

        # 4. Generate grounded answer via OpenRouter
        with st.spinner("Generating answer from document context..."):
            try:
                client = get_llm_client(active_api_key)
                answer = generate_grounded_answer(
                    client=client,
                    model=selected_model,
                    question=cleaned_query,
                    context_text=context_text,
                    conversation_history=history,
                )
                # Record assistant answer with citations
                st.session_state.conversation.add_assistant_message(answer, citations=retrieved_chunks)
            except Exception as e:
                error_msg = f"Failed to generate response: {str(e)}"
                st.session_state.conversation.add_assistant_message(error_msg)

        st.rerun()


if __name__ == "__main__":
    main()
