# Project 09: Professional Streamlit UI for AI Research Assistant

> 🌐 **Live Demo:** [https://research-assistant-rpzr.onrender.com/](https://research-assistant-rpzr.onrender.com/)

A clean, research-grade web application built around the existing AI Research Assistant pipeline. This project wraps the PDF extraction, text cleaning, semantic chunking, dense vector retrieval, and grounded multi-turn conversation engine from **Project 08** into an intuitive user interface without duplicating or rewriting the underlying backend logic.

---

## 1. Project Purpose

The goal of this project is to elevate the AI PDF Research Assistant into a production-style research tool following the core design principle:

$$\text{Conversation} \longrightarrow \text{Answer} \longrightarrow \text{Evidence}$$

The user is never left wondering where to upload a document, where to ask questions, or how to inspect the authentic evidence supporting every answer.

---

## 2. Main UI Components

The application is structured into three dedicated zones:

### A. Sidebar (Document & Configuration Hub)
- **Document Uploader:** Clean drag-and-drop file uploader strictly restricted to supported document formats (`.pdf`).
- **Active Document Panel:** Displays real-time metadata once processed:
  - Filename
  - Page count
  - Word count
  - Number of indexed vector chunks
- **Document Actions:**
  - `Download Text`: Exports the cleaned, normalized document text (`.txt`).
  - `Clear Chat`: Resets message history while preserving the loaded document index.
  - `Reset Document`: Unloads the document and resets session state for a fresh analysis.
- **Assistant Settings:**
  - `Language Model`: Select foundation model (`google/gemini-2.5-flash`, `google/gemini-2.0-flash-exp:free`, `meta-llama/llama-3.3-70b-instruct:free`, `nvidia/nemotron-3.5-lightning:free`).
  - `Embedding Model`: Choose semantic embedding model (`openai/text-embedding-3-small`, `baai/bge-large-en-v1.5`, or offline `sentence-transformers/all-MiniLM-L6-v2`).
  - `Retrieved Sources (Top-K)`: Configure how many relevant document excerpts (1 to 5) are retrieved.
  - `Conversation History Depth`: Slider controlling multi-turn conversational context turns (1 to 6).
- **API Configuration:**
  - Automatic detection from Streamlit secrets or environment variables.
  - Sidebar password input field to easily supply or override the OpenRouter API key on the fly.

### B. Main Research Workspace
- **Header & Subtitle:** Clear, minimal application branding with a concise statement of purpose.
- **Status Indicator Badge:**
  - `Document Ready`: Visual confirmation of active document and index status.
  - `Empty State`: Clear instructions when no document is loaded.
- **Chat History:**
  - Chronologically ordered conversation turns.
  - Distinct styling for user inquiries (`st.chat_message("user")`) and assistant responses (`st.chat_message("assistant")`).
  - Persistent across Streamlit reruns using session state.
- **Chat Input:**
  - Clean bottom input bar (`st.chat_input`) with dynamic contextual placeholder.

### C. Expandable Sources (Evidence Layer)
- Attached to each assistant response when relevant context was retrieved.
- **Collapsed by default** (`Sources / Retrieved Context (N excerpts)`) to keep the chat interface clean and readable.
- Displays authentic, un-fabricated metadata returned by the retrieval engine:
  - Source Filename
  - Page Number (`Page X`)
  - Chunk Identifier (`Chunk Y`)
  - Semantic Relevance Score (Cosine similarity rounded to two decimal places)
  - Word count
  - Complete, un-truncated excerpt text in a clean monospace quotation card.

---

## 3. Architecture & Integration with Existing Backend

This UI directly reuses the existing backend modules from `project-08-ai-pdf-research-assistant` without duplicating code:

```
project-09-research-assistant-ui/
├── app.py                     # Streamlit application UI
├── requirements.txt           # Project dependencies
├── test_ui_integration.py     # Module import & integration test suite
├── test_e2e_research_flow.py  # End-to-end research workflow test
├── .streamlit/
│   ├── config.toml            # Server headless configuration
│   └── secrets.toml           # OpenRouter API key configuration
└── README.md                  # Project documentation
```

### Backend Integration Mapping

| Operation | Existing Project 08 Module | Function / Class Reused |
| :--- | :--- | :--- |
| **PDF Extraction** | `src.pdf_processor` | `validate_and_extract_pdf` |
| **Text Cleaning** | `src.text_cleaner` | `clean_text`, `clean_pages_text` |
| **Document Chunking** | `src.document_processor` | `chunk_document`, `save_extracted_txt` |
| **Semantic Retrieval** | `src.retriever` | `DocumentRetriever`, `format_context_for_prompt` |
| **LLM Inference** | `src.llm` | `get_llm_client`, `generate_grounded_answer` |
| **Conversation State**| `src.conversation` | `ConversationManager` |

The application dynamically resolves the path to `project-08-ai-pdf-research-assistant` via `sys.path.insert(0, str(PROJECT_08_DIR))`, ensuring zero duplication and strict architectural reusability.

---

## 4. Key Engineering Decisions

1. **State Persistence & Rerun Protection:**
   - Document extraction and vector indexing execute **only once** per uploaded file. If Streamlit reruns on user interaction, cached index objects in `st.session_state` prevent redundant processing.
2. **Model Caching (`@st.cache_resource`):**
   - The `DocumentRetriever` instance is cached using Streamlit's resource cache, avoiding reloading embedding pipelines or re-initializing API clients on each turn.
3. **Graceful Error Handling:**
   - Empty files, corrupted headers, or image-only scanned PDFs display clear, human-readable alerts rather than raw Python tracebacks.
   - Upstream API errors (such as expired or invalid API keys) are caught and presented as structured system messages in the chat stream.
4. **Strict Grounding:**
   - Grounded answering system instructions strictly mandate the model to answer only based on retrieved excerpts or explicitly state when the document does not contain the requested information.

---

## 5. Getting Started

### Prerequisites
Use the shared Python virtual environment located at the workspace root:

```bash
# Activate workspace virtual environment (PowerShell)
..\.venv\Scripts\Activate.ps1
```

### Installation
All core dependencies (`streamlit`, `pypdf`, `openai`, `torch`, `transformers`, `python-dotenv`) are installed in the shared `.venv`. To verify dependencies:

```bash
pip install -r requirements.txt
```

### Configuration
Provide your OpenRouter API key using any of the following methods:
1. In `.streamlit/secrets.toml`:
   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   ```
2. In `.env`:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-...
   ```
3. Directly inside the Streamlit sidebar under **API Configuration**.

---

## 6. Running the Application

Launch the Streamlit app from within this project folder:

```bash
cd project-09-research-assistant-ui
streamlit run app.py
```

Or run via the shared virtual environment binary from the workspace root:

```powershell
& "f:\Projects\AI and LLM\.venv\Scripts\streamlit.exe" run project-09-research-assistant-ui\app.py
```

The application will be accessible at:
- **Live Deployment:** [https://research-assistant-rpzr.onrender.com/](https://research-assistant-rpzr.onrender.com/)
- **Local:** `http://localhost:8501`

---

## 7. Running Tests

Run the integration and end-to-end test suites:

```bash
# Test 1: Integration & import resolution
python -m unittest test_ui_integration.py

# Test 2: Full end-to-end research flow
python -m unittest test_e2e_research_flow.py
```
