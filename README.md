# AI and LLM Learning Workspace

A hands-on workspace exploring core concepts in Generative AI, Large Language Models (LLMs), tokenization, conversational architectures, prompt engineering, and system prompts through practical, beginner-friendly Streamlit applications.

---

## Workspace Structure

This workspace uses a shared Python virtual environment at the root (`.venv`) to manage dependencies across all projects cleanly.

```text
AI and LLM/
│
├── .venv/                              # Shared Python virtual environment
├── .gitignore                          # Global gitignore protecting secrets and caches
├── README.md                           # Workspace overview and project index
│
├── project-01-arabic-token-inflation/  # Project 01: Tokenization & Subword Splits
│   ├── app.py                          # Streamlit application
│   ├── tokenizer_analysis.py           # Analysis module (tiktoken & AraBERT)
│   ├── requirements.txt
│   └── README.md
│
├── project-02-multiturn-conversation/  # Project 02: Multi-Turn Conversation
│   ├── app.py                          # Streamlit chat interface
│   ├── requirements.txt
│   └── README.md
│
├── project-03-ai-writing-assistant/    # Project 03: Prompt Engineering Writing Assistant
│   ├── app.py                          # Streamlit writing assistant UI
│   ├── requirements.txt
│   └── README.md
│
├── project-04-dubai-business-writer/   # Project 04: System Prompts & Personas
│   ├── app.py                          # Streamlit executive assistant UI
│   ├── requirements.txt
│   └── README.md
│
├── project-05-(huggingface-sentiment)/ # Project 05: Hugging Face Sentiment Analysis
│   ├── app.py                          # Streamlit application entrypoint
│   ├── main.py                         # Streamlit UI & interactive playground
│   ├── sentiment_analyzer.py           # DistilBERT pipeline inference
│   ├── sentences.py                    # Benchmark test sentences
│   ├── requirements.txt
│   └── README.md
│
├── project-06-arabic-morphology/       # Project 06: Arabic Morphological Analysis
│   ├── main.py                         # Streamlit learning interface & CAMeL Tools pipeline
│   ├── requirements.txt
│   └── README.md
│
├── project-07-semantic-similarity-tool/ # Project 07: Semantic Similarity Tool
│   ├── app.py                          # Streamlit application entrypoint
│   ├── similarity_engine.py            # Sentence embeddings & cosine similarity logic
│   ├── requirements.txt
│   └── README.md
├── project-08-ai-pdf-research-assistant/ # Project 08: AI PDF Research Assistant
│   ├── app.py                          # Streamlit application entrypoint
│   ├── src/                            # Modular pipeline (PDF, cleaning, retrieval, LLM)
│   ├── requirements.txt
│   └── README.md
│
└── project-09-research-assistant-ui/       # Project 09: Streamlit UI for Research Assistant
    ├── app.py                          # Streamlit UI entrypoint
    ├── test_ui_integration.py          # Integration verification tests
    ├── test_e2e_research_flow.py       # End-to-end research workflow test
    ├── requirements.txt
    └── README.md
```

---

## Projects Overview

### Project 01: English vs Arabic Tokenization Analysis
- **Folder:** [`project-01-arabic-token-inflation/`](./project-01-arabic-token-inflation/)
- **Core Concepts:** Subword tokenization, Byte-Pair Encoding (`tiktoken`/`cl100k_base`), WordPiece (`AraBERT`), token counts, context window limits, and the empirical **Arabic Token Inflation Factor**.
- **Key Takeaway:** Token inflation is an empirical outcome of tokenizer vocabulary distribution, not an inherent linguistic property of Arabic.

### Project 02: Multi-Turn Conversation with Full Message History
- **Folder:** [`project-02-multiturn-conversation/`](./project-02-multiturn-conversation/)
- **Core Concepts:** Stateless LLM APIs, Chat Completions, message roles (`system`, `user`, `assistant`), session state persistence, and transmitting cumulative message history on each turn via OpenRouter.
- **Key Takeaway:** An LLM does not remember past requests; conversational continuity exists entirely because the application re-sends the cumulative conversation history.

### Project 03: AI Writing Assistant with Prompt Engineering
- **Folder:** [`project-03-ai-writing-assistant/`](./project-03-ai-writing-assistant/)
- **Core Concepts:** Prompt engineering principles, role prompting, explicit task instructions, tone and length constraints, negative constraints, and structured output formatting via OpenRouter.
- **Key Takeaway:** Structuring prompts with explicit roles, guidelines, and output constraints yields predictable, high-quality responses compared to raw user queries.

### Project 04: Dubai Business Writer (System Prompts & Personas)
- **Folder:** [`project-04-dubai-business-writer/`](./project-04-dubai-business-writer/)
- **Core Concepts:** System prompts, persona design, UAE commercial etiquette, behavioral boundaries, and strict architectural separation between system instructions and user tasks.
- **Key Takeaway:** Foundation models are general-purpose; the system prompt gives the assistant its specialized persona, cultural etiquette, and professional boundaries, while the user provides the task.

### Project 05: Hugging Face Sentiment Analysis
- **Folder:** [`project-05-(huggingface-sentiment)/`](./project-05-(huggingface-sentiment)/)
- **Core Concepts:** Transformer pipelines, local neural inference with PyTorch, DistilBERT architecture, binary classification fine-tuning (SST-2), and confidence score interpretation.
- **Key Takeaway:** High-level Hugging Face pipelines abstract tokenization, model loading, and forward passes into a simple API while running deterministic classification locally without external API dependencies.

### Project 06: Arabic Morphological Analysis with CAMeL Tools
- **Folder:** [`project-06-arabic-morphology/`](./project-06-arabic-morphology/)
- **Core Concepts:** Arabic linguistic processing layers, word tokenization (`simple_word_tokenize`), morphological analysis with Modern Standard Arabic databases (`MorphologyDB` / `calima-msa-r13`), root and lemma extraction, grammatical feature decoding (POS, gender, number, aspect, case), and orthographic normalization.
- **Key Takeaway:** Generic tokenizers treat Arabic as raw character sequences; Arabic-aware morphological analysis exposes internal linguistic properties (roots, lemmas, clitics) vital for downstream retrieval, RAG, and LLM systems.

### Project 07: Semantic Similarity Tool
- **Folder:** [`project-07-semantic-similarity-tool/`](./project-07-semantic-similarity-tool/)
- **Core Concepts:** Sentence embeddings, dense vector representations, local transformer inference (`all-MiniLM-L6-v2`), mean pooling over token embeddings, and cosine similarity metric calculation.
- **Key Takeaway:** Keyword matching fails when different words express the same meaning; semantic embeddings project sentences into a geometric vector space where conceptual similarity is accurately captured by vector angles.

### Project 08: AI PDF Research Assistant
- **Folder:** [`project-08-ai-pdf-research-assistant/`](./project-08-ai-pdf-research-assistant/)
- **Core Concepts:** PDF text extraction (`pypdf`), text normalization and `.txt` export, sliding-window chunking, dense semantic vector retrieval via local Transformers, grounded question answering with strict anti-hallucination guardrails, and multi-turn conversational memory.
- **Key Takeaway:** Merging document extraction, dense semantic retrieval, and conversational history creates a coherent research workflow where answers are strictly grounded in document evidence and follow-up questions retain context.

### Project 09: Streamlit UI for AI Research Assistant
- **Folder:** [`project-09-research-assistant-ui/`](./project-09-research-assistant-ui/)
- **Live Demo:** [https://research-assistant-rpzr.onrender.com/](https://research-assistant-rpzr.onrender.com/)
- **Core Concepts:** Professional UI design, Conversation-Answer-Evidence paradigm, sidebar document management, expandable citation cards with authentic retrieval metadata (page number, chunk ID, relevance score), session state caching, and clean separation between backend and presentation logic.
- **Key Takeaway:** A production-style AI research application requires restrained, predictable UI workflows that elevate verified source evidence without overwhelming the researcher with unnecessary complexity or duplicate backend logic.

---

## Getting Started

### 1. Environment Activation (Windows)

Open PowerShell at the root of `AI and LLM`:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Configure API Secrets

For projects using OpenRouter (Projects 02, 03, 04, 08, & 09), create `.streamlit/secrets.toml` inside the respective project folder:

```toml
OPENROUTER_API_KEY = "your-openrouter-api-key-here"
```

> Secrets files are strictly ignored by `.gitignore` and must never be committed to version control.

### 3. Running Any Project

Navigate to the project folder and launch Streamlit:

```powershell
# Example: Run Project 01
cd project-01-arabic-token-inflation
streamlit run app.py

# Example: Run Project 02
cd project-02-multiturn-conversation
streamlit run app.py

# Example: Run Project 03
cd project-03-ai-writing-assistant
streamlit run app.py

# Example: Run Project 04
cd project-04-dubai-business-writer
streamlit run app.py

# Example: Run Project 05
cd "project-05-(huggingface-sentiment)"
streamlit run app.py

# Example: Run Project 06
cd project-06-arabic-morphology
streamlit run main.py

# Example: Run Project 07
cd project-07-semantic-similarity-tool
streamlit run app.py

# Example: Run Project 08
cd project-08-ai-pdf-research-assistant
streamlit run app.py

# Example: Run Project 09
cd project-09-research-assistant-ui
streamlit run app.py
```
