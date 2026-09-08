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
└── project-04-dubai-business-writer/   # Project 04: System Prompts & Personas
    ├── app.py                          # Streamlit executive assistant UI
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

---

## Getting Started

### 1. Environment Activation (Windows)

Open PowerShell at the root of `AI and LLM`:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Configure API Secrets

For projects using OpenRouter (Projects 02, 03, & 04), create `.streamlit/secrets.toml` inside the respective project folder:

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
```
