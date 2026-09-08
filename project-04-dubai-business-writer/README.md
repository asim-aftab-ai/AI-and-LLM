# Project 04 — Dubai Business Writer

A practical, concept-focused GenAI application demonstrating **System Prompts & Personas** in modern Large Language Model (LLM) architectures using Streamlit and the OpenRouter API.

---

## Overview

**Dubai Business Writer** is an interactive writing assistant tailored for executive and commercial communication within the United Arab Emirates (UAE) and Gulf business landscape.

- **Persona:** Professional • Culturally Aware • Business-Focused
- **Core Role:** Helps professionals compose, refine, rewrite, formalize, shorten, and structure executive emails, meeting summaries, proposals, and stakeholder correspondence.

---

## The Core Concept: System Prompts & Personas

Modern foundation models (such as Llama 3, Mistral, or Nemotron) are general-purpose text predictors with broad world knowledge. They do not naturally possess a specialized corporate identity or regional etiquette by default.

A **System Prompt** provides foundational, out-of-band guidance that establishes:
1. **Persona & Role:** Who the AI is and what domain it specializes in.
2. **Tone & Style:** How the AI speaks (diplomatic, concise, formal, culturally sensitive).
3. **Behavioral Constraints:** Explicit boundaries (never invent facts, statistics, or company policies; avoid cultural caricatures).
4. **Output Preferences:** Providing the completed business copy first, followed by optional contextual notes.

---

## Separation of System Instructions vs User Input

A critical architectural principle in generative AI applications is the strict separation between:

1. **System Message (`role: system`):**
   Controlled entirely by the application. It enforces the Dubai Business Writer persona and rules. The end user cannot overwrite or replace this prompt.
2. **Cumulative Conversation History (`role: user` & `role: assistant`):**
   Stores previous turns so that follow-up requests (e.g., *"Make it more formal"*, *"Now shorten it to three bullet points"*) retain conversational context.
3. **Current User Input (`role: user`):**
   The specific writing task or revision requested by the user.

### API Request Payload Structure

```json
[
  {
    "role": "system",
    "content": "You are Dubai Business Writer, an expert professional business writing assistant specialized in the UAE..."
  },
  {
    "role": "user",
    "content": "Write a follow-up email to a client regarding a delayed proposal."
  },
  {
    "role": "assistant",
    "content": "Dear Mr. Al Mansoori, Thank you for your continued partnership..."
  },
  {
    "role": "user",
    "content": "Make it more diplomatic."
  }
]
```

The underlying LLM is general-purpose; the system prompt gives it the Dubai Business Writer role, and the user provides the task.

---

## Architecture & Data Flow

```text
User Action (Chat Input or Starter Example)
  ↓
Streamlit Session State (Cumulative Dialogue History)
  ↓
Payload Assembly: [System Message (Persona)] + [Dialogue History]
  ↓
OpenRouter API Gateway (https://openrouter.ai/api/v1)
  ↓
Selected Model (e.g., google/gemini-2.5-flash)
  ↓
Generated Business Writing
  ↓
Render in Streamlit Chat UI + Save to Session State
```

---

## Project Structure

```text
project-04-dubai-business-writer/
├── .streamlit/
│   └── secrets.toml      # Local secrets file storing OPENROUTER_API_KEY (never committed)
├── app.py                # Main Streamlit application and persona system prompt logic
├── requirements.txt      # Project dependencies (streamlit, openai)
└── README.md             # Project documentation, architecture, and setup instructions
```

---

## Setup and Running Instructions

### 1. Activate the Shared Root Virtual Environment

From the root `AI and LLM` folder in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Verify Dependencies

Dependencies (`streamlit`, `openai`) are already installed in the root `.venv`. To re-verify:

```powershell
pip install -r project-04-dubai-business-writer\requirements.txt
```

### 3. Configure the OpenRouter API Key

Open the secrets file:

`project-04-dubai-business-writer/.streamlit/secrets.toml`

Ensure your OpenRouter API key is configured:

```toml
OPENROUTER_API_KEY = "your-actual-openrouter-key-here"
```

> **Security Note:** `.streamlit/secrets.toml` is ignored by the workspace `.gitignore` and must never be committed to version control.

### 4. Run the Streamlit Application

Navigate to the project folder and launch Streamlit:

```powershell
cd project-04-dubai-business-writer
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Learning Takeaways

1. **Personas are System Instructions:** An LLM does not need fine-tuning to adopt a nuanced persona; a well-crafted system prompt establishes the tone, domain focus, and behavioral constraints.
2. **Separation of Concerns:** The application controls the role (system prompt), while the user controls the assignment (user prompt).
3. **Conversational Refinement:** Multi-turn chat enables users to guide the assistant through iterative drafts (e.g., draft → refine tone → shorten) without retyping the original context.
