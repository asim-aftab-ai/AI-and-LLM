# Project 03 — AI Writing Assistant

A focused educational GenAI project demonstrating prompt engineering concepts (role prompting, task clarity, tone/length constraints, and structured output formatting) using Streamlit and the OpenRouter API.

---

## Overview

The AI Writing Assistant allows users to generate tailored content by specifying:
- **Topic:** The subject matter or writing task.
- **Tone:** The voice and style of the piece (Professional, Casual, or Funny).
- **Length:** The target depth and length (Short, Medium, or Long).

When the user clicks **Generate**, the application builds a structured, instruction-driven prompt, submits it to an LLM via OpenRouter, and presents the resulting text with a download option.

---

## Architecture & Flow

```text
User Input (Topic, Tone, Length)
  ↓
Streamlit UI
  ↓
Prompt Construction (Role, Task, Tone, Length, Constraints)
  ↓
OpenRouter API Gateway (https://openrouter.ai/api/v1)
  ↓
Selected LLM (e.g., nvidia/nemotron-3.5-lightning:free)
  ↓
Generated Content
  ↓
Display in Streamlit + Download Button
```

---

## Prompt Engineering Concepts Demonstrated

This project applies core prompt engineering principles directly in code (`construct_prompt()` in `app.py`):

1. **Role Prompting:**
   Establishes context and persona:
   `ROLE: You are an AI writing assistant.`

2. **Explicit Task Definition:**
   Clearly states what needs to be written:
   `TASK: Write content based on the user's topic provided below.`

3. **Input Constraints (Tone & Length):**
   Explicitly defines what the selected tone and length mean in practice rather than leaving them ambiguous:
   - *Professional:* Formal, objective, well-structured, authoritative.
   - *Casual:* Friendly, conversational, engaging, easy to read.
   - *Funny:* Witty, humorous, entertaining, clever remarks.
   - *Short:* ~100-150 words (1-2 paragraphs).
   - *Medium:* ~250-400 words (3-4 paragraphs).
   - *Long:* ~500-750 words (5+ paragraphs).

4. **Negative Constraints & Formatting Rules:**
   Instructs the model to avoid conversational filler, introductory remarks ("Sure, here is your article:"), or meta-commentary about the prompt.

5. **Output Constraint:**
   `OUTPUT CONSTRAINT: Return only the requested content.`

---

## Technologies Used

- **Python 3:** Application logic and execution.
- **Streamlit:** Interactive web interface and session state management.
- **OpenRouter API:** Unified LLM provider giving access to open and proprietary models via an OpenAI-compatible endpoint.
- **OpenAI Python SDK:** Official client library configured to target OpenRouter's endpoint (`https://openrouter.ai/api/v1`).

---

## Project Structure

```text
project-03-ai-writing-assistant/
├── .streamlit/
│   └── secrets.toml      # Local secrets file storing OPENROUTER_API_KEY (never committed)
├── app.py                # Main Streamlit application and prompt construction logic
├── requirements.txt      # Project dependencies (streamlit, openai)
└── README.md             # Project documentation and setup guide
```

---

## Setup and Running Instructions

### 1. Activate the Existing Root Virtual Environment

From the root `AI and LLM` workspace in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Project Dependencies

Install dependencies into the active virtual environment:

```powershell
pip install -r project-03-ai-writing-assistant\requirements.txt
```

### 3. Configure the OpenRouter API Key

Open the secrets file:

`project-03-ai-writing-assistant/.streamlit/secrets.toml`

Add your OpenRouter API key:

```toml
OPENROUTER_API_KEY = "your-key-here"
```

> **Security Note:** `.streamlit/secrets.toml` is ignored by `.gitignore` and must never be pushed to version control.

### 4. Run the Streamlit Application

Navigate to the project folder and launch Streamlit:

```powershell
cd project-03-ai-writing-assistant
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Learning Takeaways

1. **Prompt Construction Drives Output Quality:** Instead of sending raw user input directly to an LLM, structuring the prompt with clear roles, constraints, and instructions produces significantly more reliable, predictable results.
2. **Stateless Generation:** Unlike multi-turn chat (Project 02), a writing assistant typically executes a single-turn completion task with a self-contained, highly specified prompt.
3. **Decoupled Architecture:** Using OpenRouter allows switching models without rewriting application code or changing client SDKs.
