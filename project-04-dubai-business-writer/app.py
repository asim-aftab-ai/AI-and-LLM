"""
Project 04: Dubai Business Writer
==================================
A professional UAE-focused business writing assistant demonstrating
how a dedicated system prompt defines the AI's persona, expertise,
cultural tone, and behavioral constraints in a multi-turn conversation.

Run with:
    streamlit run app.py
"""

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dubai Business Writer",
    layout="centered"
)

# ---------------------------------------------------------------------------
# OpenRouter API Configuration
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash"

# ---------------------------------------------------------------------------
# System Prompt & Persona Definition
# ---------------------------------------------------------------------------
# The system prompt controls the persona, tone, cultural etiquette, and constraints.
# The user controls the task; the application enforces this persona.
SYSTEM_PROMPT = """You are Dubai Business Writer, an expert professional business writing assistant specialized in the United Arab Emirates (UAE) and Gulf commercial landscape.

Your primary mission is to help professionals draft, refine, adapt, rewrite, shorten, expand, and polish executive communications, including emails, formal letters, meeting follow-ups, project updates, and proposals.

COMMUNICATION PRINCIPLES:
1. Professional & Diplomatic: Maintain high commercial standards, respectful phrasing, and appropriate formality.
2. Culturally Aware: Respect UAE and Gulf commercial etiquette, including courteous greetings, relational warmth, collaborative phrasing, and constructive positioning without exaggeration or cultural stereotypes.
3. Clear & Concise: Prioritize clarity, executive readability, and logical paragraph organization. Avoid convoluted or overly ornate prose.
4. Business-Focused: Keep all messaging aligned with practical business goals, stakeholder relationships, and mutual success.

BEHAVIORAL CONSTRAINTS:
- Understand the user's specific business purpose before generating copy.
- Preserve the user's intended meaning unless specifically asked to change it.
- Never invent facts, company policies, legal commitments, financial claims, or artificial statistics.
- Do not pretend to have verified external proprietary data or live legal requirements.
- If important context is missing, make a reasonable, professional assumption and briefly mention it.
- Present the requested business writing first. Include brief, helpful contextual notes only when genuinely useful to the user."""

# ---------------------------------------------------------------------------
# API Key Loading from Streamlit Secrets
# ---------------------------------------------------------------------------
api_key = None
try:
    if "OPENROUTER_API_KEY" in st.secrets:
        key_value = st.secrets["OPENROUTER_API_KEY"].strip()
        if key_value and key_value not in ["your-key-here", "PASTE_YOUR_OPENROUTER_API_KEY_HERE"]:
            api_key = key_value
except Exception:
    pass

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
# We maintain conversation messages (user and assistant turns) in session state.
# The system prompt is prepended automatically upon each API request.
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar Settings & Persona Profile
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")

# Persona profile summary card
st.sidebar.markdown(
    """
    **Persona Profile**  
    Dubai Business Writer  
    Professional • Culturally Aware • Business-Focused
    """
)
st.sidebar.caption("Tailored for UAE executive correspondence, proposals, and client relations.")

# Model selection focusing on Google models
model_name = st.sidebar.selectbox(
    "OpenRouter Model",
    options=[
        DEFAULT_MODEL,
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash-lite",
        "google/gemini-3.7-flash",
        "nvidia/nemotron-3.5-lightning:free"
    ],
    index=0,
    help="Select the underlying LLM to power the Dubai Business Writer persona."
)

# Temperature / Creativity Control
temperature = st.sidebar.slider(
    "Creativity (Temperature)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Lower values produce more deterministic, formal business writing."
)

# Clear conversation control
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# Optional manual API key entry if secrets.toml is missing
if not api_key:
    manual_key = st.sidebar.text_input(
        "OpenRouter API Key (Manual Entry)",
        type="password",
        help="Enter your OpenRouter key here if not configured in .streamlit/secrets.toml"
    ).strip()
    if manual_key and manual_key not in ["your-key-here", "PASTE_YOUR_OPENROUTER_API_KEY_HERE"]:
        api_key = manual_key

# Educational expanders for inspecting architecture
with st.sidebar.expander("Inspect System Prompt (Persona)", expanded=False):
    st.caption("This system instruction governs the model's persona and constraints:")
    st.text(SYSTEM_PROMPT)

with st.sidebar.expander("Inspect API Request Structure", expanded=False):
    st.caption("The exact payload structure sent to OpenRouter on each request:")
    sample_payload = [
        {"role": "system", "content": "[System Prompt: Dubai Business Writer Persona]"}
    ] + st.session_state.messages
    st.json(sample_payload)

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.title("Dubai Business Writer")
st.caption("A professional UAE-focused business writing assistant.")

st.markdown(
    """
    **Persona:** Dubai Business Writer  
    *Professional • Culturally Aware • Business-Focused*
    """
)
st.markdown("---")

# ---------------------------------------------------------------------------
# API Key Verification Banner
# ---------------------------------------------------------------------------
if not api_key:
    st.info(
        "OpenRouter API key is not configured. "
        "Add OPENROUTER_API_KEY to project-04-dubai-business-writer/.streamlit/secrets.toml "
        "or enter it in the sidebar."
    )

# ---------------------------------------------------------------------------
# Starter Examples (Displayed only when conversation is empty)
# ---------------------------------------------------------------------------
pending_prompt = None

if len(st.session_state.messages) == 0:
    st.markdown("##### Starter Examples")
    st.caption("Click an example to start your business writing session:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Follow-up on delayed project proposal"):
            pending_prompt = "Write a professional follow-up email to a UAE client regarding a project proposal that has experienced a slight scheduling delay."
        if st.button("Rewrite email to a more diplomatic tone"):
            pending_prompt = "Rewrite this email to make it more diplomatic and suitable for a senior executive partner in Dubai: 'We need your approval by tomorrow or we cannot deliver the work on time.'"

    with col2:
        if st.button("Meeting follow-up with Dubai business partner"):
            pending_prompt = "Draft a formal meeting follow-up email to a Dubai-based commercial partner summarizing our introductory discussion and outlining agreed next steps."
        if st.button("Executive project milestone update"):
            pending_prompt = "Draft a concise project milestone update for regional stakeholders highlighting achievements, upcoming targets, and thanking the team for their collaboration."

# ---------------------------------------------------------------------------
# Render Conversation History
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------------------------------------------------------
# Chat Input & Multi-Turn Processing
# ---------------------------------------------------------------------------
chat_prompt = st.chat_input("Tell me what you want to write...")

# Determine if input was submitted via text input or starter button
active_input = chat_prompt or pending_prompt

if active_input:
    clean_input = active_input.strip()

    if not clean_input:
        st.warning("Please enter a valid request.")
    elif not api_key:
        st.error(
            "Cannot generate response: OpenRouter API key is missing. "
            "Please configure OPENROUTER_API_KEY in .streamlit/secrets.toml."
        )
    else:
        # Step 1: Append user message to conversation history
        st.session_state.messages.append({"role": "user", "content": clean_input})

        # Step 2: Render the user message in the UI immediately
        with st.chat_message("user"):
            st.write(clean_input)

        # Step 3: Construct the full message payload:
        # [System Message (Persona)] + [Cumulative Conversation History]
        full_payload = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + st.session_state.messages

        # Step 4: Call OpenRouter Chat Completions
        with st.chat_message("assistant"):
            with st.spinner("Dubai Business Writer is generating your response..."):
                try:
                    client = OpenAI(
                        base_url=OPENROUTER_BASE_URL,
                        api_key=api_key
                    )

                    response = client.chat.completions.create(
                        model=model_name,
                        messages=full_payload,
                        temperature=temperature
                    )

                    assistant_text = response.choices[0].message.content

                    if not assistant_text or not assistant_text.strip():
                        st.warning("The model returned an empty response. Please try again or adjust settings.")
                    else:
                        st.write(assistant_text)
                        # Step 5: Save assistant response in session state history
                        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                except Exception as error:
                    error_msg = str(error)
                    if "401" in error_msg or "Unauthorized" in error_msg:
                        st.error("Authentication failed. Please verify your OpenRouter API key in .streamlit/secrets.toml.")
                    elif "429" in error_msg or "rate limit" in error_msg.lower():
                        st.error("Rate limit reached. Please wait a moment or switch models in the sidebar.")
                    else:
                        st.error(f"Error communicating with OpenRouter: {error_msg}")
