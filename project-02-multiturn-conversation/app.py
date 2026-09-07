"""
Project 02: Multi-Turn Conversation with Full Message History
=============================================================
A minimal educational GenAI project demonstrating how conversational context
is maintained across turns by appending messages to an in-memory history list
and sending the complete history with each OpenRouter API call.

Run with:
    streamlit run app.py
"""

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Turn Conversation",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Multi-Turn Conversation")
st.caption("Understanding message history with the OpenRouter API")

# ---------------------------------------------------------------------------
# OpenRouter API & Model Configuration
# ---------------------------------------------------------------------------
# OpenRouter's OpenAI-compatible API endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The default model is configured in this single clear location:
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"

# ---------------------------------------------------------------------------
# API Key from Streamlit Secrets
# ---------------------------------------------------------------------------
# The API key must come from Streamlit secrets (.streamlit/secrets.toml)
api_key = None
try:
    if "OPENROUTER_API_KEY" in st.secrets:
        key_value = st.secrets["OPENROUTER_API_KEY"].strip()
        # Ensure it is not the default placeholder
        if key_value and key_value != "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
            api_key = key_value
except Exception:
    pass

# ---------------------------------------------------------------------------
# Sidebar Settings & Controls
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")

# Allow selecting from active OpenRouter models
model_name = st.sidebar.selectbox(
    "Model",
    options=[
        DEFAULT_MODEL,
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "liquid/lfm-2.5-2.6b:free",
        "google/gemma-4-26b-a4b-it:free",
        "meta-llama/llama-3.3-70b-instruct"
    ],
    index=0,
    help="Select the model to route requests to through OpenRouter."
)

# Optional sidebar fallback if secrets.toml has not been edited yet
if not api_key:
    sidebar_key = st.sidebar.text_input(
        "OpenRouter API Key (Manual Entry)",
        type="password",
        help="Paste your key here if not yet configured in .streamlit/secrets.toml"
    ).strip()
    if sidebar_key and sidebar_key != "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
        api_key = sidebar_key

# ---------------------------------------------------------------------------
# System Message & Session State Initialization
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a helpful AI assistant. Maintain context from the conversation and answer clearly."
}

# The conversation history is stored in Streamlit session state as a list of dicts:
# [
#     {"role": "system", "content": "..."},
#     {"role": "user", "content": "..."},
#     {"role": "assistant", "content": "..."},
#     ...
# ]
if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_MESSAGE]

# ---------------------------------------------------------------------------
# Control: New Conversation
# ---------------------------------------------------------------------------
if st.sidebar.button("New Conversation"):
    # Reset conversation history back to only the initial system message
    st.session_state.messages = [SYSTEM_MESSAGE]
    st.rerun()

# ---------------------------------------------------------------------------
# Inspect Message History (Educational View)
# ---------------------------------------------------------------------------
with st.sidebar.expander("Inspect Message History", expanded=False):
    st.caption("This is the exact message list sent to OpenRouter with each turn:")
    st.json(st.session_state.messages)

# ---------------------------------------------------------------------------
# Check for Missing API Key
# ---------------------------------------------------------------------------
if not api_key:
    st.info(
        "OpenRouter API key is not configured. "
        "Open project-02-multiturn-conversation/.streamlit/secrets.toml "
        "and replace PASTE_YOUR_OPENROUTER_API_KEY_HERE with your real key, "
        "or enter it in the sidebar."
    )

# ---------------------------------------------------------------------------
# Display Existing Conversation History
# ---------------------------------------------------------------------------
# Render all user and assistant messages currently stored in session state.
# We do not display the internal system message as a chat bubble.
for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# ---------------------------------------------------------------------------
# Chat Input & Multi-Turn Execution
# ---------------------------------------------------------------------------
if user_prompt := st.chat_input("Type your message here..."):
    if not api_key:
        st.error(
            "Cannot send message: OpenRouter API key is missing. "
            "Please configure OPENROUTER_API_KEY in .streamlit/secrets.toml."
        )
    else:
        # Step 1: Add the user message to conversation history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Step 2: Immediately display user message in the chat interface
        with st.chat_message("user"):
            st.write(user_prompt)

        # Step 3: Send the FULL conversation history to OpenRouter
        with st.chat_message("assistant"):
            try:
                # Initialize client pointing to OpenRouter's OpenAI-compatible endpoint
                client = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=api_key
                )

                # Send the entire message list (system + all turns + latest user message)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=st.session_state.messages
                )

                # Step 4: Extract assistant response
                assistant_content = response.choices[0].message.content
                st.write(assistant_content)

                # Step 5: Add assistant response to conversation history
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})

            except Exception as error:
                # User-friendly error message without exposing credentials or large tracebacks
                error_message = str(error)
                if "401" in error_message or "Unauthorized" in error_message:
                    st.error("Authentication failed. Please verify your OpenRouter API key in .streamlit/secrets.toml.")
                elif "429" in error_message or "rate limit" in error_message.lower():
                    st.error("Rate limit reached. Please wait a moment or try another model.")
                else:
                    st.error(f"Error communicating with OpenRouter: {error_message}")
