"""
Project 03: AI Writing Assistant
=================================
An educational GenAI application demonstrating prompt engineering principles:
Role prompting, clear task instructions, tone and length constraints,
and clean output formatting using OpenRouter and Streamlit.

Run with:
    streamlit run app.py
"""

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Writing Assistant",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Header & Description
# ---------------------------------------------------------------------------
st.title("AI Writing Assistant")
st.write(
    "This application generates structured content based on your chosen topic, "
    "tone, and desired length using prompt engineering techniques."
)

# ---------------------------------------------------------------------------
# OpenRouter API & Model Configuration
# ---------------------------------------------------------------------------
# OpenRouter OpenAI-compatible endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Centralized model configuration (easy to change or update)
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"

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
# Sidebar Settings
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")

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
    help="Select the OpenRouter model to handle text generation."
)

# Optional manual key entry if secrets.toml is not yet edited
if not api_key:
    manual_key = st.sidebar.text_input(
        "OpenRouter API Key (Manual Entry)",
        type="password",
        help="Enter your OpenRouter key here if not configured in .streamlit/secrets.toml"
    ).strip()
    if manual_key and manual_key not in ["your-key-here", "PASTE_YOUR_OPENROUTER_API_KEY_HERE"]:
        api_key = manual_key

# ---------------------------------------------------------------------------
# Prompt Construction Function
# ---------------------------------------------------------------------------
def construct_prompt(topic: str, tone: str, length: str) -> str:
    """
    Constructs a structured prompt applying core prompt engineering concepts:
    - Role definition
    - Explicit task instructions
    - Input constraints (tone and length definitions)
    - Negative constraints and output constraints
    """
    length_guidelines = {
        "Short": "approximately 100-150 words (1 to 2 concise paragraphs)",
        "Medium": "approximately 250-400 words (3 to 4 well-developed paragraphs)",
        "Long": "approximately 500-750 words (5 or more detailed paragraphs with subheadings if appropriate)"
    }

    tone_guidelines = {
        "Professional": "formal, objective, well-structured, and authoritative",
        "Casual": "friendly, conversational, engaging, and easy to read",
        "Funny": "witty, humorous, entertaining, with clever lighthearted remarks"
    }

    target_length_desc = length_guidelines.get(length, "approximately 250 words")
    target_tone_desc = tone_guidelines.get(tone, "balanced and clear")

    prompt = f"""ROLE:
You are an AI writing assistant.

TASK:
Write content based on the user's topic provided below.

TONE:
Use a {tone} tone ({target_tone_desc}).

LENGTH:
Follow the {length} length specification: {target_length_desc}.

INSTRUCTIONS:
1. Stay strictly focused on the topic.
2. Produce useful, well-structured, readable content.
3. Consistently match the requested tone throughout the entire response.
4. Follow the requested length as closely as practical.
5. Do not discuss the prompt itself.
6. Do not add introductory or concluding meta-commentary (such as "Here is your article:" or "I hope this helps!").

OUTPUT CONSTRAINT:
Return only the requested content.

USER TOPIC:
{topic}"""
    return prompt

# ---------------------------------------------------------------------------
# Input Section
# ---------------------------------------------------------------------------
st.subheader("Input")

topic_input = st.text_area(
    "Topic",
    placeholder="Describe what you want written (e.g., The importance of continuous learning in software engineering)...",
    height=130
)

col_tone, col_length = st.columns(2)

with col_tone:
    tone_selection = st.selectbox(
        "Tone",
        options=["Professional", "Casual", "Funny"],
        index=0,
        help="Select the tone of voice for the generated text."
    )

with col_length:
    length_selection = st.selectbox(
        "Length",
        options=["Short", "Medium", "Long"],
        index=1,
        help="Select the approximate length of the generated output."
    )

generate_button = st.button("Generate", type="primary")

# ---------------------------------------------------------------------------
# Generation & Output Section
# ---------------------------------------------------------------------------
if generate_button:
    # Validation 1: Check for empty topic
    if not topic_input.strip():
        st.warning("Please enter a topic before generating content.")
    # Validation 2: Check for API key
    elif not api_key:
        st.error(
            "OpenRouter API key is missing. "
            "Please configure OPENROUTER_API_KEY in .streamlit/secrets.toml "
            "or enter it in the sidebar."
        )
    else:
        # Construct the engineered prompt
        prompt = construct_prompt(topic_input.strip(), tone_selection, length_selection)

        with st.spinner("Generating content via OpenRouter..."):
            try:
                # Initialize OpenRouter client
                client = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=api_key
                )

                # Send request to the selected model
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                # Extract response text
                generated_text = response.choices[0].message.content

                # Validation 3: Check for empty response
                if not generated_text or not generated_text.strip():
                    st.warning("The model returned an empty response. Please try again or choose another model.")
                else:
                    # Store generated text in session state for persistence
                    st.session_state["generated_content"] = generated_text.strip()
                    st.session_state["last_prompt"] = prompt

            except Exception as error:
                error_msg = str(error)
                if "401" in error_msg or "Unauthorized" in error_msg:
                    st.error("Authentication failed. Please verify your OpenRouter API key in .streamlit/secrets.toml.")
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    st.error("Rate limit reached. Please wait a moment or switch models in the sidebar.")
                else:
                    st.error(f"Error communicating with OpenRouter: {error_msg}")

# Display previously generated content if available
if "generated_content" in st.session_state:
    st.markdown("---")
    st.subheader("Generated Content")
    
    # Display the generated content
    content = st.session_state["generated_content"]
    st.write(content)

    # Download button allowing user to save content locally
    st.download_button(
        label="Download Content as Text",
        data=content,
        file_name="generated_content.txt",
        mime="text/plain"
    )

    # Educational View: Inspect the structured prompt
    with st.expander("Inspect Constructed Prompt (Prompt Engineering View)", expanded=False):
        st.caption("This is the exact prompt structure sent to the LLM:")
        st.code(st.session_state.get("last_prompt", ""), language="text")
