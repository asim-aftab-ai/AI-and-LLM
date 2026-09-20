"""Semantic Similarity Tool - Streamlit Application.

A minimalist, beginner-friendly application comparing the semantic meaning of two
texts using local sentence embeddings and cosine similarity.
"""

import sys
import streamlit as st
from similarity_engine import load_embedding_model, compute_similarity, interpret_score

# Page configuration
st.set_page_config(
    page_title="Semantic Similarity Tool",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Ensure UTF-8 output encoding across environments
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the embedding model once across Streamlit reruns."""
    return load_embedding_model()


def apply_minimalist_css():
    """Inject clean, minimalist CSS for an uncluttered technical utility aesthetic."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 780px;
            padding-top: 2.5rem;
            padding-bottom: 3.5rem;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.3rem;
        }
        .main-header p {
            font-size: 1rem;
            color: #94a3b8;
            margin: 0;
        }
        .score-card {
            text-align: center;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1.8rem 1.5rem;
            background: rgba(15, 23, 42, 0.5);
            margin: 1.5rem 0;
        }
        .score-number {
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1;
            font-family: monospace;
            margin: 0.5rem 0;
        }
        .score-badge {
            display: inline-block;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            margin-bottom: 0.8rem;
        }
        .score-explanation {
            font-size: 0.95rem;
            color: #cbd5e1;
            max-width: 580px;
            margin: 1rem auto 0 auto;
            line-height: 1.6;
        }
        .how-it-works-box {
            font-family: monospace;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 1rem;
            font-size: 0.85rem;
            color: #94a3b8;
            line-height: 1.6;
            margin: 0.8rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_minimalist_css()

    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>Semantic Similarity Tool</h1>
            <p>Compare the semantic meaning of two pieces of text</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for pre-filling text inputs via examples
    if "text_a_input" not in st.session_state:
        st.session_state.text_a_input = ""
    if "text_b_input" not in st.session_state:
        st.session_state.text_b_input = ""

    # Two text input areas
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Text A**")
        text_a = st.text_area(
            label="Text A Input",
            placeholder="Enter the first piece of text...",
            height=160,
            value=st.session_state.text_a_input,
            label_visibility="collapsed",
            key="text_a_box",
        )
    with col_b:
        st.markdown("**Text B**")
        text_b = st.text_area(
            label="Text B Input",
            placeholder="Enter the second piece of text...",
            height=160,
            value=st.session_state.text_b_input,
            label_visibility="collapsed",
            key="text_b_box",
        )

    # Action Button centered
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        calculate_clicked = st.button(
            "Calculate Semantic Similarity",
            use_container_width=True,
            type="primary",
        )

    # Process calculation on button click
    if calculate_clicked:
        cleaned_a = text_a.strip()
        cleaned_b = text_b.strip()

        if not cleaned_a and not cleaned_b:
            st.warning("Please enter text in both Text A and Text B to compare.")
        elif not cleaned_a:
            st.warning("Text A is empty. Please enter text in the first field.")
        elif not cleaned_b:
            st.warning("Text B is empty. Please enter text in the second field.")
        else:
            try:
                with st.spinner("Generating embeddings and computing cosine similarity..."):
                    tokenizer, model = get_model()
                    score = compute_similarity(cleaned_a, cleaned_b, tokenizer, model)
                    result = interpret_score(score)

                # Output presentation
                st.markdown("<hr style='border-color: #334155; margin: 2rem 0;'>", unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div class="score-card">
                        <div style="font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.06em; text-transform: uppercase;">
                            Similarity Score
                        </div>
                        <div class="score-number" style="color: {result['color']};">
                            {result['score']} <span style="font-size: 1.2rem; color: #64748b; font-weight: normal;">/ 1.00</span>
                        </div>
                        <div class="score-badge" style="background: {result['color']}22; color: {result['color']}; border: 1px solid {result['color']}66;">
                            {result['badge']}
                        </div>
                        <div class="score-explanation">
                            {result['explanation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Progress bar for visual representation
                st.progress(float(score))

            except Exception as e:
                st.error(f"An unexpected error occurred during similarity computation: {e}")

    # Example Section
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Try an Example", expanded=False):
        st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.6rem;'>Click an example to load it into the inputs above:</div>", unsafe_allow_html=True)
        
        ex_cols = st.columns(3)
        with ex_cols[0]:
            if st.button("High Similarity Example", use_container_width=True):
                st.session_state.text_a_input = "How can I renew my residence permit?"
                st.session_state.text_b_input = "What is the process for extending a residence visa?"
                st.rerun()

        with ex_cols[1]:
            if st.button("Moderate Similarity Example", use_container_width=True):
                st.session_state.text_a_input = "Electric vehicles produce zero tailpipe emissions."
                st.session_state.text_b_input = "Solar panels generate clean renewable electricity for homes."
                st.rerun()

        with ex_cols[2]:
            if st.button("Low Similarity Example", use_container_width=True):
                st.session_state.text_a_input = "The chef baked a fresh loaf of sourdough bread."
                st.session_state.text_b_input = "Quantum computers leverage superposition to process information."
                st.rerun()

    # Educational "How does this work?" expandable section
    with st.expander("How does this work?", expanded=False):
        st.markdown(
            """
            The application converts both texts into high-dimensional numerical representations called **embeddings**.
            These embeddings capture the conceptual meaning of words in context rather than just matching characters.
            
            ```text
            Text A                     Text B
               ↓                          ↓
            Embedding                  Embedding
               ↓                          ↓
            Vector A                   Vector B
                \\                        /
                 \\                      /
                  Cosine Similarity (A, B)
                             ↓
                    Score from 0.0 to 1.0
            ```

            **1. Text to Vector**: A transformer neural network (`all-MiniLM-L6-v2`) projects each sentence into a 384-dimensional vector space.
            
            **2. Geometric Angle**: **Cosine Similarity** measures the cosine of the angle between the two vectors. If the concepts are identical, the angle is 0° and the similarity is 1.0. If the concepts are completely unrelated, the vectors point in orthogonal directions and the similarity approaches 0.0.
            """
        )


if __name__ == "__main__":
    main()
