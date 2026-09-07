"""
Streamlit Application: English vs Arabic Tokenization Analyzer
==============================================================
An interactive educational GenAI learning tool to compare token counts,
subword segmentations, and token inflation factors between English and
Arabic text using tiktoken (cl100k_base) and Hugging Face (AraBERT).

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, Any

from tokenizer_analysis import (
    SAMPLE_DATASETS,
    analyze_with_tiktoken,
    analyze_with_hf,
    calculate_inflation_factor,
    get_inflation_interpretation,
    benchmark_built_in_samples,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="English vs Arabic Tokenization Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for aesthetic subword chips, RTL text, and metrics
st.markdown(
    """
    <style>
    .subword-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        padding: 12px;
        background-color: rgba(125, 125, 125, 0.08);
        border-radius: 8px;
        border: 1px solid rgba(125, 125, 125, 0.2);
        max-height: 280px;
        overflow-y: auto;
    }
    .token-chip {
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    .token-chip-en {
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }
    .token-chip-ar {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
        direction: rtl;
        text-align: right;
    }
    .token-id {
        font-size: 0.70rem;
        opacity: 0.75;
        margin-left: 4px;
        margin-right: 4px;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-callout {
        padding: 16px;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
        background-color: rgba(59, 130, 246, 0.06);
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Settings & Educational Notes
# ---------------------------------------------------------------------------
st.sidebar.title("Tokenizer Settings")

st.sidebar.markdown(
    """
    This project explores how **subword tokenizers** represent English vs Arabic text,
    and measures the **Arabic Token Inflation Factor**.
    """
)

encoding_name = st.sidebar.selectbox(
    "Tiktoken Encoding",
    options=["cl100k_base"],
    help="cl100k_base is the Byte-Pair Encoding (BPE) used by GPT-4 and GPT-3.5-turbo."
)

st.sidebar.caption(
    "**Note on GPT Tokenizers:** `cl100k_base` has a vocabulary of ~100,000 tokens, "
    "predominantly trained on English web corpora."
)

hf_model_name = st.sidebar.selectbox(
    "Hugging Face Arabic Tokenizer",
    options=["aubmindlab/bert-base-arabertv02"],
    help="AraBERT is an Arabic-specific BERT model trained on ~77GB of Arabic text."
)

st.sidebar.caption(
    "**Note on AraBERT:** Uses a specialized Arabic WordPiece vocabulary (64,000 tokens) "
    "with prefix/suffix morphological preservation."
)

st.sidebar.markdown("---")
st.sidebar.subheader("What is Token Inflation?")
st.sidebar.markdown(
    r"""
    $$\text{Inflation Factor} = \frac{\text{Arabic Tokens}}{\text{English Tokens}}$$

    - **1.0x:** Equal token count
    - **> 1.0x:** Arabic uses more tokens
    - **< 1.0x:** Arabic uses fewer tokens
    """
)

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.title("English vs Arabic Tokenization Analyzer")
st.markdown(
    """
    **GenAI Fundamentals Project 01:** Compare how different tokenizers represent English and Arabic text.
    Explore subword segmentations, measure the empirical **Arabic Token Inflation Factor**, and see why
    tokenization directly impacts **LLM API costs** and **context window limits**.
    """
)

# ---------------------------------------------------------------------------
# Section 1: Input & Presets
# ---------------------------------------------------------------------------
st.subheader("1. Text Inputs & Example Presets")

col_preset, col_sub = st.columns([1, 2])

with col_preset:
    category_choice = st.radio(
        "Select Example Category:",
        options=["Banking", "News", "Custom Input"],
        horizontal=True,
    )

selected_en = ""
selected_ar = ""

if category_choice in ["Banking", "News"]:
    examples = SAMPLE_DATASETS[category_choice]
    example_names = list(examples.keys())
    with col_sub:
        selected_example_name = st.selectbox(
            f"Select a {category_choice} Scenario:",
            options=example_names,
        )
    selected_en = examples[selected_example_name]["english"]
    selected_ar = examples[selected_example_name]["arabic"]
else:
    with col_sub:
        st.info("Enter your own custom English and Arabic text below to experiment.")
    selected_en = "Your subscription renewal has been processed successfully."
    selected_ar = "تمت معالجة تجديد اشتراكك بنجاح."

col_in_en, col_in_ar = st.columns(2)

with col_in_en:
    english_input = st.text_area(
        "English Text:",
        value=selected_en,
        height=130,
        help="Type or paste English text here."
    )
    st.caption(f"Length: {len(english_input)} characters | {len(english_input.split())} words")

with col_in_ar:
    arabic_input = st.text_area(
        "Arabic Text (نص عربي):",
        value=selected_ar,
        height=130,
        help="Type or paste Arabic text here."
    )
    st.caption(f"Length: {len(arabic_input)} characters | {len(arabic_input.split())} words")

# Run Analysis
with st.spinner("Analyzing tokenization with tiktoken and AraBERT..."):
    # Tiktoken analysis
    tt_en = analyze_with_tiktoken(english_input, encoding_name)
    tt_ar = analyze_with_tiktoken(arabic_input, encoding_name)
    tt_inflation = calculate_inflation_factor(tt_ar["token_count"], tt_en["token_count"])
    tt_interp = get_inflation_interpretation(tt_inflation)

    # Hugging Face AraBERT analysis
    hf_en = analyze_with_hf(english_input, hf_model_name, include_special_tokens=False)
    hf_ar = analyze_with_hf(arabic_input, hf_model_name, include_special_tokens=False)
    hf_inflation = calculate_inflation_factor(hf_ar["token_count"], hf_en["token_count"])
    hf_interp = get_inflation_interpretation(hf_inflation)

# ---------------------------------------------------------------------------
# Section 2: Dynamic Token Counts & Inflation Factor
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. Measured Token Counts & Inflation Metrics")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("#### GPT-Style Tokenizer (`tiktoken` / `cl100k_base`)")
    c1, c2, c3 = st.columns(3)
    c1.metric("English Tokens", f"{tt_en['token_count']}")
    c2.metric("Arabic Tokens", f"{tt_ar['token_count']}")
    inflation_display = f"{tt_inflation:.2f}x" if tt_inflation is not None else "N/A"
    c3.metric("Arabic Inflation", inflation_display, delta=tt_interp["badge"])
    st.caption(f"{tt_interp['description']}")

with col_m2:
    st.markdown(f"#### Arabic-Focused Tokenizer (`{hf_model_name.split('/')[-1]}`)")
    c1, c2, c3 = st.columns(3)
    c1.metric("English Tokens", f"{hf_en['token_count']}")
    c2.metric("Arabic Tokens", f"{hf_ar['token_count']}")
    hf_inflation_display = f"{hf_inflation:.2f}x" if hf_inflation is not None else "N/A"
    c3.metric("Arabic Inflation", hf_inflation_display, delta=hf_interp["badge"])
    st.caption(f"{hf_interp['description']}")

# Important Safety Note
st.info(
    "**Key Insight:** Token inflation is **not** a universal property of Arabic grammar or language. "
    "Rather, it is an empirical outcome of how tokenizer vocabularies were trained. Because GPT tokenizers "
    "like `cl100k_base` allocate most vocabulary slots to English subwords, Arabic words are frequently "
    "fragmented into smaller subwords or raw UTF-8 bytes."
)

# ---------------------------------------------------------------------------
# Section 3: Tiktoken Detailed Breakdown
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("3. Tiktoken Subword Representation (`cl100k_base`)")
st.markdown(
    "Inspect the exact subwords and token IDs generated by OpenAI's Byte-Pair Encoding. "
    "Notice how English words often map to whole words, whereas Arabic text can be split into smaller letter chunks."
)

tt_col_en, tt_col_ar = st.columns(2)

with tt_col_en:
    st.markdown(f"**English Subwords ({tt_en['token_count']} tokens):**")
    if tt_en["subwords"]:
        chips_html = "".join(
            f'<span class="token-chip token-chip-en">{repr(sub)[1:-1]}<span class="token-id">[{tid}]</span></span>'
            for sub, tid in zip(tt_en["subwords"], tt_en["token_ids"])
        )
        st.markdown(f'<div class="subword-container">{chips_html}</div>', unsafe_allow_html=True)
    else:
        st.write("*(No tokens - enter text above)*")

    with st.expander("Raw Token IDs (English)"):
        st.write(tt_en["token_ids"])

with tt_col_ar:
    st.markdown(f"**Arabic Subwords ({tt_ar['token_count']} tokens):**")
    if tt_ar["subwords"]:
        chips_html = "".join(
            f'<span class="token-chip token-chip-ar">{sub}<span class="token-id">[{tid}]</span></span>'
            for sub, tid in zip(tt_ar["subwords"], tt_ar["token_ids"])
        )
        st.markdown(f'<div class="subword-container">{chips_html}</div>', unsafe_allow_html=True)
    else:
        st.write("*(No tokens - enter text above)*")

    with st.expander("Raw Token IDs (Arabic)"):
        st.write(tt_ar["token_ids"])

# ---------------------------------------------------------------------------
# Section 4: Hugging Face / AraBERT Detailed Breakdown
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader(f"4. Hugging Face Arabic Tokenizer (`{hf_model_name.split('/')[-1]}`)")
st.markdown(
    "AraBERT uses WordPiece tokenization trained specifically on Arabic corpora. "
    "Tokens prefixed with `##` indicate subword continuations attached to the previous root/token."
)

hf_col_en, hf_col_ar = st.columns(2)

with hf_col_en:
    st.markdown(f"**English Subwords ({hf_en['token_count']} tokens):**")
    if hf_en["subwords"]:
        chips_html = "".join(
            f'<span class="token-chip token-chip-en">{tok}<span class="token-id">[{tid}]</span></span>'
            for tok, tid in zip(hf_en["subwords"], hf_en["token_ids"])
        )
        st.markdown(f'<div class="subword-container">{chips_html}</div>', unsafe_allow_html=True)
    else:
        st.write("*(No tokens)*")
    st.caption("Notice: AraBERT was not optimized for English, so English words are broken down into more pieces.")

with hf_col_ar:
    st.markdown(f"**Arabic Subwords ({hf_ar['token_count']} tokens):**")
    if hf_ar["subwords"]:
        chips_html = "".join(
            f'<span class="token-chip token-chip-ar">{tok}<span class="token-id">[{tid}]</span></span>'
            for tok, tid in zip(hf_ar["subwords"], hf_ar["token_ids"])
        )
        st.markdown(f'<div class="subword-container">{chips_html}</div>', unsafe_allow_html=True)
    else:
        st.write("*(No tokens)*")

    with st.expander("Raw Token IDs (AraBERT Arabic)"):
        st.write(hf_ar["token_ids"])

# ---------------------------------------------------------------------------
# Section 5: Side-by-Side Comparison & Benchmark Table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("5. Side-by-Side Comparison & Benchmark Suite")

# Active selection comparison table
active_df = pd.DataFrame([
    {
        "Tokenizer": "GPT-style (tiktoken cl100k_base)",
        "English Tokens": tt_en["token_count"],
        "Arabic Tokens": tt_ar["token_count"],
        "Arabic Inflation Factor": f"{tt_inflation:.2f}x" if tt_inflation else "N/A",
        "Assessment": tt_interp["badge"]
    },
    {
        "Tokenizer": f"AraBERT ({hf_model_name.split('/')[-1]})",
        "English Tokens": hf_en["token_count"],
        "Arabic Tokens": hf_ar["token_count"],
        "Arabic Inflation Factor": f"{hf_inflation:.2f}x" if hf_inflation else "N/A",
        "Assessment": hf_interp["badge"]
    }
])
st.markdown("##### Current Text Comparison:")
st.dataframe(active_df, use_container_width=True, hide_index=True)

# Full Benchmark Table across Banking & News
with st.expander("View Complete Benchmark on All Built-in Banking & News Samples", expanded=False):
    st.markdown("This table calculates token counts and inflation dynamically across all paired dataset samples:")
    benchmark_df = benchmark_built_in_samples(hf_model_name)
    st.dataframe(
        benchmark_df[["Domain", "Example", "Tokenizer", "English Tokens", "Arabic Tokens", "Inflation Factor"]],
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------------------------
# Section 6: Visualizations
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("6. Token Count Visualizations")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("##### Current Text: English vs Arabic Token Counts")
    chart_data = pd.DataFrame({
        "Tokenizer": [
            "GPT (cl100k)", "GPT (cl100k)",
            "AraBERT", "AraBERT"
        ],
        "Language": ["English", "Arabic", "English", "Arabic"],
        "Tokens": [
            tt_en["token_count"], tt_ar["token_count"],
            hf_en["token_count"], hf_ar["token_count"]
        ]
    })
    
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("Tokenizer:N", title="Tokenizer Model"),
        y=alt.Y("Tokens:Q", title="Number of Tokens"),
        color=alt.Color(
            "Language:N",
            scale=alt.Scale(domain=["English", "Arabic"], range=["#0284c7", "#d97706"])
        ),
        xOffset="Language:N",
        tooltip=["Tokenizer", "Language", "Tokens"]
    ).properties(height=320)
    
    st.altair_chart(bar_chart, use_container_width=True)

with chart_col2:
    st.markdown("##### Inflation Factor by Model (Current Text)")
    inf_data = pd.DataFrame({
        "Tokenizer": ["GPT (cl100k_base)", "AraBERT"],
        "Inflation Factor": [tt_inflation or 0.0, hf_inflation or 0.0]
    })
    
    inf_chart = alt.Chart(inf_data).mark_bar(size=40).encode(
        x=alt.X("Tokenizer:N", title="Tokenizer"),
        y=alt.Y("Inflation Factor:Q", title="Inflation Factor (Arabic/English)"),
        color=alt.condition(
            alt.datum["Inflation Factor"] > 1.0,
            alt.value("#ea580c"),  # orange/red if > 1.0
            alt.value("#16a34a")   # green if <= 1.0
        ),
        tooltip=["Tokenizer", "Inflation Factor"]
    ).properties(height=320)
    
    # Add a horizontal reference line at 1.0x (parity)
    rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
        strokeDash=[5, 5], color="gray", strokeWidth=2
    ).encode(y="y:Q")
    
    st.altair_chart(inf_chart + rule, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 7: Educational Deep Dive
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("7. Deep Dive: Why Does Tokenization Matter for LLMs?")

with st.expander("Why Tokenization Matters for Cost & Context Windows", expanded=True):
    st.markdown(
        """
        ### 1. The Economics of Tokens (API Pricing)
        LLM cloud providers (OpenAI, Anthropic, Google, etc.) charge **per million tokens**.
        If an Arabic sentence takes **1.5x to 2.0x more tokens** than an equivalent English sentence:
        - Running a customer support chatbot in Arabic can cost **50% to 100% more** for the exact same semantic payload.
        - Embedding models and vector databases require higher dimensional storage and higher processing compute.

        ### 2. Context Window Consumption
        LLMs have strict maximum context windows (e.g., 8k, 32k, 128k tokens).
        When Arabic text produces higher token counts:
        - You can fit **fewer historical chat turns** in memory.
        - Retrieval-Augmented Generation (RAG) chunk limits hold **fewer words of Arabic** than English.

        ### 3. Subwords are Computational Units, Not Linguistic Roots
        In linguistics, Arabic is a root-and-pattern (semitic) language where prefixes (like conjunctions 'و' or prepositions 'ب')
        attach directly to nouns. Standard English-centric tokenizers often split Arabic into arbitrary character n-grams or
        even individual UTF-8 bytes rather than meaningful morphological constituents.

        ### 4. Arabic Linguistic Realities
        - **Modern Standard Arabic (MSA):** Formal written Arabic used in banking and news.
        - **Dialects (Khaliji, Egyptian, Levantine, Maghrebi):** Spoken varieties that introduce regional vocabulary and slang.
        - **Arabizi (Franco-Arabic):** Arabic written in Latin characters with numerals (e.g., '3' for 'ع'), creating entirely different tokenization patterns.
        - **Spelling & Diacritics (Tashkeel):** Adding vowels/tashkeel dramatically increases token counts on standard BPE tokenizers.
        """
    )
