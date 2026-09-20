"""Arabic Morphological Analysis Learning Interface.

A technical Streamlit educational application demonstrating the end-to-end
Arabic NLP pipeline powered by CAMeL Tools.
"""

import sys
import streamlit as st

from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)

# Configure Streamlit page (Strictly NO emojis)
st.set_page_config(
    page_title="Arabic Morphological Analysis | CAMeL Tools",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Ensure UTF-8 stdout encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


@st.cache_resource(show_spinner=False)
def load_morphology_analyzer():
    """Load and cache the built-in Modern Standard Arabic morphology database."""
    db = MorphologyDB.builtin_db()
    return Analyzer(db)


def apply_custom_css():
    """Inject custom CSS for a professional technical laboratory aesthetic."""
    st.markdown(
        """
        <style>
        /* General styling */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Technical header */
        .tech-header {
            border-left: 4px solid #0284c7;
            padding: 0.8rem 1.2rem;
            background: rgba(2, 132, 199, 0.06);
            border-radius: 0 8px 8px 0;
            margin-bottom: 1.5rem;
        }
        .tech-header h1 {
            font-size: 1.8rem;
            margin: 0 0 0.4rem 0;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .tech-header p {
            margin: 0;
            font-size: 0.95rem;
            color: #94a3b8;
        }

        /* Stage boxes and cards */
        .tech-card {
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1.2rem;
            background: rgba(15, 23, 42, 0.6);
            margin-bottom: 1rem;
        }
        .tech-card-header {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #38bdf8;
            font-weight: 600;
            margin-bottom: 0.6rem;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 0.4rem;
        }

        /* Pipeline flowchart styling */
        .pipeline-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            padding: 1rem 0.5rem;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid #334155;
            border-radius: 8px;
            margin: 1.2rem 0;
            overflow-x: auto;
        }
        .pipeline-step {
            flex: 1;
            min-width: 130px;
            text-align: center;
            padding: 0.75rem 0.5rem;
            border-radius: 6px;
            border: 1px solid #475569;
            background: #0f172a;
            transition: all 0.25s ease;
        }
        .pipeline-step.active {
            border-color: #38bdf8;
            background: rgba(56, 189, 248, 0.12);
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.25);
        }
        .pipeline-step-title {
            font-size: 0.8rem;
            font-weight: 700;
            color: #f1f5f9;
            letter-spacing: 0.02em;
        }
        .pipeline-step-sub {
            font-size: 0.7rem;
            color: #94a3b8;
            margin-top: 0.2rem;
        }
        .pipeline-arrow {
            color: #64748b;
            font-size: 1.2rem;
            font-weight: 700;
            user-select: none;
        }
        .pipeline-arrow.active {
            color: #38bdf8;
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0% { opacity: 0.4; transform: translateX(-2px); }
            50% { opacity: 1; transform: translateX(2px); }
            100% { opacity: 0.4; transform: translateX(-2px); }
        }

        /* Arabic text rendering */
        .arabic-display {
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
            font-size: 1.5rem;
            direction: rtl;
            text-align: right;
            padding: 0.8rem 1rem;
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid #475569;
            border-radius: 6px;
            color: #f8fafc;
            line-height: 1.8;
        }
        .arabic-token-badge {
            display: inline-block;
            direction: rtl;
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
            font-size: 1.15rem;
            padding: 0.35rem 0.8rem;
            margin: 0.25rem;
            background: #1e293b;
            border: 1px solid #0284c7;
            border-radius: 6px;
            color: #e0f2fe;
        }

        /* Analysis table badges */
        .feat-badge {
            display: inline-block;
            font-family: monospace;
            font-size: 0.75rem;
            padding: 0.15rem 0.45rem;
            background: #1e293b;
            border: 1px solid #475569;
            border-radius: 4px;
            color: #cbd5e1;
            margin-right: 0.3rem;
            margin-bottom: 0.2rem;
        }

        /* Comparison columns */
        .comp-box {
            height: 100%;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1.2rem;
            background: #090d16;
        }
        .comp-box.highlight {
            border-color: #0284c7;
            background: rgba(2, 132, 199, 0.04);
        }

        /* Notice tags */
        .tag-implemented {
            display: inline-block;
            font-size: 0.7rem;
            padding: 0.2rem 0.6rem;
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid #22c55e;
            border-radius: 4px;
            font-weight: 600;
        }
        .tag-future {
            display: inline-block;
            font-size: 0.7rem;
            padding: 0.2rem 0.6rem;
            background: rgba(148, 163, 184, 0.15);
            color: #94a3b8;
            border: 1px solid #64748b;
            border-radius: 4px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_1_header():
    """SECTION 1: Project Explanation."""
    st.markdown(
        """
        <div class="tech-header">
            <h1>Arabic Morphological Analysis with CAMeL Tools</h1>
            <p>
                This system demonstrates how Arabic text can be processed using Arabic-specific
                linguistic analysis before being passed to downstream AI systems.
                CAMeL Tools acts as the Arabic linguistic processing layer that extracts structural,
                grammatical, and morphological properties from raw text tokens.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_2_input():
    """SECTION 2: Arabic Input."""
    st.markdown("### 1. Arabic Input")
    default_text = "الطلاب يدرسون اللغة العربية"

    col_input, col_meta = st.columns([3, 1])
    with col_input:
        sentence = st.text_input(
            label="Enter Arabic sentence to analyze:",
            value=default_text,
            help="Type or paste any Arabic sentence. The pipeline will automatically re-run.",
        )
    with col_meta:
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94a3b8; padding-top: 1.8rem;">
                Status: Live pipeline active<br>
                Engine: CAMeL Tools MSA Database
            </div>
            """,
            unsafe_allow_html=True,
        )

    return sentence.strip() if sentence.strip() else default_text


def render_section_3_pipeline_flow(active_stage):
    """SECTION 3: Animated / Moving Data Flow."""
    st.markdown("### 2. Pipeline Data Flow")
    st.caption(
        "Visual progression showing how Arabic text transitions through linguistic layers."
    )

    stages = [
        ("Input", "Arabic Text", 1),
        ("Tokenization", "Word Units", 2),
        ("Morphology", "CAMeL Tools", 3),
        ("Feature Extraction", "Root/Lemma/POS", 4),
        ("Normalization", "Orthographic Rules", 5),
        ("Representation", "Downstream AI", 6),
    ]

    cols = st.columns(len(stages) * 2 - 1)
    for i, (title, sub, step_num) in enumerate(stages):
        col_idx = i * 2
        is_active = step_num <= active_stage
        border_color = "#38bdf8" if is_active else "#334155"
        bg_color = "rgba(56, 189, 248, 0.12)" if is_active else "rgba(15, 23, 42, 0.6)"
        text_color = "#f1f5f9" if is_active else "#94a3b8"

        cols[col_idx].markdown(
            f"""
            <div style="text-align: center; padding: 0.6rem 0.3rem; border: 1px solid {border_color};
                        background: {bg_color}; border-radius: 6px;">
                <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600;">STAGE {step_num}</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: {text_color};">{title}</div>
                <div style="font-size: 0.68rem; color: #64748b;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if i < len(stages) - 1:
            arrow_col = col_idx + 1
            arrow_color = "#38bdf8" if step_num < active_stage else "#475569"
            cols[arrow_col].markdown(
                f"""
                <div style="text-align: center; padding-top: 1.1rem; color: {arrow_color}; font-weight: bold; font-size: 1.1rem;">
                    &rarr;
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_section_4_tokenization(sentence):
    """SECTION 4: Tokenization."""
    st.markdown("### 3. Tokenization Stage")
    tokens = simple_word_tokenize(sentence)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-card-header">Raw Input Sentence</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="arabic-display">{sentence}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">
                    Character Count: {len(sentence)} | Whitespace Segments: {len(sentence.split())}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-card-header">Extracted Tokens</div>
            """,
            unsafe_allow_html=True,
        )
        badges_html = "".join(
            f'<span class="arabic-token-badge">{tok}</span>' for tok in tokens
        )
        st.markdown(
            f'<div style="min-height: 52px;">{badges_html}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">
                    Token Count: {len(tokens)} | Tokenizer: CAMeL Tools simple_word_tokenize
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Tokenization separates the sentence into units that can be processed individually."
    )

    return tokens


def render_section_5_and_6_morphology(tokens, analyzer):
    """SECTIONS 5 & 6: Morphological Analysis & CAMeL Tools Explanation."""
    st.markdown("### 4. Morphological Analysis (CAMeL Tools)")

    st.markdown(
        """
        <div style="padding: 0.8rem 1rem; border-left: 3px solid #38bdf8; background: rgba(56, 189, 248, 0.05); margin-bottom: 1.2rem; font-size: 0.9rem;">
            <strong>Linguistic Principle:</strong> One Arabic surface word can have multiple valid morphological analyses depending on unwritten vowels (diacritics), grammatical case, and semantic context. Morphological analysis exposes all possible valid configurations. Morphological disambiguation (a downstream task) selects the single most probable analysis given sentence context.
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_keys = [
        "gen",  # gender
        "num",  # number
        "per",  # person
        "asp",  # aspect
        "vox",  # voice
        "mod",  # mood
        "cas",  # case
        "stt",  # state
        "prc0",  # proclitic 0
        "enc0",  # enclitic 0
    ]

    all_analyses_by_token = {}

    token_tabs = st.tabs([f"Token: {t}" for t in tokens])

    for tab, token in zip(token_tabs, tokens):
        with tab:
            analyses = analyzer.analyze(token)
            all_analyses_by_token[token] = analyses

            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <div>
                        <span style="font-size: 1.3rem; font-weight: bold; font-family: 'Segoe UI', Arial; direction: rtl;">{token}</span>
                        <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 0.8rem;">({len(analyses)} candidate analyses returned)</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #38bdf8; font-family: monospace;">
                        Status: Analyzed via MorphologyDB (calima-msa-r13)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not analyses:
                st.warning(
                    f"No morphological analyses found for token: '{token}' (out of vocabulary or non-standard orthography)."
                )
                continue

            for idx, analysis in enumerate(analyses, 1):
                diac = analysis.get("diac", "N/A")
                root = analysis.get("root", "N/A")
                lemma = analysis.get("lex", "N/A")
                pos = analysis.get("pos", "N/A")
                gloss = analysis.get("gloss", "N/A")

                active_features = {
                    k: analysis[k]
                    for k in feature_keys
                    if k in analysis and analysis[k] != "na"
                }

                feat_badges = "".join(
                    f'<span class="feat-badge">{k}={v}</span>'
                    for k, v in active_features.items()
                )

                with st.expander(
                    f"Analysis {idx} of {len(analyses)}: Diacritized: {diac} | Lemma: {lemma} | POS: {pos}",
                    expanded=(idx == 1),
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(
                        f"**Diacritized Form**<br><span style='font-size:1.15rem; font-family:\"Segoe UI\"; direction:rtl;'>{diac}</span>",
                        unsafe_allow_html=True,
                    )
                    c2.markdown(
                        f"**Root**<br><span style='font-size:1.15rem; font-family:\"Segoe UI\"; direction:rtl;'>{root}</span>",
                        unsafe_allow_html=True,
                    )
                    c3.markdown(
                        f"**Lemma (Dictionary Entry)**<br><span style='font-size:1.15rem; font-family:\"Segoe UI\"; direction:rtl;'>{lemma}</span>",
                        unsafe_allow_html=True,
                    )
                    c4.markdown(f"**Part of Speech**<br>`{pos}`", unsafe_allow_html=True)

                    st.markdown(
                        f"**English Gloss:** `{gloss}`", unsafe_allow_html=True
                    )
                    st.markdown(
                        f"**Morphological Features:**<br>{feat_badges if feat_badges else 'None'}",
                        unsafe_allow_html=True,
                    )

    # Section 6: Explanation panel
    st.markdown("#### What CAMeL Tools is doing")
    st.markdown(
        """
        <div class="tech-card">
            <p style="margin-bottom: 0.6rem; color: #f1f5f9;">
                CAMeL Tools does not simply split Arabic text into words. It examines the linguistic structure
                of each word and provides possible morphological analyses.
            </p>
            <p style="margin-bottom: 0; color: #94a3b8;">
                Instead of only seeing a word as a string of characters, the system can inspect information such
                as its root, lemma, part of speech, and grammatical features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return all_analyses_by_token


def render_section_7_comparison():
    """SECTION 7: Normal vs Arabic-aware Processing."""
    st.markdown("### 5. Generic vs Arabic-aware Processing")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="comp-box">
                <div style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 0.8rem; border-bottom: 1px solid #334155; padding-bottom: 0.4rem;">
                    Generic Text Processing
                </div>
                <div style="font-family: monospace; font-size: 0.85rem; color: #cbd5e1; line-height: 1.8;">
                    Arabic text<br>
                    &darr;<br>
                    Tokens (whitespace / BPE subwords)<br>
                    &darr;<br>
                    String representation<br>
                    &darr;<br>
                    Downstream AI
                </div>
                <div style="margin-top: 1rem; font-size: 0.82rem; color: #94a3b8; border-top: 1px solid #1e293b; padding-top: 0.6rem;">
                    Sees Arabic as arbitrary character strings or byte-pair tokens without understanding Arabic grammatical structure or morphological roots.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="comp-box highlight">
                <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 0.8rem; border-bottom: 1px solid #0284c7; padding-bottom: 0.4rem;">
                    Arabic-aware Processing
                </div>
                <div style="font-family: monospace; font-size: 0.85rem; color: #e0f2fe; line-height: 1.8;">
                    Arabic text<br>
                    &darr;<br>
                    Arabic tokenization<br>
                    &darr;<br>
                    Morphological analysis<br>
                    &darr;<br>
                    Root / Lemma / POS / Features<br>
                    &darr;<br>
                    Normalization<br>
                    &darr;<br>
                    Arabic-aware representation<br>
                    &darr;<br>
                    Downstream AI
                </div>
                <div style="margin-top: 1rem; font-size: 0.82rem; color: #bae6fd; border-top: 1px solid rgba(2, 132, 199, 0.3); padding-top: 0.6rem;">
                    Exposes explicit linguistic structures: extracts shared trilateral roots, lemmatizes clitic prefixes/suffixes, and identifies grammatical categories.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="margin-top: 0.8rem; font-size: 0.88rem; color: #cbd5e1;">
            <strong>Key Distinction:</strong> Generic processing mainly sees text as tokens. Arabic-aware processing can additionally represent linguistic information that is specific to Arabic. Generic NLP is capable of accepting Arabic text, but Arabic-aware preprocessing provides explicit linguistic indicators that standard tokenizers obscure.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_8_why_morphology_matters(all_analyses):
    """SECTION 8: Why Morphology Matters for Arabic."""
    st.markdown("### 6. Why Morphology Matters for Arabic")

    st.markdown(
        """
        Arabic is a root-and-pattern language. Words are generated by fitting roots into morphological templates (patterns). Multiple distinct surface forms can share the same root and core semantic foundation.
        """
    )

    # Collect actual roots and lemmas from current sentence analysis
    collected_relationships = []
    for token, analyses in all_analyses.items():
        if analyses:
            first = analyses[0]
            collected_relationships.append(
                {
                    "token": token,
                    "root": first.get("root", "N/A"),
                    "lemma": first.get("lex", "N/A"),
                    "pos": first.get("pos", "N/A"),
                }
            )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-card-header">Linguistic Principle: Root & Pattern Generation</div>
                <div style="font-family: monospace; font-size: 0.85rem; color: #cbd5e1; line-height: 1.8;">
                    Different surface forms<br>
                    &darr;<br>
                    Shared linguistic information<br>
                    &darr;<br>
                    Root / lemma / morphological relationships
                </div>
                <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.8rem;">
                    For example, the verb form <em>يدرسون</em> (they study) and noun form <em>دِرَاسَة</em> (study/learning) share the identical root <strong>د.ر.س</strong>. Traditional keyword search misses this relationship without morphological awareness.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-card-header">Live Analysis of Current Sentence</div>
            """,
            unsafe_allow_html=True,
        )
        if collected_relationships:
            for item in collected_relationships:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1e293b; padding: 0.3rem 0; font-size: 0.85rem;">
                        <span style="font-family: 'Segoe UI'; font-size: 1rem; direction: rtl;">{item['token']}</span>
                        <span style="color: #38bdf8;">Root: <strong>{item['root']}</strong></span>
                        <span style="color: #94a3b8;">Lemma: {item['lemma']}</span>
                        <span style="color: #64748b;">({item['pos']})</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


def render_section_9_normalization(sentence):
    """SECTION 9: Arabic Normalization."""
    st.markdown("### 7. Arabic Normalization")

    st.markdown(
        """
        Arabic text frequently contains orthographic variations and diacritical marks that can cause matching failures in standard retrieval systems.
        Normalization standardizes these variations without aggressively stripping essential linguistic signals.
        """
    )

    # Implemented normalization operations using installed CAMeL Tools utils
    norm_dediac = dediac_ar(sentence)
    norm_alef = normalize_alef_ar(sentence)
    norm_teh = normalize_teh_marbuta_ar(sentence)
    norm_maksura = normalize_alef_maksura_ar(sentence)

    # Combined standard normalization
    norm_all = normalize_alef_maksura_ar(
        normalize_teh_marbuta_ar(normalize_alef_ar(norm_dediac))
    )

    st.markdown(
        """
        <div style="font-family: monospace; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.8rem;">
            Original Arabic &rarr; Normalization Stage &rarr; Normalized Representation
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-card-header">
                    Dediacritization <span class="tag-implemented">Implemented</span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.4rem;">
                    Removes short vowel marks (Harakat) such as Fatha, Damma, Kasra, and Sukun.
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="arabic-display" style="font-size: 1.1rem;">{norm_dediac}</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-card-header">
                    Alef Normalization <span class="tag-implemented">Implemented</span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.4rem;">
                    Standardizes variant forms of Alef (أ, إ, آ) into bare Alef (ا).
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="arabic-display" style="font-size: 1.1rem;">{norm_alef}</div></div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-card-header">
                    Full Pipeline Normalization <span class="tag-implemented">Implemented</span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.4rem;">
                    Combined dediacritization, Alef, Teh Marbuta, and Alef Maksura standardization.
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="arabic-display" style="font-size: 1.1rem;">{norm_all}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">
            <span class="tag-future">Conceptual stage &mdash; implementation will be added later</span>: Advanced phonetic normalization, dialect-to-MSA orthographic normalization, and context-preserving punctuation normalization.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_10_future_architecture():
    """SECTION 10: Where this fits in future Arabic AI systems."""
    st.markdown("### 8. Where This Fits in Future Arabic AI Systems")

    st.markdown(
        """
        <div class="tech-card">
            <div class="tech-card-header">Complete Arabic AI Architecture Blueprint</div>
            <div style="font-family: monospace; font-size: 0.85rem; color: #cbd5e1; line-height: 2;">
                Arabic Documents<br>
                &darr;<br>
                Arabic Preprocessing<br>
                &darr;<br>
                <span style="color: #38bdf8; font-weight: bold;">CAMeL Tools (Linguistic Processing Layer &mdash; Implemented in this project)</span><br>
                &darr;<br>
                Arabic-aware Representation (Roots, Lemmas, Morphological Vectors)<br>
                &darr;<br>
                Hybrid Retrieval (Lexical BM25 on Lemmas + Dense Semantic Embeddings)<br>
                &darr;<br>
                RAG (Retrieval-Augmented Generation)<br>
                &darr;<br>
                Arabic LLM (Context-enriched prompt injection)<br>
                &darr;<br>
                Enterprise AI Systems
            </div>
            <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid #1e293b; font-size: 0.85rem; color: #94a3b8;">
                <strong>Scope Clarification:</strong> The current project only implements the Arabic preprocessing and morphological analysis layer. The later stages (Hybrid Retrieval, RAG, Arabic LLM, Enterprise AI) are future projects. The purpose of this architecture diagram is to understand where Arabic-specific NLP sits inside a larger production AI system.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_11_future_use_cases():
    """SECTION 11: Future Use Cases."""
    st.markdown("### 9. Future Applications")

    use_cases = [
        (
            "Arabic Search",
            "Enables lemma-based and root-based indexing so queries match inflected variants across documents.",
        ),
        (
            "RAG Retrieval",
            "Improves document chunk matching by resolving morphological variations between queries and reference texts.",
        ),
        (
            "Document Processing",
            "Extracts structured linguistic entities, parts of speech, and grammatical relations from enterprise archives.",
        ),
        (
            "Arabic Classification",
            "Feeds root and lemma representations into classifiers to reduce vocabulary sparsity and handle dialectal shifts.",
        ),
        (
            "Arabic LLM Preprocessing",
            "Supplies explicit morphological annotations and normalized texts to improve prompt clarity and token economy.",
        ),
        (
            "Enterprise Knowledge Systems",
            "Builds robust Arabic knowledge graphs structured around canonical lemmas and ontological root hierarchies.",
        ),
    ]

    cols = st.columns(3)
    for idx, (title, description) in enumerate(use_cases):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="tech-card" style="min-height: 120px;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.4rem;">
                        {title}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_section_12_technical_transparency():
    """SECTION 12: Technical Transparency."""
    st.markdown("### 10. Technical Transparency & Implementation Scope")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-card-header" style="color: #4ade80; border-color: rgba(34, 197, 94, 0.3);">
                    Currently Implemented
                </div>
                <ul style="font-size: 0.82rem; color: #cbd5e1; padding-left: 1.2rem; line-height: 1.8; margin-bottom: 0;">
                    <li>Interactive Arabic sentence input</li>
                    <li>CAMeL Tools simple_word_tokenize word segmentation</li>
                    <li>Modern Standard Arabic MorphologyDB (calima-msa-r13)</li>
                    <li>CAMeL Tools Analyzer comprehensive analysis execution</li>
                    <li>Inspection of Root, Lemma, Part of Speech, and grammatical features</li>
                    <li>Basic orthographic and diacritic normalization (CAMeL Tools utils)</li>
                    <li>Multi-analysis inspection for polysemous / ambiguous tokens</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-card-header" style="color: #94a3b8; border-color: #334155;">
                    Not Implemented Yet (Future Steps)
                </div>
                <ul style="font-size: 0.82rem; color: #94a3b8; padding-left: 1.2rem; line-height: 1.8; margin-bottom: 0;">
                    <li>Contextual morphological disambiguation (selecting single best analysis)</li>
                    <li>Advanced dialectal normalization (Egyptian, Gulf, Levantine)</li>
                    <li>Dense / sparse hybrid search indexing</li>
                    <li>Retrieval-Augmented Generation (RAG) integration</li>
                    <li>Large Language Model (LLM) fine-tuning or prompting</li>
                    <li>Production API microservice deployment</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    apply_custom_css()

    # Section 1: Project Explanation
    render_section_1_header()

    # Section 2: Arabic Input
    sentence = render_section_2_input()

    # Step progression control
    st.sidebar.markdown("### Pipeline Step Control")
    st.sidebar.caption("Highlight active progression stage:")
    active_stage = st.sidebar.slider(
        label="Pipeline Step Progression",
        min_value=1,
        max_value=6,
        value=4,
        step=1,
        help="Controls the highlighted stage in the pipeline flow diagram.",
    )
    st.sidebar.markdown(
        """
        ---
        **Active Stage Index:**
        1. Arabic Input
        2. Tokenization
        3. Morphological Analysis
        4. Feature Extraction
        5. Normalization
        6. Arabic-aware Representation
        ---
        **Environment Info:**
        - Shared root `.venv`
        - CAMeL Tools 1.6.0
        - Streamlit frontend
        """
    )

    # Section 3: Visual Moving Data Flow
    render_section_3_pipeline_flow(active_stage)

    # Section 4: Tokenization
    tokens = render_section_4_tokenization(sentence)

    # Load cached CAMeL Tools Analyzer
    with st.spinner("Loading CAMeL Tools Morphology Database..."):
        analyzer = load_morphology_analyzer()

    # Sections 5 & 6: Morphological Analysis & Explanation
    all_analyses = render_section_5_and_6_morphology(tokens, analyzer)

    # Section 7: Generic vs Arabic-aware Processing
    render_section_7_comparison()

    # Section 8: Why Morphology Matters
    render_section_8_why_morphology_matters(all_analyses)

    # Section 9: Arabic Normalization
    render_section_9_normalization(sentence)

    # Section 10: Future Architecture
    render_section_10_future_architecture()

    # Section 11: Future Use Cases
    render_section_11_future_use_cases()

    # Section 12: Technical Transparency
    render_section_12_technical_transparency()


if __name__ == "__main__":
    main()
