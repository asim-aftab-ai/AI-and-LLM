"""Streamlit frontend for Hugging Face sentiment analysis demonstration.

Displays 10 predefined sentences, triggers pipeline inference upon user action,
presents predicted sentiment labels and confidence scores in an enhanced table,
and provides an interactive playground for custom sentence evaluation.
"""

import streamlit as st
import pandas as pd
from sentences import get_predefined_sentences
from sentiment_analyzer import analyze_sentences, analyze_single_sentence, MODEL_NAME


def main():
    st.set_page_config(
        page_title="Hugging Face Sentiment Analysis",
        page_icon="🤗",
        layout="wide"
    )

    # Sidebar Information & Settings
    st.sidebar.title("🤗 Model Information")
    st.sidebar.markdown(
        f"""
        **Architecture:** DistilBERT  
        **Model ID:** `{MODEL_NAME}`  
        **Parameters:** ~66.9 Million  
        **Task:** Text Classification (SST-2)  
        **Framework:** Hugging Face + PyTorch  
        """
    )
    st.sidebar.divider()
    st.sidebar.subheader("Dataset Context")
    st.sidebar.info(
        "Trained on **SST-2 (Stanford Sentiment Treebank)**. "
        "Because SST-2 is a binary dataset, every input is strictly assigned "
        "either `POSITIVE` or `NEGATIVE`."
    )

    # Main Header
    st.title("🤗 Hugging Face Sentiment Analysis")
    st.caption("Practical NLP Sentiment Classification using DistilBERT & Transformers Pipeline")

    st.write(
        "Sentiment analysis is a natural language processing technique that determines "
        "the emotional tone behind a body of text. It classifies input into emotional categories "
        "accompanied by a confidence score indicating the model's certainty."
    )

    tab_benchmark, tab_playground, tab_deep_dive = st.tabs([
        "📋 Predefined Benchmark (10 Sentences)",
        "🧪 Interactive Playground",
        "💡 SST-2 & Architecture Deep Dive"
    ])

    # -----------------------------------------------------------------------
    # TAB 1: Predefined Benchmark
    # -----------------------------------------------------------------------
    with tab_benchmark:
        sentences = get_predefined_sentences()

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Benchmark Sentences")
            st.write("The canonical 10 test sentences covering positive, negative, and neutral phrasing:")

            for index, sentence in enumerate(sentences, start=1):
                st.markdown(f"**{index}.** {sentence}")

        with col_right:
            st.subheader("Run Benchmark")
            st.write("Click below to execute transformer inference on all 10 sentences:")
            run_button = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

        st.divider()

        if run_button:
            with st.spinner("Executing DistilBERT inference pipeline..."):
                try:
                    results = analyze_sentences(sentences)

                    pos_count = sum(1 for r in results if r["label"] == "POSITIVE")
                    neg_count = sum(1 for r in results if r["label"] == "NEGATIVE")
                    avg_conf = sum(r["score"] for r in results) / len(results) * 100

                    # Metrics Summary
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Sentences", len(results))
                    m2.metric("Positive Count", f"{pos_count} ({pos_count/len(results)*100:.0f}%)")
                    m3.metric("Negative Count", f"{neg_count} ({neg_count/len(results)*100:.0f}%)")
                    m4.metric("Average Confidence", f"{avg_conf:.2f}%")

                    st.subheader("Analysis Results")

                    table_rows = []
                    for item in results:
                        is_pos = item["label"] == "POSITIVE"
                        badge = "🟢 POSITIVE" if is_pos else "🔴 NEGATIVE"
                        table_rows.append({
                            "Sentence": item["sentence"],
                            "Sentiment": badge,
                            "Confidence": f"{item['score'] * 100:.2f}%",
                            "Raw Score": round(item["score"], 4)
                        })

                    df = pd.DataFrame(table_rows)
                    st.dataframe(
                        df[["Sentence", "Sentiment", "Confidence"]],
                        use_container_width=True,
                        hide_index=True
                    )

                except Exception as error:
                    st.error(f"An error occurred during sentiment analysis: {error}")

    # -----------------------------------------------------------------------
    # TAB 2: Interactive Playground
    # -----------------------------------------------------------------------
    with tab_playground:
        st.subheader("Test Any Custom Text")
        st.write("Type your own sentence or select a quick preset to evaluate real-time sentiment:")

        sample_inputs = [
            "The product exceeded all my expectations and arrived two days early!",
            "Customer support refused to resolve my issue and kept transferring me.",
            "The battery life is acceptable, though charging takes longer than expected.",
            "I booked a flight ticket to London for next Tuesday."
        ]

        selected_sample = st.selectbox(
            "Or choose a sample input:",
            options=["-- Type custom input below --"] + sample_inputs
        )

        default_text = "" if selected_sample.startswith("--") else selected_sample
        user_sentence = st.text_area(
            "Enter sentence to analyze:",
            value=default_text,
            placeholder="e.g. The flight was delayed but the cabin crew was very polite and helpful.",
            height=90
        )

        if st.button("🔍 Analyze Custom Text", type="primary"):
            cleaned = user_sentence.strip()
            if not cleaned:
                st.warning("Please enter a sentence to analyze.")
            else:
                with st.spinner("Analyzing custom text..."):
                    try:
                        res = analyze_single_sentence(cleaned)
                        label = res["label"]
                        score = res["score"]

                        st.markdown("---")
                        res_col1, res_col2 = st.columns([1, 2])

                        with res_col1:
                            if label == "POSITIVE":
                                st.success(f"### 🟢 {label}")
                            else:
                                st.error(f"### 🔴 {label}")
                            st.metric("Confidence Score", f"{score * 100:.2f}%")

                        with res_col2:
                            st.write("**Analyzed Text:**")
                            st.info(f'"{cleaned}"')
                            st.progress(score, text=f"Certainty: {score * 100:.2f}%")

                    except Exception as error:
                        st.error(f"Error during analysis: {error}")

    # -----------------------------------------------------------------------
    # TAB 3: Architecture & SST-2 Deep Dive
    # -----------------------------------------------------------------------
    with tab_deep_dive:
        st.subheader("💡 Why Neutral Sentences Get Classified as Positive or Negative")
        st.markdown(
            """
            In our 10-sentence benchmark, note sentences such as:
            - *"The package arrived yesterday in the afternoon."*
            - *"The company sent me an email about my order."*

            Even though these are factual or neutral, the model classifies them as `POSITIVE` or `NEGATIVE`.
            
            ### Why does this happen?
            1. **Binary Classification Dataset (SST-2):**
               The Stanford Sentiment Treebank (SST-2) dataset contains only two classes: **Positive** (1) and **Negative** (0). There is no "Neutral" class.
            2. **Softmax Normalization:**
               The final classification layer outputs two logits ($z_{pos}$ and $z_{neg}$). The softmax function normalizes them so $P(pos) + P(neg) = 1.0$. The model is forced to pick whichever probability is higher, even for neutral text.
            3. **DistilBERT Distillation:**
               DistilBERT compresses the original BERT architecture by 40% while retaining 97% of its language understanding capabilities and running 60% faster, making it an ideal choice for interactive web deployment.
            """
        )


if __name__ == "__main__":
    main()

