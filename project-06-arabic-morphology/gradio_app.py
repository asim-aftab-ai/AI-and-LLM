"""Bilingual Gradio Text Summarizer Interface.

A lightweight educational web interface built with Gradio demonstrating extractive
text summarization for Arabic and English text with side-by-side English translations.
"""

import gradio as gr
from summarizer import summarize_text
from translator import translate_to_english

# Educational sample texts for quick testing
SAMPLE_ARABIC = (
    "اللغة العربية هي إحدى أكثر اللغات انتشاراً في العالم، حيث يتحدث بها أكثر من 400 مليون نسمة "
    "في الشرق الأوسط وشمال أفريقيا. وتتميز اللغة العربية بنظام صرفي فريد يعتمد على الجذور والأوزان، "
    "مما يسمح بتوليد مئات الكلمات المشتقة من جذر ثلاثي واحد. يعتبر التحليل الصرفي مدخلاً رئيسياً "
    "لفهم النصوص العربية ومعالجتها آلياً. وتساهم تقنيات الذكاء الاصطناعي الحديثة في تحسين دقة "
    "البحث الذكي وأنظمة الإجابة عن الأسئلة باللغة العربية."
)

SAMPLE_ENGLISH = (
    "Artificial intelligence is rapidly transforming modern computational systems and software development. "
    "Natural language processing is a specialized subfield of AI focused on enabling computers to understand, "
    "interpret, and generate human language. Text summarization is an essential NLP task that extracts or "
    "abstracts core information from extensive documents into concise overviews. Extractive summarization "
    "selects the most informative sentences directly from the original source without rewriting. "
    "These techniques significantly reduce reading time and improve information retrieval in enterprise workflows."
)


def process_bilingual_summarization(text: str):
    """Summarize text extractively and generate corresponding English translations.

    Args:
        text: Input text (Arabic or English).

    Returns:
        tuple: (english_meaning, arabic_summary, english_summary, orig_count, summ_count, compression_ratio)
    """
    raw_text = text.strip() if text else ""
    if not raw_text:
        return "", "", "", "0", "0", "0.0%"

    # 1. Generate extractive summary and metrics
    orig_count, summary, summ_count, comp_ratio = summarize_text(raw_text)

    # 2. Translate original text into English
    english_meaning = translate_to_english(raw_text)

    # 3. Translate generated summary into English (guaranteeing exact correspondence)
    english_summary = translate_to_english(summary) if summary else ""

    return (
        english_meaning,
        summary,
        english_summary,
        str(orig_count),
        str(summ_count),
        comp_ratio,
    )


def create_gradio_interface():
    """Build and configure the Bilingual Gradio Interface application."""
    # Input component: Arabic/Original Text
    input_component = gr.Textbox(
        lines=6,
        label="Arabic Text",
        placeholder="Enter Arabic (or English) text here...",
        value=SAMPLE_ARABIC,
    )

    # Output components reflecting the bilingual educational layout
    output_components = [
        gr.Textbox(lines=4, label="English Meaning", interactive=False),
        gr.Textbox(lines=4, label="Arabic Summary", interactive=False),
        gr.Textbox(lines=4, label="English Meaning of Summary", interactive=False),
        gr.Textbox(label="Original Word Count", interactive=False),
        gr.Textbox(label="Summary Word Count", interactive=False),
        gr.Textbox(label="Compression Ratio", interactive=False),
    ]

    # Explicit Action Button using gr.Button
    summarize_button = gr.Button("Summarize", variant="primary")

    # Construct the Gradio Interface
    interface = gr.Interface(
        fn=process_bilingual_summarization,
        inputs=input_component,
        outputs=output_components,
        title="Text Summarizer (Bilingual Arabic-English)",
        description=(
            "Enter Arabic text to generate an extractive Arabic summary alongside "
            "direct English translations to aid language learners and NLP students."
        ),
        submit_btn=summarize_button,
        examples=[
            [SAMPLE_ARABIC],
            [SAMPLE_ENGLISH],
        ],
        flagging_mode="never",
    )

    return interface


if __name__ == "__main__":
    demo = create_gradio_interface()
    # Launch with public sharing enabled as required for learning demonstration
    demo.launch(share=True)
