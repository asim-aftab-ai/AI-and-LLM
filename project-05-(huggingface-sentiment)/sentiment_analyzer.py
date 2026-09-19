"""Sentiment analysis logic using Hugging Face Transformers pipeline.

This module encapsulates model loading and inference, keeping them completely
decoupled from any user interface logic.
"""

from transformers import pipeline

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Cached pipeline instance to avoid reloading the model repeatedly
_pipeline = None


def get_sentiment_pipeline():
    """Load and return the cached Hugging Face sentiment-analysis pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
    return _pipeline


def analyze_sentences(sentences):
    """Analyze sentiment for a list of input sentences.

    Args:
        sentences (list of str): The text sentences to analyze.

    Returns:
        list of dict: Prediction results containing sentence, label, and score.
    """
    classifier = get_sentiment_pipeline()
    raw_results = classifier(sentences)

    processed_results = []
    for sentence, result in zip(sentences, raw_results):
        processed_results.append({
            "sentence": sentence,
            "label": result["label"],
            "score": float(result["score"])
        })

    return processed_results


def analyze_single_sentence(sentence: str):
    """Analyze sentiment for a single input sentence.

    Args:
        sentence (str): The text sentence to analyze.

    Returns:
        dict: Prediction result containing sentence, label, and score.
    """
    results = analyze_sentences([sentence])
    return results[0] if results else None

