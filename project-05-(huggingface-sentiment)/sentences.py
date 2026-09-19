"""Predefined sentences for sentiment analysis demonstration.

This module provides a fixed list of 10 English sentences representing
different sentiment profiles (positive, negative, and neutral/factual).
"""

SENTENCES = [
    "I absolutely loved the quality of this product.",
    "The customer service was excellent and very helpful.",
    "This is one of the best experiences I have ever had.",
    "The delivery was fast and everything arrived safely.",
    "I am very disappointed with the service.",
    "The product broke after only two days.",
    "The experience was frustrating and completely unacceptable.",
    "The package arrived yesterday in the afternoon.",
    "The company sent me an email about my order.",
    "The restaurant was okay, but nothing special."
]


def get_predefined_sentences():
    """Return the predefined list of 10 sentences."""
    return list(SENTENCES)
