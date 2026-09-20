"""Semantic Similarity Engine.

This module encapsulates model loading, text embedding generation using
mean pooling over transformer token embeddings, cosine similarity computation,
and human-readable score interpretation.
"""

from typing import Tuple, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = MODEL_ID) -> Tuple[AutoTokenizer, AutoModel]:
    """Load the tokenizer and transformer backbone for sentence embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute sentence representation by averaging token embeddings weighted by attention mask."""
    token_embeddings = model_output[0]  # First element contains hidden state embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def compute_embedding(text: str, tokenizer: AutoTokenizer, model: AutoModel) -> torch.Tensor:
    """Generate a normalized 1D embedding vector for the provided text."""
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(**encoded)
        sentence_emb = mean_pooling(output, encoded["attention_mask"])
        normalized_emb = F.normalize(sentence_emb, p=2, dim=1)
    return normalized_emb


def compute_similarity(
    text_a: str,
    text_b: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
) -> float:
    """Compute cosine similarity between Text A and Text B.

    Returns:
        float: Cosine similarity score bounded between 0.0 and 1.0.
    """
    emb_a = compute_embedding(text_a, tokenizer, model)
    emb_b = compute_embedding(text_b, tokenizer, model)

    # Cosine similarity between unit-normalized vectors is simply their dot product
    raw_sim = F.cosine_similarity(emb_a, emb_b).item()

    # Clamp to [0.0, 1.0] for intuitive percentage/bounded representation
    bounded_sim = max(0.0, min(1.0, raw_sim))
    return round(bounded_sim, 4)


def interpret_score(score: float) -> Dict[str, str]:
    """Provide an intuitive interpretation and educational explanation for a similarity score."""
    if score >= 0.80:
        label = "Very High Semantic Similarity"
        badge = "VERY HIGH"
        color = "#22c55e"  # Green
        explanation = (
            "These two texts express very similar meanings. They may use different words "
            "or syntactic structures, but their underlying concepts and core messages "
            "are closely aligned."
        )
    elif score >= 0.60:
        label = "High Semantic Similarity"
        badge = "HIGH"
        color = "#3b82f6"  # Blue
        explanation = (
            "These two texts share substantial semantic overlap. They address the same general "
            "subject with compatible context, though some specific details or framing may differ."
        )
    elif score >= 0.40:
        label = "Moderate Semantic Similarity"
        badge = "MODERATE"
        color = "#eab308"  # Amber
        explanation = (
            "These two texts have related themes or touch on overlapping domains, "
            "but they diverge in their specific assertions, tone, or primary focal points."
        )
    elif score >= 0.20:
        label = "Low Semantic Similarity"
        badge = "LOW"
        color = "#f97316"  # Orange
        explanation = (
            "These two texts exhibit minimal semantic connection. They might share a general "
            "context or isolated terms, but convey fundamentally different ideas."
        )
    else:
        label = "Very Low Semantic Similarity"
        badge = "VERY LOW"
        color = "#ef4444"  # Red
        explanation = (
            "These two texts appear to discuss completely different topics with no discernible "
            "semantic relationship or conceptual overlap."
        )

    return {
        "score": f"{score:.2f}",
        "score_float": score,
        "label": label,
        "badge": badge,
        "color": color,
        "explanation": explanation,
    }
