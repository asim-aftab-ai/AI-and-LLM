"""LLM Client Module.

Handles interaction with OpenRouter via the OpenAI-compatible client,
system prompt grounding, and context-injected answer generation.
"""

from typing import List, Dict, Any, Generator, Tuple
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash"

SYSTEM_PROMPT = """You are an AI PDF Research Assistant. Your role is to answer questions strictly and accurately based on the provided document excerpts.

CRITICAL INSTRUCTIONS:
1. Ground your answers ONLY in the provided document context.
2. If the answer cannot be found or substantiated from the document context, explicitly state:
   "Based on the provided document, this information was not found."
   Do NOT invent, extrapolate, or hallucinate facts not present in the document.
3. Be concise, objective, and clear.
4. When relevant, reference the source page or section indicated in the excerpt headers.
5. If the user asks a follow-up question, use the conversation history to understand context, but still answer strictly using the document context.
"""


def get_llm_client(api_key: str) -> OpenAI:
    """Create an OpenAI client configured for OpenRouter."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def generate_grounded_answer(
    client: OpenAI,
    model: str,
    question: str,
    context_text: str,
    conversation_history: List[Dict[str, str]],
) -> str:
    """Generate an answer grounded strictly in the retrieved context using OpenRouter.

    Args:
        client: Initialized OpenAI client instance.
        model: OpenRouter model identifier.
        question: Current user question.
        context_text: Formatted excerpts retrieved from the document.
        conversation_history: Prior turns for multi-turn conversational context.

    Returns:
        str: Model response text.
    """
    # Build payload messages
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Append recent conversation history
    for msg in conversation_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    # User message incorporating the retrieved context and question
    user_prompt = f"""DOCUMENT CONTEXT:
{context_text}

---
USER QUESTION:
{question}

Provide an answer strictly supported by the document context above."""

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,  # Low temperature for factual consistency
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()
