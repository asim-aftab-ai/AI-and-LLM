"""Retriever Module.

Performs semantic retrieval across document chunks using high-performance
dense vector embeddings and cosine similarity ranking.

Supported Embedding Providers:
1. OpenAI text-embedding-3-small via OpenRouter (Default, 1536 dimensions, SOTA retrieval).
2. BAAI bge-large-en-v1.5 via OpenRouter (1024 dimensions, MTEB leader).
3. Local sentence-transformers/all-MiniLM-L6-v2 (Fallback for offline execution).
"""

import os
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from openai import OpenAI

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class DocumentRetriever:
    """Manages chunk vector indexing and semantic retrieval."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()

        self.chunks: List[Dict[str, Any]] = []
        self.chunk_embeddings: Optional[torch.Tensor] = None

        # Local fallback model state
        self._local_tokenizer = None
        self._local_model = None

    def _get_api_client(self) -> Optional[OpenAI]:
        """Obtain OpenAI client instance for OpenRouter if API key is present."""
        if self.api_key and self.api_key != "your_openrouter_api_key_here":
            return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=self.api_key)
        return None

    def _encode_via_api(self, client: OpenAI, texts: List[str]) -> torch.Tensor:
        """Generate normalized dense embeddings via OpenRouter embedding endpoint."""
        all_embeddings: List[List[float]] = []
        batch_size = 50

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            # Maintain original order
            batch_embs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embs)

        tensor_embs = torch.tensor(all_embeddings, dtype=torch.float32)
        return F.normalize(tensor_embs, p=2, dim=1)

    def _encode_via_local(self, texts: List[str]) -> torch.Tensor:
        """Fallback local encoding using HuggingFace Transformers."""
        if self._local_model is None:
            from transformers import AutoTokenizer, AutoModel

            local_name = "sentence-transformers/all-MiniLM-L6-v2"
            self._local_tokenizer = AutoTokenizer.from_pretrained(local_name)
            self._local_model = AutoModel.from_pretrained(local_name)
            self._local_model.eval()

        encoded = self._local_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._local_model(**encoded)
            token_embeddings = output[0]
            attention_mask = encoded["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return F.normalize(embeddings, p=2, dim=1)

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text strings into normalized dense vectors."""
        if not texts:
            return torch.empty(0, 1536)

        client = self._get_api_client()
        # If API key is valid and an API model is requested, use OpenRouter API
        if client and self.model_name in ["openai/text-embedding-3-small", "openai/text-embedding-3-large", "baai/bge-large-en-v1.5"]:
            try:
                return self._encode_via_api(client, texts)
            except Exception:
                # In case of network error, fallback to local transformer model
                return self._encode_via_local(texts)
        else:
            return self._encode_via_local(texts)

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Index document chunks by computing and storing their dense vector representations."""
        self.chunks = chunks
        if not chunks:
            self.chunk_embeddings = None
            return

        texts = [chunk["text"] for chunk in chunks]
        self.chunk_embeddings = self._encode_texts(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the top-K most relevant document chunks for a query.

        Args:
            query: The user's question or search query.
            top_k: Number of relevant chunks to return (default 3).

        Returns:
            List of chunk dicts sorted by descending cosine similarity score.
        """
        if not query.strip() or not self.chunks or self.chunk_embeddings is None:
            return []

        # Encode user query
        query_emb = self._encode_texts([query.strip()])

        # Compute cosine similarity between query vector and all chunk vectors
        similarities = F.cosine_similarity(query_emb, self.chunk_embeddings, dim=1)

        # Select top-k matches
        k = min(top_k, len(self.chunks))
        top_scores, top_indices = torch.topk(similarities, k=k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["score"] = round(float(score), 4)
            results.append(chunk_copy)

        return results


def format_context_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved document excerpts into a structured context block for the LLM prompt."""
    if not retrieved_chunks:
        return "No relevant document excerpts were retrieved."

    formatted_parts = []
    for item in retrieved_chunks:
        page_info = f"Page {item.get('page_num', '?')}"
        chunk_info = f"Chunk {item.get('chunk_id', '?')}"
        score_info = f"Relevance Score: {item.get('score', 0.0):.2f}"
        header = f"--- [Document Excerpt | {page_info}, {chunk_info} | {score_info}] ---"
        body = item.get("text", "").strip()
        formatted_parts.append(f"{header}\n{body}")

    return "\n\n".join(formatted_parts)
