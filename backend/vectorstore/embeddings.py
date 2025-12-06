"""
Embeddings generation using sentence-transformers.
Uses a lightweight model for encoding text into dense vectors.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Use a lightweight but effective model
# all-MiniLM-L6-v2: 384 dimensions, fast, good for semantic search
MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy load the model
_model = None


def _get_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embedding(text: str) -> List[float]:
    """
    Generate a dense vector embedding for the given text.

    Args:
        text: Input text to embed

    Returns:
        List of floats representing the embedding vector (384 dimensions)
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def batch_generate_embeddings(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for multiple texts efficiently.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process at once

    Returns:
        List of embedding vectors
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """Return the dimension of the embedding vectors."""
    return 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors
