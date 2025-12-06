"""
Vector storage module for semantic search and retrieval.
"""
from vectorstore.embeddings import generate_embedding, batch_generate_embeddings
from vectorstore.storage import store_chunk_embedding, get_chunk_embedding
from vectorstore.retrieval import search_similar_chunks, retrieve_context_for_query

__all__ = [
    "generate_embedding",
    "batch_generate_embeddings",
    "store_chunk_embedding",
    "get_chunk_embedding",
    "search_similar_chunks",
    "retrieve_context_for_query",
]
