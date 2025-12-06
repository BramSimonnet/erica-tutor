"""
Retrieval functions for semantic search using vector similarity.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from pymongo import MongoClient

from vectorstore.embeddings import generate_embedding
from vectorstore.storage import get_all_embeddings

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client.erica


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score between -1 and 1 (1 = identical, 0 = orthogonal, -1 = opposite)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 5,
    min_similarity: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Find the most similar chunks to a query embedding.

    Note: This is a brute-force search. For large datasets, consider:
    - MongoDB Atlas Vector Search
    - FAISS
    - Pinecone, Weaviate, or other vector databases

    Args:
        query_embedding: The query vector
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold (0 to 1)

    Returns:
        List of dicts with keys: chunk_id, similarity, chunk_data
    """
    all_embeddings = get_all_embeddings()

    if not all_embeddings:
        return []

    # Calculate similarities
    similarities = []
    for doc in all_embeddings:
        chunk_id = doc["chunk_id"]
        embedding = doc["embedding"]

        similarity = cosine_similarity(query_embedding, embedding)

        if similarity >= min_similarity:
            similarities.append({
                "chunk_id": chunk_id,
                "similarity": float(similarity),
                "metadata": doc.get("metadata", {})
            })

    # Sort by similarity (descending) and take top_k
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = similarities[:top_k]

    # Fetch the actual chunk data
    results = []
    for item in top_results:
        chunk = db.chunk_documents.find_one({"id": item["chunk_id"]})
        if chunk:
            results.append({
                "chunk_id": item["chunk_id"],
                "similarity": item["similarity"],
                "chunk_data": {
                    "text": chunk.get("text", ""),
                    "source_type": chunk.get("source_type", ""),
                    "metadata": chunk.get("metadata", {}),
                    "chunk_index": chunk.get("chunk_index", 0)
                }
            })

    return results


def retrieve_context_for_query(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.3
) -> Dict[str, Any]:
    """
    High-level function to retrieve relevant context for a user query.

    Args:
        query: Natural language query
        top_k: Number of chunks to retrieve
        min_similarity: Minimum similarity threshold

    Returns:
        Dict with:
            - query: original query
            - results: list of relevant chunks with similarity scores
            - context_text: concatenated text from all retrieved chunks
    """
    # Generate embedding for the query
    query_embedding = generate_embedding(query)

    # Search for similar chunks
    results = search_similar_chunks(
        query_embedding,
        top_k=top_k,
        min_similarity=min_similarity
    )

    # Concatenate text for easy use with LLM
    context_text = "\n\n---\n\n".join([
        f"[Source: {r['chunk_data']['source_type']}, Similarity: {r['similarity']:.3f}]\n{r['chunk_data']['text']}"
        for r in results
    ])

    return {
        "query": query,
        "results": results,
        "context_text": context_text,
        "num_results": len(results)
    }


def search_by_text(
    text: str,
    top_k: int = 5,
    min_similarity: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Convenience function to search using text instead of pre-computed embedding.

    Args:
        text: Query text
        top_k: Number of results
        min_similarity: Minimum similarity threshold

    Returns:
        List of similar chunks
    """
    embedding = generate_embedding(text)
    return search_similar_chunks(embedding, top_k, min_similarity)


if __name__ == "__main__":
    # Example usage
    query = "What is machine learning?"
    context = retrieve_context_for_query(query, top_k=3)

    print(f"Query: {context['query']}")
    print(f"Found {context['num_results']} results")
    print("\nContext text:")
    print(context['context_text'])
