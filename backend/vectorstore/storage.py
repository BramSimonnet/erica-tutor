"""
Vector storage in MongoDB.
Stores embeddings alongside chunk documents for semantic search.
"""
from pymongo import MongoClient
from typing import List, Optional, Dict, Any
import numpy as np

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client.erica

# Collection for storing vector embeddings
embeddings_collection = db.chunk_embeddings


def ensure_indexes():
    """
    Create necessary indexes for efficient retrieval.
    Note: MongoDB doesn't have native vector search in community edition.
    For production, consider MongoDB Atlas with vector search or a dedicated vector DB.
    """
    # Index on chunk_id for fast lookups
    embeddings_collection.create_index("chunk_id", unique=True)
    print("Created indexes for chunk_embeddings collection")


def store_chunk_embedding(
    chunk_id: str,
    embedding: List[float],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Store an embedding vector for a chunk.

    Args:
        chunk_id: ID of the chunk document
        embedding: Vector embedding (list of floats)
        metadata: Optional metadata to store with the embedding

    Returns:
        The chunk_id
    """
    doc = {
        "chunk_id": chunk_id,
        "embedding": embedding,
        "metadata": metadata or {}
    }

    # Upsert: update if exists, insert if not
    embeddings_collection.update_one(
        {"chunk_id": chunk_id},
        {"$set": doc},
        upsert=True
    )

    return chunk_id


def get_chunk_embedding(chunk_id: str) -> Optional[List[float]]:
    """
    Retrieve the embedding vector for a chunk.

    Args:
        chunk_id: ID of the chunk document

    Returns:
        Embedding vector or None if not found
    """
    doc = embeddings_collection.find_one({"chunk_id": chunk_id})
    if doc:
        return doc.get("embedding")
    return None


def batch_store_embeddings(chunks_with_embeddings: List[Dict[str, Any]]) -> int:
    """
    Store multiple embeddings efficiently.

    Args:
        chunks_with_embeddings: List of dicts with keys:
            - chunk_id: str
            - embedding: List[float]
            - metadata: Optional[Dict]

    Returns:
        Number of embeddings stored
    """
    operations = []
    for item in chunks_with_embeddings:
        doc = {
            "chunk_id": item["chunk_id"],
            "embedding": item["embedding"],
            "metadata": item.get("metadata", {})
        }
        operations.append({
            "update_one": {
                "filter": {"chunk_id": item["chunk_id"]},
                "update": {"$set": doc},
                "upsert": True
            }
        })

    if operations:
        result = embeddings_collection.bulk_write([
            op["update_one"] for op in operations
        ])
        return result.upserted_count + result.modified_count

    return 0


def get_all_embeddings() -> List[Dict[str, Any]]:
    """
    Retrieve all stored embeddings.
    Use with caution on large datasets.

    Returns:
        List of documents with chunk_id and embedding
    """
    return list(embeddings_collection.find({}, {"_id": 0}))


def delete_chunk_embedding(chunk_id: str) -> bool:
    """
    Delete an embedding.

    Args:
        chunk_id: ID of the chunk

    Returns:
        True if deleted, False if not found
    """
    result = embeddings_collection.delete_one({"chunk_id": chunk_id})
    return result.deleted_count > 0


def count_embeddings() -> int:
    """Return the total number of stored embeddings."""
    return embeddings_collection.count_documents({})


if __name__ == "__main__":
    ensure_indexes()
    print(f"Total embeddings in database: {count_embeddings()}")
