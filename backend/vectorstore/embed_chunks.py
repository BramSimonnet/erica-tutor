"""
Script to generate and store embeddings for all chunks in the database.
Run this after ingestion and chunking to populate the vector store.
"""
from pymongo import MongoClient
from typing import List, Dict, Any
from vectorstore.embeddings import batch_generate_embeddings
from vectorstore.storage import batch_store_embeddings, ensure_indexes, count_embeddings

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client.erica


def embed_all_chunks(batch_size: int = 32):
    """
    Generate embeddings for all chunks in the database.

    Args:
        batch_size: Number of chunks to process at once
    """
    # Ensure indexes exist
    ensure_indexes()

    # Get all chunks
    chunks = list(db.chunk_documents.find({}))
    total_chunks = len(chunks)

    if total_chunks == 0:
        print("No chunks found in database. Run ingestion and chunking first.")
        return

    print(f"Found {total_chunks} chunks to embed")

    # Process in batches
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_end = min(i + batch_size, total_chunks)

        print(f"Processing chunks {i+1}-{batch_end}/{total_chunks}...")

        # Extract texts and chunk IDs
        texts = [chunk["text"] for chunk in batch]
        chunk_ids = [chunk["id"] for chunk in batch]

        # Generate embeddings
        embeddings = batch_generate_embeddings(texts, batch_size=len(texts))

        # Prepare data for storage
        chunks_with_embeddings = []
        for chunk, embedding in zip(batch, embeddings):
            chunks_with_embeddings.append({
                "chunk_id": chunk["id"],
                "embedding": embedding,
                "metadata": {
                    "source_type": chunk.get("source_type", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "raw_document_id": chunk.get("raw_document_id", "")
                }
            })

        # Store embeddings
        stored = batch_store_embeddings(chunks_with_embeddings)
        print(f"  Stored {stored} embeddings")

    final_count = count_embeddings()
    print(f"\n✓ Complete! Total embeddings in database: {final_count}")


def embed_new_chunks_only(batch_size: int = 32):
    """
    Generate embeddings only for chunks that don't have them yet.

    Args:
        batch_size: Number of chunks to process at once
    """
    ensure_indexes()

    # Get all chunk IDs
    all_chunks = list(db.chunk_documents.find({}, {"id": 1, "text": 1, "source_type": 1, "chunk_index": 1, "raw_document_id": 1}))

    # Get chunk IDs that already have embeddings
    existing_embeddings = db.chunk_embeddings.find({}, {"chunk_id": 1})
    existing_ids = {doc["chunk_id"] for doc in existing_embeddings}

    # Filter to only new chunks
    new_chunks = [chunk for chunk in all_chunks if chunk["id"] not in existing_ids]

    if not new_chunks:
        print("All chunks already have embeddings!")
        return

    print(f"Found {len(new_chunks)} new chunks (out of {len(all_chunks)} total)")

    # Process in batches
    total_new = len(new_chunks)
    for i in range(0, total_new, batch_size):
        batch = new_chunks[i:i + batch_size]
        batch_end = min(i + batch_size, total_new)

        print(f"Embedding chunks {i+1}-{batch_end}/{total_new}...")

        texts = [chunk["text"] for chunk in batch]
        embeddings = batch_generate_embeddings(texts, batch_size=len(texts))

        chunks_with_embeddings = []
        for chunk, embedding in zip(batch, embeddings):
            chunks_with_embeddings.append({
                "chunk_id": chunk["id"],
                "embedding": embedding,
                "metadata": {
                    "source_type": chunk.get("source_type", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "raw_document_id": chunk.get("raw_document_id", "")
                }
            })

        stored = batch_store_embeddings(chunks_with_embeddings)
        print(f"  Stored {stored} embeddings")

    final_count = count_embeddings()
    print(f"\n✓ Complete! Total embeddings: {final_count}")


def verify_embeddings():
    """
    Verify that all chunks have embeddings.
    """
    total_chunks = db.chunk_documents.count_documents({})
    total_embeddings = count_embeddings()

    print(f"Chunks in database: {total_chunks}")
    print(f"Embeddings in database: {total_embeddings}")

    if total_chunks == total_embeddings:
        print("✓ All chunks have embeddings!")
    else:
        print(f"⚠ Missing embeddings for {total_chunks - total_embeddings} chunks")
        print("  Run embed_new_chunks_only() to add missing embeddings")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_embeddings()
    elif len(sys.argv) > 1 and sys.argv[1] == "--new-only":
        embed_new_chunks_only()
    else:
        # Default: embed all chunks (will update existing ones)
        embed_all_chunks()
