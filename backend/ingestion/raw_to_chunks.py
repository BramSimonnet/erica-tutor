import uuid
from pymongo import MongoClient
from ingestion.chunk import split_into_chunks

client = MongoClient("mongodb://mongo:27017")
db = client.erica

raw_docs = db.raw_documents
chunk_docs = db.chunk_documents

def process_all():
    for raw in raw_docs.find():
        text = raw.get("raw_text", "")
        chunks = split_into_chunks(text)

        for idx, chunk in enumerate(chunks):
            chunk_docs.insert_one({
                "id": str(uuid.uuid4()),
                "raw_document_id": raw["id"],
                "chunk_index": idx,
                "text": chunk,
                "source_type": raw["type"],
                "metadata": raw.get("metadata", {})
            })

        print(f"[OK] {raw['type']} {raw['id']} → {len(chunks)} chunks")

if __name__ == "__main__":
    process_all()
