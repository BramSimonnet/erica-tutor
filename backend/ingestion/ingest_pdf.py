import uuid
from pymongo import MongoClient
from pypdf import PdfReader

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client["erica"]
raw_docs = db["raw_documents"]

def ingest_pdf(path: str, metadata: dict = None):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    document = {
        "id": str(uuid.uuid4()),
        "type": "pdf",
        "file_path": path,
        "raw_text": text,
        "metadata": metadata or {}
    }

    raw_docs.insert_one(document)
    print(f"[OK] Ingested PDF: {path}")


if __name__ == "__main__":
    ingest_pdf("data/pdfs/The Learning Problem – Engineering AI Agents.pdf",
               metadata={"source": "course_lecture"})