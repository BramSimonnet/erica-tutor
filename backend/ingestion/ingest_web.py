import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import uuid

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client["erica"]
raw_docs = db["raw_documents"]

def ingest_webpage(url: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator="\n")

    document = {
        "id": str(uuid.uuid4()),
        "type": "web",
        "url": url,
        "raw_text": text
    }

    raw_docs.insert_one(document)
    print(f"[OK] Ingested webpage: {url}")


if __name__ == "__main__":
    # EXAMPLE: replace with your actual course site
    ingest_webpage("https://pantelis.github.io/aiml-common/projects/nlp/ai-tutor/")
