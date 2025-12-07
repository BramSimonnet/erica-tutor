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
    # Course pages to ingest
    urls = [
        # Assignment page
        "https://pantelis.github.io/aiml-common/projects/nlp/ai-tutor/",

        # Main course pages
        "https://pantelis.github.io/courses/ai/",
        "https://pantelis.github.io/courses/cv/",

        # Book sections - LLM/Transformers/Attention
        "https://pantelis.github.io/book/llm/",

        # Deep Learning foundations
        "https://pantelis.github.io/book/foundations/",
        "https://pantelis.github.io/book/dnn/",

        # Computer Vision (for CLIP)
        "https://pantelis.github.io/book/2d-perception/",

        # Math foundations (for variational bounds, Jensen's inequality)
        "https://pantelis.github.io/aiml-common/lectures/ml-math/",
        "https://pantelis.github.io/aiml-common/lectures/ml-math/probability/",
    ]

    print(f"Ingesting {len(urls)} web pages...")
    for i, url in enumerate(urls, 1):
        try:
            print(f"\n[{i}/{len(urls)}] Processing: {url}")
            ingest_webpage(url)
        except Exception as e:
            print(f"[ERROR] Failed to ingest {url}: {e}")
            continue

    print(f"\n✓ Completed! Check MongoDB for results.")
