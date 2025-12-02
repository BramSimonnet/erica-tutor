import requests

import uuid
from pymongo import MongoClient
import yt_dlp

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client["erica"]
raw_docs = db["raw_documents"]

def get_youtube_transcript(url: str):
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
        "outtmpl": "%(id)s"
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    transcript = info.get("automatic_captions", {}).get("en", [{}])[0].get("url")
    return transcript


def ingest_youtube(url: str):
    transcript_url = get_youtube_transcript(url)
    if not transcript_url:
        print("[ERR] No transcript found.")
        return

    text = requests.get(transcript_url).text

    document = {
        "id": str(uuid.uuid4()),
        "type": "youtube",
        "url": url,
        "raw_text": text
    }

    raw_docs.insert_one(document)
    print(f"[OK] Ingested YouTube transcript: {url}")


if __name__ == "__main__":
    ingest_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
