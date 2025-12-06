import uuid
import json
from pymongo import MongoClient
from llm.qwenclient import call_llm

client = MongoClient("mongodb://mongo:27017")
db = client.erica

def inspect_entities_per_chunk():
    chunks = list(db.chunk_documents.find({}))
    print(f"Found {len(chunks)} chunks.\n")
    for chunk in chunks:
        chunk_id = chunk.get("id")
        text_preview = chunk.get("text", "")[:120].replace("\n", " ")
        concepts = list(db.entity_nodes.find({"type": "concept", "source_chunk": chunk_id}))
        resources = list(db.entity_nodes.find({"type": "resource", "source_chunk": chunk_id}))
        examples = list(db.entity_nodes.find({"type": "example", "source_chunk": chunk_id}))
        print("=" * 80)
        print(f"CHUNK {chunk_id}")
        print(f"Text preview: {text_preview!r}")
        print(f"  Concepts:  {len(concepts)}")
        print(f"  Resources: {len(resources)}")
        print(f"  Examples:  {len(examples)}")
        if concepts:
            print("    Sample concept:", concepts[0].get("title"))
        if resources:
            print("    Sample resource:", resources[0].get("description", resources[0].get("span", "")))
        if examples:
            print("    Sample example:", examples[0].get("text"))
    print("\nDone inspecting entities per chunk.")

RELATION_PROMPT = """
You extract relationships between entities in text.

Relationships allowed:
- prereq_of (concept → concept)
- explains (resource → concept)
- exemplifies (example → concept)
- near_transfer (concept ↔ concept)

Return only JSON:

{
  "relations": [
    {
      "type": "prereq_of",
      "source": "<entity_id>",
      "target": "<entity_id>"
    }
  ]
}

Chunk:
<<<TEXT>>>

Entities:
<<<ENTITIES>>>
"""

def save_relations(chunk_id, relations):
    for rel in relations:
        edge_doc = {
            "id": str(uuid.uuid4()),
            "type": rel["type"],
            "source": rel["source"],
            "target": rel["target"],
            "chunk_id": chunk_id
        }
        db.entity_edges.insert_one(edge_doc)
    print(f"Saved {len(relations)} relations for chunk {chunk_id}")

def extract_relations_for_one_chunk(chunk_id):
    chunk = db.chunk_documents.find_one({"id": chunk_id})
    if not chunk:
        print("Chunk not found:", chunk_id)
        return

    entities = list(db.entity_nodes.find({"source_chunk": chunk_id}))

    entities_for_prompt = []
    for e in entities:
        entities_for_prompt.append({
            "id": e["id"],
            "type": e["type"],
            "title": e.get("title"),
            "text": e.get("text"),
            "description": e.get("description")
        })

    prompt = RELATION_PROMPT.replace("<<<TEXT>>>", chunk["text"]).replace(
        "<<<ENTITIES>>>", json.dumps(entities_for_prompt, indent=2)
    )

    response = call_llm(prompt)
    clean = response.strip()

    if clean.startswith("```"):
        clean = clean.replace("```", "")
    clean = clean.lstrip()
    if clean.startswith("json"):
        clean = clean[4:].lstrip()
    clean = clean.strip()

    try:
        data = json.loads(clean)
    except Exception:
        print("JSON parse error")
        print("--- RAW RESPONSE START ---")
        print(response)
        print("--- CLEANED VERSION START ---")
        print(clean)
        print("--- CLEANED VERSION END ---")
        return

    print(json.dumps(data, indent=2))

    if "relations" in data:
        save_relations(chunk_id, data["relations"])
    else:
        print("No relations found.")

def extract_relations_for_all_chunks():
    chunks = list(db.chunk_documents.find({}))
    print(f"Found {len(chunks)} chunks.")
    db.entity_edges.delete_many({})
    print("Cleared old edges.")
    for chunk in chunks:
        chunk_id = chunk["id"]
        entity_count = db.entity_nodes.count_documents({"source_chunk": chunk_id})
        if entity_count == 0:
            print(f"Skipping chunk {chunk_id} — no entities.")
            continue
        print(f"Extracting relations for chunk {chunk_id}...")
        try:
            extract_relations_for_one_chunk(chunk_id)
        except Exception as e:
            print(f"Error in chunk {chunk_id}: {e}")
            continue
    total = db.entity_edges.count_documents({})
    print(f"Done. Total relations: {total}")

if __name__ == "__main__":
    extract_relations_for_all_chunks()
