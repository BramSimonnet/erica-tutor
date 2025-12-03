import uuid
import json
from pymongo import MongoClient
from llm.qwenclient import call_llm 

ENTITY_EXTRACTION_PROMPT = """
You are an AI that extracts structured knowledge from text.

Given the following chunk of text, extract three kinds of information:

1. CONCEPTS:
- title
- definitions (list of 1-3 short definitions)
- difficulty (one of: "easy", "medium", "hard")
- aliases (list, can be empty)

2. RESOURCES:
- resource_type (one of: "pdf", "slide", "web", "video")
- span (page number, section name, or timestamp)
- description (short description of what this resource covers)

3. EXAMPLES:
- text (short example from the chunk)
- concepts (list of concept titles mentioned in the example)

You MUST follow these rules:

- Respond ONLY with a single valid JSON object.
- Do NOT include any explanation, comments, markdown, or code fences.
- Do NOT wrap the JSON in ```json or ``` or anything similar.
- The JSON must have EXACTLY the following top-level keys:
  "concepts", "resources", "examples".
- Each of those keys must map to a list (possibly empty).

The JSON structure MUST be:

{
  "concepts": [
    {
      "title": "string",
      "definitions": ["string", "..."],
      "difficulty": "easy" | "medium" | "hard",
      "aliases": ["string", "..."]
    }
  ],
  "resources": [
    {
      "resource_type": "pdf" | "slide" | "web" | "video",
      "span": "string",
      "description": "string"
    }
  ],
  "examples": [
    {
      "text": "string",
      "concepts": ["string", "..."]
    }
  ]
}

Chunk text:
<<<CHUNK>>>
"""

client = MongoClient("mongodb://mongo:27017")
db = client.erica


def _parse_llm_json(raw_response: str):
    """
    Try to turn the LLM raw output into a Python dict.
    Cleans up common issues like code fences and extra text.
    Returns dict on success, or None on failure.
    """
    if not raw_response:
        return None

    clean = raw_response.strip()

    if clean.startswith("```"):
        parts = clean.split("```")
        candidate = None
        for p in parts:
            p = p.strip()
            if p.startswith("{") and p.endswith("}"):
                candidate = p
                break
        if candidate:
            clean = candidate

    if not (clean.startswith("{") and clean.endswith("}")):
        if "{" in clean and "}" in clean:
            clean = clean[clean.find("{") : clean.rfind("}") + 1]

    try:
        return json.loads(clean)
    except Exception:
        return None


def extract_entities_for_all_chunks():
    chunks = list(db.chunk_documents.find({}))
    print(f"Found {len(chunks)} chunks.")

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        if not chunk_text.strip():
            print(f"Empty text for chunk {chunk.get('id')}, skipping.")
            continue

        prompt = ENTITY_EXTRACTION_PROMPT.replace("<<<CHUNK>>>", chunk_text)

        response = call_llm(prompt)

        data = _parse_llm_json(response)
        if data is None:
            print("JSON parsing error. Skipping chunk.")
            # print("RAW LLM RESPONSE:", response[:400])
            continue

        concepts = data.get("concepts", []) or []
        resources = data.get("resources", []) or []
        examples = data.get("examples", []) or []

        if not (concepts or resources or examples):
            print(f"No entities found for chunk {chunk.get('id')}, skipping insert.")
            continue

        chunk_id = chunk.get("id")

        for concept in concepts:
            if not isinstance(concept, dict):
                continue
            concept_doc = {
                **concept,
                "id": str(uuid.uuid4()),
                "type": "concept",
                "source_chunk": chunk_id,
            }
            db.entity_nodes.insert_one(concept_doc)

        for resource in resources:
            if not isinstance(resource, dict):
                continue
            resource_doc = {
                **resource,
                "id": str(uuid.uuid4()),
                "type": "resource",
                "source_chunk": chunk_id,
            }
            db.entity_nodes.insert_one(resource_doc)

        for example in examples:
            if not isinstance(example, dict):
                continue
            example_doc = {
                **example,
                "id": str(uuid.uuid4()),
                "type": "example",
                "source_chunk": chunk_id,
            }
            db.entity_nodes.insert_one(example_doc)

        print(f"Saved entities for chunk {chunk_id}.")


if __name__ == "__main__":
    extract_entities_for_all_chunks()
