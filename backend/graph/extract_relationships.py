"""
Extract relationships between entities using LLM.
Creates edges in the knowledge graph for GraphRAG.
"""
import uuid
import json
from pymongo import MongoClient
from llm.qwenclient import call_llm
from typing import List, Dict, Any

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client.erica


RELATIONSHIP_EXTRACTION_PROMPT = """You are analyzing educational content to identify relationships between concepts.

Given a list of concepts, identify the following relationships:

1. PREREQUISITE: Concept A must be understood before Concept B
   - Example: "Linear Algebra" is prerequisite for "Neural Networks"

2. RELATED: Concepts with near-transfer relationships (similar or complementary)
   - Example: "Gradient Descent" is related to "Backpropagation"

Respond ONLY with valid JSON. No markdown, no code fences, no explanation.

JSON structure:
{
  "prerequisites": [
    {"source": "concept_title", "target": "concept_title", "reason": "why"}
  ],
  "related": [
    {"concept1": "title", "concept2": "title", "reason": "why"}
  ]
}

Concepts to analyze:
<<<CONCEPTS>>>

Remember: Only output the JSON object, nothing else."""


CONCEPT_RESOURCE_PROMPT = """You are linking educational resources to concepts they explain.

Given a concept and a list of resources, identify which resources explain this concept.

Concept: <<<CONCEPT>>>

Resources:
<<<RESOURCES>>>

Respond ONLY with valid JSON. No markdown, no code fences, no explanation.

JSON structure:
{
  "explains": [
    {"resource_description": "description text", "relevance": "high|medium|low"}
  ]
}

Only output the JSON object."""


def _parse_llm_json(raw_response: str):
    """Parse LLM response, handling common formatting issues."""
    if not raw_response:
        return None

    clean = raw_response.strip()

    # Remove code fences
    if clean.startswith("```"):
        parts = clean.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                clean = p
                break

    # Extract JSON if embedded in text
    if not (clean.startswith("{") and clean.endswith("}")):
        if "{" in clean and "}" in clean:
            clean = clean[clean.find("{"):clean.rfind("}") + 1]

    try:
        return json.loads(clean)
    except Exception:
        return None


def extract_concept_relationships(batch_size: int = 10):
    """
    Extract prerequisite and related relationships between concepts.

    Args:
        batch_size: Number of concepts to analyze together
    """
    concepts = list(db.entity_nodes.find({"type": "concept"}))
    total = len(concepts)
    print(f"Found {total} concepts")

    if total < 2:
        print("Need at least 2 concepts to extract relationships")
        return

    # Process concepts in batches
    for i in range(0, total, batch_size):
        batch = concepts[i:i + batch_size]
        batch_end = min(i + batch_size, total)

        print(f"Processing concepts {i+1}-{batch_end}/{total}...")

        # Format concept list for prompt
        concept_list = []
        for c in batch:
            title = c.get("title", "")
            definitions = c.get("definitions", [])
            difficulty = c.get("difficulty", "")
            concept_list.append(f"- {title} ({difficulty}): {definitions[0] if definitions else ''}")

        concepts_text = "\n".join(concept_list)
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.replace("<<<CONCEPTS>>>", concepts_text)

        response = call_llm(prompt)
        data = _parse_llm_json(response)

        if not data:
            print("  Failed to parse LLM response, skipping batch")
            continue

        # Create concept title to ID mapping
        title_to_id = {c.get("title", "").lower(): c["id"] for c in batch}

        # Store prerequisite relationships
        prerequisites = data.get("prerequisites", []) or []
        for prereq in prerequisites:
            if not isinstance(prereq, dict):
                continue

            source_title = prereq.get("source", "").lower()
            target_title = prereq.get("target", "").lower()
            reason = prereq.get("reason", "")

            if source_title in title_to_id and target_title in title_to_id:
                relationship = {
                    "id": str(uuid.uuid4()),
                    "source_id": title_to_id[source_title],
                    "target_id": title_to_id[target_title],
                    "relationship_type": "prerequisite",
                    "metadata": {"reason": reason}
                }
                db.entity_relationships.insert_one(relationship)

        # Store related relationships
        related = data.get("related", []) or []
        for rel in related:
            if not isinstance(rel, dict):
                continue

            concept1_title = rel.get("concept1", "").lower()
            concept2_title = rel.get("concept2", "").lower()
            reason = rel.get("reason", "")

            if concept1_title in title_to_id and concept2_title in title_to_id:
                # Create bidirectional relationships
                relationship1 = {
                    "id": str(uuid.uuid4()),
                    "source_id": title_to_id[concept1_title],
                    "target_id": title_to_id[concept2_title],
                    "relationship_type": "related",
                    "metadata": {"reason": reason}
                }
                db.entity_relationships.insert_one(relationship1)

        print(f"  Stored relationships for batch")


def link_resources_to_concepts():
    """
    Create 'explains' edges from resources to concepts.
    Uses resource descriptions and concept definitions to find matches.
    """
    concepts = list(db.entity_nodes.find({"type": "concept"}))
    resources = list(db.entity_nodes.find({"type": "resource"}))

    print(f"Linking {len(resources)} resources to {len(concepts)} concepts...")

    for concept in concepts:
        concept_title = concept.get("title", "")
        concept_defs = concept.get("definitions", [])
        concept_text = f"{concept_title}: {' '.join(concept_defs)}"

        # Format resources
        resource_list = []
        for r in resources:
            desc = r.get("description", "")
            rtype = r.get("resource_type", "")
            resource_list.append(f"- [{rtype}] {desc}")

        if not resource_list:
            continue

        resources_text = "\n".join(resource_list)
        prompt = CONCEPT_RESOURCE_PROMPT.replace("<<<CONCEPT>>>", concept_text)
        prompt = prompt.replace("<<<RESOURCES>>>", resources_text)

        response = call_llm(prompt)
        data = _parse_llm_json(response)

        if not data:
            continue

        # Match explanations back to resources
        explains = data.get("explains", []) or []
        for item in explains:
            if not isinstance(item, dict):
                continue

            resource_desc = item.get("resource_description", "")
            relevance = item.get("relevance", "medium")

            # Find matching resource
            for resource in resources:
                if resource_desc in resource.get("description", ""):
                    relationship = {
                        "id": str(uuid.uuid4()),
                        "source_id": resource["id"],
                        "target_id": concept["id"],
                        "relationship_type": "explains",
                        "metadata": {"relevance": relevance}
                    }
                    db.entity_relationships.insert_one(relationship)
                    break

    print("Resource-concept linking complete")


def link_examples_to_concepts():
    """
    Create 'example_of' edges from examples to concepts.
    Uses the concepts field in example entities.
    """
    examples = list(db.entity_nodes.find({"type": "example"}))
    concepts = list(db.entity_nodes.find({"type": "concept"}))

    print(f"Linking {len(examples)} examples to concepts...")

    # Create concept title to ID mapping
    title_to_id = {c.get("title", "").lower(): c["id"] for c in concepts}

    for example in examples:
        concept_titles = example.get("concepts", [])

        for title in concept_titles:
            title_lower = title.lower()
            if title_lower in title_to_id:
                relationship = {
                    "id": str(uuid.uuid4()),
                    "source_id": example["id"],
                    "target_id": title_to_id[title_lower],
                    "relationship_type": "example_of",
                    "metadata": {}
                }
                db.entity_relationships.insert_one(relationship)

    print("Example-concept linking complete")


def extract_relationships_for_all_entities():
    """
    Run all relationship extraction pipelines.
    """
    print("=== Starting Relationship Extraction ===\n")

    print("Step 1: Extracting concept-to-concept relationships...")
    extract_concept_relationships()

    print("\nStep 2: Linking resources to concepts...")
    link_resources_to_concepts()

    print("\nStep 3: Linking examples to concepts...")
    link_examples_to_concepts()

    total_relationships = db.entity_relationships.count_documents({})
    print(f"\n=== Complete! Total relationships: {total_relationships} ===")


if __name__ == "__main__":
    extract_relationships_for_all_entities()
