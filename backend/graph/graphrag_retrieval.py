"""
GraphRAG retrieval: Use knowledge graph traversal for context retrieval.
Combines graph structure with vector similarity for enhanced retrieval.
"""
import networkx as nx
from typing import List, Dict, Any, Set, Tuple
from graph.build_graph import (
    load_graph,
    find_prerequisites,
    find_resources_for_concept,
    find_examples_for_concept,
    get_prerequisite_chain,
    find_related_concepts
)
from vectorstore.embeddings import generate_embedding
from vectorstore.retrieval import cosine_similarity
from vectorstore.storage import get_chunk_embedding
from pymongo import MongoClient

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client.erica


def find_relevant_concepts(
    query: str,
    G: nx.MultiDiGraph,
    top_k: int = 5,
    min_similarity: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Find concepts relevant to a query using hybrid approach:
    1. Vector similarity on concept definitions
    2. Graph structure for ranking

    Args:
        query: User query
        G: Knowledge graph
        top_k: Number of concepts to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of concept nodes with similarity scores
    """
    query_embedding = generate_embedding(query)

    # Get all concept nodes
    concepts = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "concept"]

    if not concepts:
        return []

    # Calculate similarity for each concept
    concept_scores = []

    for node_id, node_data in concepts:
        # Create concept text from title and definitions
        title = node_data.get("title", "")
        definitions = node_data.get("definitions", [])
        concept_text = f"{title}. {' '.join(definitions)}"

        # Get embedding for concept
        concept_embedding = generate_embedding(concept_text)

        # Calculate similarity
        similarity = cosine_similarity(query_embedding, concept_embedding)

        if similarity >= min_similarity:
            # Boost score based on graph centrality
            in_degree = G.in_degree(node_id)
            out_degree = G.out_degree(node_id)
            centrality_boost = (in_degree + out_degree) * 0.01  # Small boost

            concept_scores.append({
                "node_id": node_id,
                "title": title,
                "definitions": definitions,
                "difficulty": node_data.get("difficulty", "medium"),
                "similarity": similarity,
                "boosted_score": similarity + centrality_boost,
                "degree": in_degree + out_degree
            })

    # Sort by boosted score
    concept_scores.sort(key=lambda x: x["boosted_score"], reverse=True)

    return concept_scores[:top_k]


def retrieve_subgraph(
    concept_ids: List[str],
    G: nx.MultiDiGraph,
    include_prerequisites: bool = True,
    include_resources: bool = True,
    include_examples: bool = True,
    max_depth: int = 2
) -> Dict[str, Any]:
    """
    Retrieve a subgraph centered on given concepts.
    This is the core of GraphRAG - we get related knowledge via graph traversal.

    Args:
        concept_ids: List of concept node IDs
        G: Knowledge graph
        include_prerequisites: Include prerequisite concepts
        include_resources: Include resource nodes
        include_examples: Include example nodes
        max_depth: Maximum depth for graph traversal

    Returns:
        Dict with nodes, edges, and organized context
    """
    subgraph_nodes = set(concept_ids)
    subgraph_edges = []

    # 1. Add prerequisite chains (for scaffolding)
    prerequisites_by_concept = {}
    if include_prerequisites:
        for concept_id in concept_ids:
            prereqs = find_prerequisites(G, concept_id)
            prerequisites_by_concept[concept_id] = prereqs
            subgraph_nodes.update(prereqs)

            # Add prerequisite edges
            for prereq in prereqs:
                subgraph_edges.append({
                    "source": prereq,
                    "target": concept_id,
                    "type": "prerequisite"
                })

    # 2. Add related concepts (near-transfer)
    related_concepts = set()
    for concept_id in concept_ids:
        related = find_related_concepts(G, concept_id)
        related_concepts.update(related)
        subgraph_nodes.update(related)

        for rel in related:
            subgraph_edges.append({
                "source": concept_id,
                "target": rel,
                "type": "related"
            })

    # 3. Add resources that explain the concepts
    resources_by_concept = {}
    if include_resources:
        for concept_id in list(concept_ids) + list(related_concepts):
            resources = find_resources_for_concept(G, concept_id)
            if resources:
                resources_by_concept[concept_id] = resources
                for resource in resources:
                    subgraph_nodes.add(resource["id"])
                    subgraph_edges.append({
                        "source": resource["id"],
                        "target": concept_id,
                        "type": "explains"
                    })

    # 4. Add examples
    examples_by_concept = {}
    if include_examples:
        for concept_id in list(concept_ids) + list(related_concepts):
            examples = find_examples_for_concept(G, concept_id)
            if examples:
                examples_by_concept[concept_id] = examples
                for example in examples:
                    subgraph_nodes.add(example["id"])
                    subgraph_edges.append({
                        "source": example["id"],
                        "target": concept_id,
                        "type": "example_of"
                    })

    # Collect all node data
    nodes = []
    for node_id in subgraph_nodes:
        if G.has_node(node_id):
            node_data = G.nodes[node_id].copy()
            node_data["id"] = node_id
            nodes.append(node_data)

    return {
        "nodes": nodes,
        "edges": subgraph_edges,
        "main_concepts": concept_ids,
        "prerequisites": prerequisites_by_concept,
        "resources": resources_by_concept,
        "examples": examples_by_concept,
        "related_concepts": list(related_concepts)
    }


def get_scaffolded_context(
    concept_ids: List[str],
    G: nx.MultiDiGraph
) -> List[Dict[str, Any]]:
    """
    Build scaffolded learning path from simple to complex concepts.
    Uses prerequisite chains to order concepts.

    Args:
        concept_ids: Target concept IDs
        G: Knowledge graph

    Returns:
        Ordered list of concepts (simple → complex) with metadata
    """
    # Get all prerequisite chains for each concept
    all_concepts_in_chains = set()
    chains_by_concept = {}

    for concept_id in concept_ids:
        chains = get_prerequisite_chain(G, concept_id)
        chains_by_concept[concept_id] = chains

        # Add all concepts from chains
        for chain in chains:
            all_concepts_in_chains.update(chain)

    # Build difficulty ordering: easy → medium → hard
    concepts_by_difficulty = {"easy": [], "medium": [], "hard": []}

    for concept_id in all_concepts_in_chains:
        if G.has_node(concept_id):
            node_data = G.nodes[concept_id]
            difficulty = node_data.get("difficulty", "medium")
            concepts_by_difficulty[difficulty].append({
                "id": concept_id,
                "title": node_data.get("title", ""),
                "definitions": node_data.get("definitions", []),
                "difficulty": difficulty
            })

    # Return ordered: easy first, then medium, then hard
    scaffolded = (
        concepts_by_difficulty["easy"] +
        concepts_by_difficulty["medium"] +
        concepts_by_difficulty["hard"]
    )

    return scaffolded


def graphrag_retrieve(
    query: str,
    top_k_concepts: int = 3,
    min_similarity: float = 0.3
) -> Dict[str, Any]:
    """
    Main GraphRAG retrieval function.
    Combines vector similarity with graph traversal.

    Args:
        query: User query
        top_k_concepts: Number of concepts to retrieve
        min_similarity: Minimum similarity threshold

    Returns:
        Dict with:
            - query: original query
            - relevant_concepts: concepts matching query
            - subgraph: relevant subgraph data
            - scaffolded_path: ordered learning path
            - context_summary: formatted context for LLM
    """
    # Load graph
    G = load_graph()
    if G is None:
        return {
            "query": query,
            "error": "Knowledge graph not found. Run build_graph.py first.",
            "relevant_concepts": [],
            "subgraph": {},
            "scaffolded_path": []
        }

    # Step 1: Find relevant concepts using hybrid retrieval
    relevant_concepts = find_relevant_concepts(
        query, G,
        top_k=top_k_concepts,
        min_similarity=min_similarity
    )

    if not relevant_concepts:
        return {
            "query": query,
            "relevant_concepts": [],
            "subgraph": {},
            "scaffolded_path": [],
            "context_summary": "No relevant concepts found."
        }

    concept_ids = [c["node_id"] for c in relevant_concepts]

    # Step 2: Retrieve subgraph
    subgraph = retrieve_subgraph(
        concept_ids, G,
        include_prerequisites=True,
        include_resources=True,
        include_examples=True
    )

    # Step 3: Build scaffolded learning path
    scaffolded_path = get_scaffolded_context(concept_ids, G)

    # Step 4: Format context summary for LLM
    context_summary = _format_context_for_llm(
        relevant_concepts,
        scaffolded_path,
        subgraph
    )

    return {
        "query": query,
        "relevant_concepts": relevant_concepts,
        "subgraph": subgraph,
        "scaffolded_path": scaffolded_path,
        "context_summary": context_summary
    }


def _format_context_for_llm(
    relevant_concepts: List[Dict[str, Any]],
    scaffolded_path: List[Dict[str, Any]],
    subgraph: Dict[str, Any]
) -> str:
    """Format retrieved context for LLM consumption."""
    lines = []

    lines.append("=== SCAFFOLDED LEARNING PATH (Simple → Complex) ===\n")
    for i, concept in enumerate(scaffolded_path, 1):
        lines.append(f"{i}. {concept['title']} [{concept['difficulty']}]")
        if concept['definitions']:
            lines.append(f"   Definition: {concept['definitions'][0]}")
        lines.append("")

    lines.append("\n=== MAIN CONCEPTS ===\n")
    for concept in relevant_concepts:
        lines.append(f"• {concept['title']} (similarity: {concept['similarity']:.3f})")
        for defn in concept['definitions']:
            lines.append(f"  - {defn}")
        lines.append("")

    # Add resources
    if subgraph.get("resources"):
        lines.append("\n=== RESOURCES ===\n")
        for concept_id, resources in subgraph["resources"].items():
            for resource in resources:
                lines.append(f"• [{resource['resource_type']}] {resource['description']}")
                lines.append(f"  Span: {resource['span']}")
                lines.append("")

    # Add examples
    if subgraph.get("examples"):
        lines.append("\n=== WORKED EXAMPLES ===\n")
        for concept_id, examples in subgraph["examples"].items():
            for example in examples:
                lines.append(f"• {example['text'][:200]}...")
                lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test retrieval
    query = "What is attention in transformers?"
    result = graphrag_retrieve(query, top_k_concepts=3)

    print(f"Query: {result['query']}")
    print(f"\nFound {len(result['relevant_concepts'])} relevant concepts")
    print(f"Subgraph: {len(result['subgraph'].get('nodes', []))} nodes, {len(result['subgraph'].get('edges', []))} edges")
    print(f"Scaffolded path: {len(result['scaffolded_path'])} concepts")
    print(f"\nContext Summary:\n{result['context_summary']}")
