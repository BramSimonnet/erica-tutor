"""
Build and manage the knowledge graph using NetworkX.
The graph supports GraphRAG by enabling traversal-based retrieval.
"""
import networkx as nx
from pymongo import MongoClient
from typing import Dict, List, Any, Optional, Set
import pickle
import json

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client.erica


def build_knowledge_graph() -> nx.MultiDiGraph:
    """
    Build a directed multigraph from entity nodes and relationship edges.

    Node types:
    - concept: Concepts with difficulty levels, definitions, aliases
    - resource: Resources (PDF, slide, web, video) with spans
    - example: Worked examples linked to concepts

    Edge types:
    - prerequisite: concept A is prerequisite for concept B
    - explains: resource/example explains concept
    - related: near-transfer relationship between concepts
    - example_of: example demonstrates concept

    Returns:
        NetworkX MultiDiGraph with entity nodes and relationships
    """
    G = nx.MultiDiGraph()

    # Add all entity nodes
    entities = list(db.entity_nodes.find({}))
    print(f"Adding {len(entities)} entity nodes to graph...")

    for entity in entities:
        node_id = entity["id"]
        node_type = entity["type"]

        # Base attributes for all nodes
        node_attrs = {
            "type": node_type,
            "source_chunk": entity.get("source_chunk", ""),
        }

        # Type-specific attributes
        if node_type == "concept":
            node_attrs.update({
                "title": entity.get("title", ""),
                "definitions": entity.get("definitions", []),
                "difficulty": entity.get("difficulty", "medium"),
                "aliases": entity.get("aliases", []),
            })
        elif node_type == "resource":
            node_attrs.update({
                "resource_type": entity.get("resource_type", ""),
                "span": entity.get("span", ""),
                "description": entity.get("description", ""),
            })
        elif node_type == "example":
            node_attrs.update({
                "text": entity.get("text", ""),
                "concepts": entity.get("concepts", []),
            })

        G.add_node(node_id, **node_attrs)

    # Add relationship edges
    relationships = list(db.entity_relationships.find({}))
    print(f"Adding {len(relationships)} relationship edges to graph...")

    for rel in relationships:
        source = rel.get("source_id")
        target = rel.get("target_id")
        rel_type = rel.get("relationship_type")

        if source and target and rel_type:
            # Check that both nodes exist
            if G.has_node(source) and G.has_node(target):
                G.add_edge(
                    source,
                    target,
                    relationship_type=rel_type,
                    metadata=rel.get("metadata", {})
                )

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def save_graph(G: nx.MultiDiGraph, filepath: str = "/app/data/knowledge_graph.pkl"):
    """
    Save the knowledge graph to disk.

    Args:
        G: NetworkX graph
        filepath: Path to save the graph
    """
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {filepath}")


def load_graph(filepath: str = "/app/data/knowledge_graph.pkl") -> Optional[nx.MultiDiGraph]:
    """
    Load the knowledge graph from disk.

    Args:
        filepath: Path to load the graph from

    Returns:
        NetworkX graph or None if file doesn't exist
    """
    try:
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except FileNotFoundError:
        print(f"Graph file not found at {filepath}")
        return None


def get_concept_nodes(G: nx.MultiDiGraph) -> List[str]:
    """Get all concept node IDs."""
    return [n for n, d in G.nodes(data=True) if d.get("type") == "concept"]


def get_resource_nodes(G: nx.MultiDiGraph) -> List[str]:
    """Get all resource node IDs."""
    return [n for n, d in G.nodes(data=True) if d.get("type") == "resource"]


def get_example_nodes(G: nx.MultiDiGraph) -> List[str]:
    """Get all example node IDs."""
    return [n for n, d in G.nodes(data=True) if d.get("type") == "example"]


def find_prerequisites(G: nx.MultiDiGraph, concept_id: str) -> List[str]:
    """
    Find all prerequisite concepts for a given concept.

    Args:
        G: Knowledge graph
        concept_id: Target concept node ID

    Returns:
        List of prerequisite concept IDs
    """
    prerequisites = []
    for source, target, data in G.in_edges(concept_id, data=True):
        if data.get("relationship_type") == "prerequisite":
            prerequisites.append(source)
    return prerequisites


def find_dependent_concepts(G: nx.MultiDiGraph, concept_id: str) -> List[str]:
    """
    Find all concepts that depend on this concept (reverse prerequisites).

    Args:
        G: Knowledge graph
        concept_id: Source concept node ID

    Returns:
        List of dependent concept IDs
    """
    dependents = []
    for source, target, data in G.out_edges(concept_id, data=True):
        if data.get("relationship_type") == "prerequisite":
            dependents.append(target)
    return dependents


def find_related_concepts(G: nx.MultiDiGraph, concept_id: str) -> List[str]:
    """
    Find concepts with near-transfer relationships.

    Args:
        G: Knowledge graph
        concept_id: Concept node ID

    Returns:
        List of related concept IDs
    """
    related = []
    # Check both incoming and outgoing edges
    for source, target, data in G.edges(concept_id, data=True):
        if data.get("relationship_type") == "related":
            related.append(target if source == concept_id else source)
    return related


def find_resources_for_concept(G: nx.MultiDiGraph, concept_id: str) -> List[Dict[str, Any]]:
    """
    Find all resources that explain a concept.

    Args:
        G: Knowledge graph
        concept_id: Concept node ID

    Returns:
        List of resource node data with IDs
    """
    resources = []
    for source, target, data in G.in_edges(concept_id, data=True):
        if data.get("relationship_type") == "explains":
            node_data = G.nodes[source]
            if node_data.get("type") == "resource":
                resources.append({
                    "id": source,
                    **node_data
                })
    return resources


def find_examples_for_concept(G: nx.MultiDiGraph, concept_id: str) -> List[Dict[str, Any]]:
    """
    Find all examples that demonstrate a concept.

    Args:
        G: Knowledge graph
        concept_id: Concept node ID

    Returns:
        List of example node data with IDs
    """
    examples = []
    for source, target, data in G.in_edges(concept_id, data=True):
        if data.get("relationship_type") == "example_of":
            node_data = G.nodes[source]
            if node_data.get("type") == "example":
                examples.append({
                    "id": source,
                    **node_data
                })
    return examples


def get_prerequisite_chain(G: nx.MultiDiGraph, concept_id: str) -> List[List[str]]:
    """
    Get all prerequisite chains leading to a concept (for scaffolding).
    Returns paths from foundational concepts to target concept.

    Args:
        G: Knowledge graph
        concept_id: Target concept node ID

    Returns:
        List of paths, each path is a list of concept IDs (simple → complex)
    """
    # Find all concepts with no prerequisites (foundational)
    all_concepts = get_concept_nodes(G)
    foundational = [c for c in all_concepts if len(find_prerequisites(G, c)) == 0]

    # Find paths from each foundational concept to target
    chains = []
    for foundation in foundational:
        if foundation == concept_id:
            chains.append([concept_id])
        else:
            try:
                # Find all simple paths (prerequisite edges only)
                paths = []
                for path in nx.all_simple_paths(G, foundation, concept_id):
                    # Verify path uses prerequisite edges
                    valid = True
                    for i in range(len(path) - 1):
                        edges = G.get_edge_data(path[i], path[i+1])
                        if not any(e.get("relationship_type") == "prerequisite" for e in edges.values()):
                            valid = False
                            break
                    if valid:
                        paths.append(path)
                chains.extend(paths)
            except nx.NetworkXNoPath:
                continue

    return chains


def export_graph_summary(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Export graph statistics and summary.

    Returns:
        Dict with graph statistics
    """
    concepts = get_concept_nodes(G)
    resources = get_resource_nodes(G)
    examples = get_example_nodes(G)

    # Count edge types
    edge_types = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relationship_type", "unknown")
        edge_types[rel] = edge_types.get(rel, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "concepts": len(concepts),
        "resources": len(resources),
        "examples": len(examples),
        "edge_types": edge_types,
        "is_directed": G.is_directed(),
        "is_multigraph": G.is_multigraph(),
    }


if __name__ == "__main__":
    # Build and save the graph
    graph = build_knowledge_graph()
    save_graph(graph)

    # Print summary
    summary = export_graph_summary(graph)
    print("\nGraph Summary:")
    print(json.dumps(summary, indent=2))
