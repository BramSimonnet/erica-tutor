"""
Community detection for organizing concepts into topic clusters.
Uses Louvain algorithm for modularity-based clustering.
"""
import networkx as nx
from typing import Dict, List, Set, Any
import json


def detect_communities(G: nx.MultiDiGraph) -> Dict[int, List[str]]:
    """
    Detect communities in the knowledge graph using Louvain algorithm.
    Communities represent topic clusters of related concepts.

    Args:
        G: Knowledge graph (directed multigraph)

    Returns:
        Dict mapping community_id -> list of node IDs
    """
    # Convert to undirected graph for community detection
    # Only use concept nodes and their relationships
    concept_graph = nx.Graph()

    for node, data in G.nodes(data=True):
        if data.get("type") == "concept":
            concept_graph.add_node(node, **data)

    # Add edges between concepts (related and prerequisite)
    for source, target, data in G.edges(data=True):
        if concept_graph.has_node(source) and concept_graph.has_node(target):
            rel_type = data.get("relationship")
            if rel_type in ["related", "prerequisite"]:
                # Weight prerequisites higher as they indicate strong semantic connection
                weight = 2.0 if rel_type == "prerequisite" else 1.0
                concept_graph.add_edge(source, target, weight=weight)

    # Detect communities using Louvain algorithm (greedy modularity)
    communities = nx.community.louvain_communities(concept_graph, seed=42)

    # Convert to dict format
    community_dict = {}
    for i, community in enumerate(communities):
        community_dict[i] = list(community)

    print(f"Detected {len(community_dict)} communities")
    return community_dict


def get_community_summary(G: nx.MultiDiGraph, community_id: int, communities: Dict[int, List[str]]) -> Dict[str, Any]:
    """
    Get summary information about a community.

    Args:
        G: Knowledge graph
        community_id: Community ID
        communities: Community assignments

    Returns:
        Dict with community statistics and members
    """
    if community_id not in communities:
        return {}

    members = communities[community_id]

    # Get concept titles and difficulties
    concepts = []
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}

    for node_id in members:
        node_data = G.nodes[node_id]
        concepts.append({
            "id": node_id,
            "title": node_data.get("title", ""),
            "difficulty": node_data.get("difficulty", "medium")
        })
        difficulty = node_data.get("difficulty", "medium")
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    # Count internal vs external edges
    internal_edges = 0
    external_edges = 0

    for source, target in G.edges():
        if source in members and target in members:
            internal_edges += 1
        elif source in members or target in members:
            external_edges += 1

    return {
        "community_id": community_id,
        "size": len(members),
        "concepts": concepts,
        "difficulty_distribution": difficulty_counts,
        "internal_edges": internal_edges,
        "external_edges": external_edges,
    }


def find_community_for_concept(concept_id: str, communities: Dict[int, List[str]]) -> int:
    """
    Find which community a concept belongs to.

    Args:
        concept_id: Node ID
        communities: Community assignments

    Returns:
        Community ID or -1 if not found
    """
    for comm_id, members in communities.items():
        if concept_id in members:
            return comm_id
    return -1


def get_related_communities(
    G: nx.MultiDiGraph,
    community_id: int,
    communities: Dict[int, List[str]]
) -> List[int]:
    """
    Find communities that are connected to this community.

    Args:
        G: Knowledge graph
        community_id: Source community ID
        communities: Community assignments

    Returns:
        List of related community IDs
    """
    if community_id not in communities:
        return []

    members = communities[community_id]
    related_communities = set()

    # Find edges crossing community boundaries
    for node in members:
        for neighbor in G.neighbors(node):
            neighbor_comm = find_community_for_concept(neighbor, communities)
            if neighbor_comm != -1 and neighbor_comm != community_id:
                related_communities.add(neighbor_comm)

    return list(related_communities)


def export_communities(
    G: nx.MultiDiGraph,
    communities: Dict[int, List[str]],
    filepath: str = "/app/data/communities.json"
):
    """
    Export community data to JSON file.

    Args:
        G: Knowledge graph
        communities: Community assignments
        filepath: Output file path
    """
    export_data = {
        "total_communities": len(communities),
        "communities": {}
    }

    for comm_id in communities:
        summary = get_community_summary(G, comm_id, communities)
        export_data["communities"][str(comm_id)] = summary

    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Communities exported to {filepath}")


def get_community_hierarchy(
    G: nx.MultiDiGraph,
    communities: Dict[int, List[str]]
) -> Dict[str, Any]:
    """
    Build a hierarchy based on prerequisite relationships between communities.

    Args:
        G: Knowledge graph
        communities: Community assignments

    Returns:
        Dict describing the community hierarchy
    """
    # Count prerequisite edges between communities
    prereq_counts = {}

    for source, target, data in G.edges(data=True):
        if data.get("relationship") == "prerequisite":
            source_comm = find_community_for_concept(source, communities)
            target_comm = find_community_for_concept(target, communities)

            if source_comm != -1 and target_comm != -1 and source_comm != target_comm:
                key = (source_comm, target_comm)
                prereq_counts[key] = prereq_counts.get(key, 0) + 1

    # Build adjacency list
    hierarchy = {}
    for (source_comm, target_comm), count in prereq_counts.items():
        if source_comm not in hierarchy:
            hierarchy[source_comm] = []
        hierarchy[source_comm].append({
            "target_community": target_comm,
            "prerequisite_count": count
        })

    return hierarchy


if __name__ == "__main__":
    from graph.build_graph import load_graph, export_graph_summary
    import json

    # Load the graph
    G = load_graph()
    if G is None:
        print("No graph found. Run build_graph.py first.")
        exit(1)

    # Detect communities
    communities = detect_communities(G)

    # Print summaries
    print("\nCommunity Summaries:")
    for comm_id in communities:
        summary = get_community_summary(G, comm_id, communities)
        print(f"\nCommunity {comm_id}:")
        print(f"  Size: {summary['size']} concepts")
        print(f"  Difficulty: {summary['difficulty_distribution']}")
        print(f"  Concepts: {[c['title'] for c in summary['concepts'][:5]]}")

    # Export
    export_communities(G, communities)

    # Show hierarchy
    hierarchy = get_community_hierarchy(G, communities)
    print(f"\nCommunity hierarchy (prerequisite relationships):")
    print(json.dumps(hierarchy, indent=2))
