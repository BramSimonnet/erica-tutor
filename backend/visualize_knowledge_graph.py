"""
Knowledge Graph Visualization Script

"""

import matplotlib.pyplot as plt
import networkx as nx
from graph.build_graph import load_graph
from typing import Set, Dict, Any
import os


def get_concept_neighborhood(G, concept_id: str, depth: int = 1) -> nx.MultiDiGraph:
    """
    Extract a subgraph centered on a specific concept.

    Args:
        G: The full knowledge graph
        concept_id: The ID of the concept to center on
        depth: How many hops to include (default 1)

    Returns:
        Subgraph containing the concept and its neighborhood
    """
    # Get all nodes within depth hops
    neighbors = set([concept_id])

    for _ in range(depth):
        new_neighbors = set()
        for node in neighbors:
            # Add successors (outgoing edges)
            new_neighbors.update(G.successors(node))
            # Add predecessors (incoming edges)
            new_neighbors.update(G.predecessors(node))
        neighbors.update(new_neighbors)

    # Create subgraph
    subgraph = G.subgraph(neighbors).copy()
    return subgraph


def visualize_concept_neighborhood(
    G,
    concept_id: str,
    output_path: str,
    title: str = None,
    figsize: tuple = (16, 12)
):
    """
    Visualize a concept and its direct relationships.

    Args:
        G: Knowledge graph
        concept_id: ID of the concept to visualize
        output_path: Where to save the visualization
        title: Optional custom title
        figsize: Figure size (width, height)
    """
    # Get concept data
    if concept_id not in G.nodes:
        print(f"Error: Concept {concept_id} not found in graph")
        return

    concept_data = G.nodes[concept_id]
    concept_title = concept_data.get('title', concept_id)

    # Extract neighborhood
    subgraph = get_concept_neighborhood(G, concept_id, depth=1)

    # Create figure
    plt.figure(figsize=figsize)

    # Position nodes using spring layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    # Put the central concept in the center
    pos[concept_id] = (0, 0)

    # Separate nodes by type
    concepts = [n for n in subgraph.nodes() if subgraph.nodes[n].get('type') == 'concept']
    resources = [n for n in subgraph.nodes() if subgraph.nodes[n].get('type') == 'resource']
    examples = [n for n in subgraph.nodes() if subgraph.nodes[n].get('type') == 'example']

    # Draw nodes by type with different colors/shapes
    # Central concept (highlighted)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[concept_id],
                          node_color='#FF6B6B', node_size=3000,
                          node_shape='o', label='Central Concept', alpha=0.9)

    # Other concepts
    other_concepts = [c for c in concepts if c != concept_id]
    if other_concepts:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=other_concepts,
                              node_color='#4ECDC4', node_size=2000,
                              node_shape='o', label='Related Concepts', alpha=0.8)

    # Resources
    if resources:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=resources,
                              node_color='#95E1D3', node_size=1500,
                              node_shape='s', label='Resources', alpha=0.8)

    # Examples
    if examples:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=examples,
                              node_color='#F38181', node_size=1500,
                              node_shape='^', label='Examples', alpha=0.8)

    # Draw edges by relationship type
    edge_colors = {
        'prerequisite': '#E74C3C',  # Red
        'related': '#3498DB',        # Blue
        'explains': '#2ECC71',       # Green
        'example_of': '#F39C12'      # Orange
    }

    # Group edges by type
    edges_by_type = {}
    for u, v, key, data in subgraph.edges(data=True, keys=True):
        rel_type = data.get('relationship_type', 'unknown')
        if rel_type not in edges_by_type:
            edges_by_type[rel_type] = []
        edges_by_type[rel_type].append((u, v))

    # Draw edges by type
    for rel_type, edges in edges_by_type.items():
        color = edge_colors.get(rel_type, '#95A5A6')
        nx.draw_networkx_edges(subgraph, pos, edgelist=edges,
                              edge_color=color, width=2, alpha=0.6,
                              arrows=True, arrowsize=20,
                              connectionstyle='arc3,rad=0.1',
                              label=f'{rel_type} ({len(edges)})')

    # Draw labels
    labels = {}
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        node_type = node_data.get('type', 'unknown')

        # Get label based on node type
        if node_type == 'concept':
            node_title = node_data.get('title', 'Unknown Concept')
        elif node_type == 'resource':
            node_title = node_data.get('description', 'Unknown Resource')
        elif node_type == 'example':
            node_title = node_data.get('text', 'Unknown Example')
        else:
            node_title = 'Unknown'

        # Truncate long titles
        if len(node_title) > 30:
            node_title = node_title[:27] + '...'

        labels[node] = node_title

    nx.draw_networkx_labels(subgraph, pos, labels, font_size=9, font_weight='bold')

    # Add title
    if title is None:
        title = f"Knowledge Graph: {concept_title}\nand its Relationships"
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    # Add legend
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Add statistics text box
    stats_text = f"Nodes: {subgraph.number_of_nodes()} | Edges: {subgraph.number_of_edges()}\n"
    stats_text += f"Concepts: {len(concepts)} | Resources: {len(resources)} | Examples: {len(examples)}"
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.axis('off')
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def create_full_graph_overview(G, output_path: str, figsize: tuple = (20, 16)):
    """
    Create an overview visualization of the entire knowledge graph.
    """
    plt.figure(figsize=figsize)

    # Use spring layout for overall structure
    pos = nx.spring_layout(G, k=1, iterations=30, seed=42)

    # Separate nodes by type
    concepts = [n for n in G.nodes() if G.nodes[n].get('type') == 'concept']
    resources = [n for n in G.nodes() if G.nodes[n].get('type') == 'resource']
    examples = [n for n in G.nodes() if G.nodes[n].get('type') == 'example']

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=concepts,
                          node_color='#4ECDC4', node_size=100,
                          node_shape='o', label='Concepts', alpha=0.7)

    nx.draw_networkx_nodes(G, pos, nodelist=resources,
                          node_color='#95E1D3', node_size=50,
                          node_shape='s', label='Resources', alpha=0.6)

    nx.draw_networkx_nodes(G, pos, nodelist=examples,
                          node_color='#F38181', node_size=50,
                          node_shape='^', label='Examples', alpha=0.6)

    # Draw edges (thin for overview)
    nx.draw_networkx_edges(G, pos, edge_color='#95A5A6',
                          width=0.5, alpha=0.3, arrows=True, arrowsize=5)

    plt.title("Complete Knowledge Graph Overview", fontsize=18, fontweight='bold', pad=20)
    plt.legend(loc='upper left', fontsize=12)

    # Add statistics
    stats_text = f"Total Nodes: {G.number_of_nodes()}\n"
    stats_text += f"Total Edges: {G.number_of_edges()}\n"
    stats_text += f"Concepts: {len(concepts)}\n"
    stats_text += f"Resources: {len(resources)}\n"
    stats_text += f"Examples: {len(examples)}"

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved overview: {output_path}")
    plt.close()


def find_interesting_concepts(G, count: int = 5) -> list:
    """
    Find interesting concepts to visualize based on:
    - Centrality (number of connections)
    - Diversity of connection types
    - Presence of resources and examples
    """
    concept_scores = []

    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data.get('type') != 'concept':
            continue

        # Calculate score
        degree = G.degree(node)

        # Count different types of neighbors
        neighbors = list(G.successors(node)) + list(G.predecessors(node))
        neighbor_types = set(G.nodes[n].get('type') for n in neighbors if n in G.nodes)

        # Bonus for having multiple types of relationships
        diversity_score = len(neighbor_types) * 2

        total_score = degree + diversity_score

        concept_scores.append((node, node_data.get('title', node), total_score, degree))

    # Sort by score
    concept_scores.sort(key=lambda x: x[2], reverse=True)

    return concept_scores[:count]


def main():
    """Generate knowledge graph visualizations"""
    print("=" * 80)
    print("Knowledge Graph Visualization Generator")
    print("=" * 80)
    print()

    # Load graph
    G = load_graph()
    if G is None:
        print("ERROR: Knowledge graph not found!")
        print("Please run: python -m graph.build_graph")
        return

    print(f"✓ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()

    # Create output directory
    output_dir = "/app/data/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create full graph overview
    print("Generating full graph overview...")
    create_full_graph_overview(G, f"{output_dir}/graph_overview.png")
    print()

    # 2. Find interesting concepts
    print("Finding interesting concepts to visualize...")
    interesting_concepts = find_interesting_concepts(G, count=5)

    print(f"Selected {len(interesting_concepts)} concepts for detailed visualization:")
    for i, (node_id, title, score, degree) in enumerate(interesting_concepts, 1):
        print(f"  {i}. {title} (degree: {degree}, score: {score})")
    print()

    # 3. Create detailed visualizations for top concepts
    print("Generating concept neighborhood visualizations...")
    for i, (node_id, title, score, degree) in enumerate(interesting_concepts, 1):
        # Clean title for filename
        safe_title = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in title)
        safe_title = safe_title.replace(' ', '_')[:50]

        output_path = f"{output_dir}/concept_{i}_{safe_title}.png"

        visualize_concept_neighborhood(
            G,
            node_id,
            output_path,
            title=f"Concept: {title}\nRelationships to Concepts, Resources, and Examples"
        )

    print()
    print("=" * 80)
    print("✓ Visualization generation complete!")
    print()
    print(f"Generated {len(interesting_concepts) + 1} visualizations:")
    print(f"  - 1 full graph overview")
    print(f"  - {len(interesting_concepts)} concept neighborhood visualizations")
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Files created:")
    print("  - graph_overview.png")
    for i, (_, title, _, _) in enumerate(interesting_concepts, 1):
        safe_title = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in title)[:50]
        safe_title = safe_title.replace(' ', '_')
        print(f"  - concept_{i}_{safe_title}.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
