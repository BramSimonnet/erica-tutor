import json
from pymongo import MongoClient
import networkx as nx
import matplotlib.pyplot as plt

client = MongoClient("mongodb://mongo:27017")
db = client.erica

def load_graph():
    nodes = list(db.entity_nodes.find({}))
    edges = list(db.entity_edges.find({}))

    G = nx.DiGraph()

    for n in nodes:
        node_type = n.get("type", "unknown")
        label = (
            n.get("title")
            or n.get("text")
            or n.get("description")
            or n.get("id")
            or "unknown"
        )
        G.add_node(n["id"], label=label, type=node_type)

    for e in edges:
        if "source" in e and "target" in e:
            G.add_edge(e["source"], e["target"], type=e.get("type", "relation"))

    return G

def visualize():
    G = load_graph()
    pos = nx.spring_layout(G, seed=42)

    colors = []
    for node in G.nodes(data=True):
        t = node[1].get("type", "unknown")
        if t == "concept":
            colors.append("#1f77b4")
        elif t == "resource":
            colors.append("#2ca02c")
        elif t == "example":
            colors.append("#d62728")
        else:
            colors.append("#7f7f7f")

    plt.figure(figsize=(20, 14))

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=15)

    labels = {n: G.nodes[n].get("label", "unknown") for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("graph/knowledge_graph.png", dpi=300)
    print("Saved graph/knowledge_graph.png")


if __name__ == "__main__":
    visualize()
