"""
GraphRAG retrieval with strict output formatting for assignment.
"""
import networkx as nx
from typing import List, Dict, Any
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
from vectorstore.llm import ask_llm
from pymongo import MongoClient

MONGO_URI = "mongodb://mongo:27017"
client = MongoClient(MONGO_URI)
db = client.erica

def find_relevant_concepts(query: str, G: nx.MultiDiGraph, top_k=3):
    q_embed = generate_embedding(query)
    concepts = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "concept"]

    scores = []
    for nid, nd in concepts:
        title = nd.get("title", "")
        defs = nd.get("definitions", [])
        txt = f"{title}. {' '.join(defs)}"
        c_embed = generate_embedding(txt)
        sim = cosine_similarity(q_embed, c_embed)
        if sim > 0.3:
            deg = G.in_degree(nid) + G.out_degree(nid)
            scores.append({
                "node_id": nid,
                "title": title,
                "definitions": defs,
                "similarity": sim,
                "boosted": sim + deg * 0.01
            })

    scores.sort(key=lambda x: x["boosted"], reverse=True)
    return scores[:top_k]

def retrieve_subgraph(concept_ids, G):
    nodes = set(concept_ids)
    resources_by_cid = {}

    for cid in concept_ids:
        res = find_resources_for_concept(G, cid)
        if res:
            resources_by_cid[cid] = res
            for r in res:
                nodes.add(r["id"])

    final_nodes = []
    for nid in nodes:
        if G.has_node(nid):
            nd = G.nodes[nid].copy()
            nd["id"] = nid
            final_nodes.append(nd)

    return {
        "nodes": final_nodes,
        "resources": resources_by_cid
    }


def _format_context_for_llm(relevant, subgraph):
    lines = []

    lines.append("MAIN CONCEPTS:")
    for c in relevant:
        lines.append(f"- {c['title']}: {c['definitions'][0] if c['definitions'] else ''}")

    lines.append("\nRESOURCES:")
    for cid, rlist in subgraph["resources"].items():
        for r in rlist:
            lines.append(f"- {r['description']}")

    return "\n".join(lines)

class GraphRAG:
    def __init__(self, graph_path=None):
        self.G = load_graph()

    def run(self, query: str) -> Dict[str, Any]:
        if self.G is None:
            return {"final_response": "Graph not loaded."}

        relevant = find_relevant_concepts(query, self.G)
        if not relevant:
            return {"final_response": "No matching concepts found."}

        concept_ids = [c["node_id"] for c in relevant]
        sub = retrieve_subgraph(concept_ids, self.G)
        compact_context = _format_context_for_llm(relevant, sub)

        system_prompt = (
            "You are an AI tutor. Your explanation MUST:\n"
            "- Be 3–5 sentences MAX.\n"
            "- Include Jensen’s inequality, variational methods, and how VAEs use Jensen’s inequality WHEN relevant.\n"
            "- NEVER include lists, bullet points, markdown, LaTeX, or long derivations.\n"
            "- Keep explanations short and conceptual.\n"
            "- After the explanation, DO NOT add anything else."
        )

        final_answer = ask_llm(
            f"{system_prompt}\n\n"
            f"User Question: {query}\n\n"
            f"Graph Context:\n{compact_context}\n\n"
            "Write the answer now."
        )

        used_nodes = [n["id"] for n in sub["nodes"]]
        used_resources = sub["resources"]

        return {
            "final_response": final_answer.strip(),
            "nodes_used": used_nodes,
            "resources_used": used_resources,
            "system_prompt": system_prompt
        }

def graphrag_retrieve(query: str, top_k_concepts: int = 3, min_similarity: float = 0.3) -> Dict[str, Any]:
    """
    Retrieve relevant subgraph for a query (wrapper for backward compatibility).

    Returns structured data matching the expected format from graphrag_query.
    """
    G = load_graph()
    if G is None:
        return {"error": "Knowledge graph not loaded"}

    relevant_concepts = find_relevant_concepts(query, G, top_k=top_k_concepts)

    if not relevant_concepts:
        return {"error": "No relevant concepts found"}

    concept_ids = [c["node_id"] for c in relevant_concepts]

    subgraph_data = retrieve_subgraph(concept_ids, G)

    formatted_concepts = []
    for concept in relevant_concepts:
        node_id = concept["node_id"]
        node_data = G.nodes[node_id]
        formatted_concepts.append({
            "id": node_id,
            "title": concept["title"],
            "definitions": concept["definitions"],
            "difficulty": node_data.get("difficulty", "medium"),
            "similarity": concept["similarity"],
            "degree": G.in_degree(node_id) + G.out_degree(node_id)
        })

    scaffolded_path = sorted(formatted_concepts, key=lambda x: {
        'easy': 0, 'medium': 1, 'hard': 2
    }.get(x['difficulty'], 1))

    context_summary = _format_context_for_llm(relevant_concepts, subgraph_data)

    return {
        "relevant_concepts": formatted_concepts,
        "subgraph": subgraph_data,
        "scaffolded_path": scaffolded_path,
        "context_summary": context_summary,
        "num_concepts": len(formatted_concepts),
        "scaffolded_path_length": len(scaffolded_path)
    }
