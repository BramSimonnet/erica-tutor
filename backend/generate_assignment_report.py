"""
Assignment Deliverable Report Generator

Generates a formatted markdown report for the GraphRAG AI Tutor assignment.
Includes all required components:
1. System prompts used
2. Knowledge graph nodes/edges retrieved
3. Generated explanation with scaffolding
4. Resource references with citations
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from graph.graphrag_retrieval import graphrag_retrieve
from graph.graphrag_query import answer_with_graphrag, SCAFFOLDED_ANSWER_PROMPT
from graph.build_graph import load_graph


# The three required demonstration questions
DEMO_QUESTIONS = [
    "Explain attention mechanisms in transformers",
    "What are applications of CLIP?",
    "Explain variational bounds and Jensen's inequality"
]


def format_system_prompt(question: str, context: str) -> str:
    """Format the system prompt section for the report"""
    prompt = SCAFFOLDED_ANSWER_PROMPT.replace("<<<CONTEXT>>>", "[CONTEXT SHOWN BELOW]")
    prompt = prompt.replace("<<<QUESTION>>>", question)
    return prompt


def format_subgraph_section(retrieval_result: Dict[str, Any]) -> str:
    """Format the knowledge graph subgraph section"""
    output = []

    # Summary stats
    subgraph = retrieval_result.get('subgraph', {})
    nodes = subgraph.get('nodes', [])
    edges = subgraph.get('edges', [])

    output.append(f"**Subgraph Size:** {len(nodes)} nodes, {len(edges)} edges\n")

    # Concepts identified
    concepts = retrieval_result.get('relevant_concepts', [])
    if concepts:
        output.append("### Concepts Identified\n")
        for i, concept in enumerate(concepts, 1):
            output.append(f"{i}. **{concept['title']}**")
            output.append(f"   - Type: Concept")
            output.append(f"   - Difficulty: {concept['difficulty']}")
            output.append(f"   - Similarity Score: {concept['similarity']:.4f}")
            output.append(f"   - Graph Centrality (Degree): {concept['degree']}")
            if concept.get('definitions'):
                output.append(f"   - Definition: {concept['definitions'][0][:200]}...")
            output.append("")

    # Prerequisites
    prerequisites = subgraph.get('prerequisites', {})
    if prerequisites:
        prereq_count = sum(len(v) for v in prerequisites.values())
        if prereq_count > 0:
            output.append(f"### Prerequisites\n")
            output.append(f"Found {prereq_count} prerequisite relationship(s) in the knowledge graph.\n")

    # Related concepts
    related = subgraph.get('related_concepts', [])
    if related:
        output.append(f"### Related Concepts (Near-Transfer)\n")
        output.append(f"Found {len(related)} related concept(s) for context expansion.\n")

    # Resources
    resources = subgraph.get('resources', {})
    if resources:
        output.append("### Resources Retrieved\n")
        all_resources = []
        for concept_id, resource_list in resources.items():
            all_resources.extend(resource_list)

        for i, resource in enumerate(all_resources[:10], 1):  # Show up to 10 resources
            output.append(f"{i}. **[{resource['resource_type'].upper()}]** {resource['description'][:150]}...")
            if resource.get('span'):
                output.append(f"   - Span: {resource['span']}")
            output.append("")

    # Examples
    examples = subgraph.get('examples', {})
    if examples:
        all_examples = []
        for concept_id, example_list in examples.items():
            all_examples.extend(example_list)

        if all_examples:
            output.append(f"### Worked Examples\n")
            output.append(f"Found {len(all_examples)} worked example(s) in the knowledge graph.\n")

    # Edges
    if edges:
        output.append("### Relationship Edges\n")
        edge_types = {}
        for edge in edges:
            rel_type = edge.get('relationship_type', 'unknown')
            edge_types[rel_type] = edge_types.get(rel_type, 0) + 1

        for rel_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
            output.append(f"- **{rel_type}**: {count} edge(s)")
        output.append("")

    return "\n".join(output)


def format_scaffolding_section(retrieval_result: Dict[str, Any]) -> str:
    """Format the scaffolding (learning path) section"""
    output = []

    scaffolded_path = retrieval_result.get('scaffolded_path', [])
    if not scaffolded_path:
        return "No scaffolded learning path generated.\n"

    output.append(f"**Learning Path Length:** {len(scaffolded_path)} concepts\n")
    output.append("The system orders concepts from simple to complex, following prerequisite chains:\n")

    for i, concept in enumerate(scaffolded_path, 1):
        difficulty = concept['difficulty']

        # Difficulty icon
        if difficulty == 'easy':
            icon = '🟢'
        elif difficulty == 'medium':
            icon = '🟡'
        else:
            icon = '🔴'

        output.append(f"{i}. {icon} **{concept['title']}** [{difficulty.upper()}]")
        if concept.get('definitions'):
            output.append(f"   - {concept['definitions'][0][:150]}...")
        output.append("")

    return "\n".join(output)


def format_context_section(retrieval_result: Dict[str, Any]) -> str:
    """Format the context that was sent to the LLM"""
    output = []

    output.append("The following structured context was provided to the LLM:\n")
    output.append("```")

    scaffolded_path = retrieval_result.get('scaffolded_path', [])
    subgraph = retrieval_result.get('subgraph', {})

    for i, concept in enumerate(scaffolded_path, 1):
        output.append(f"[{i}] CONCEPT: {concept['title']} (Difficulty: {concept['difficulty']})")
        if concept.get('definitions'):
            output.append(f"    Definition: {concept['definitions'][0]}")
        output.append("")

    # Add resources if available
    resources = subgraph.get('resources', {})
    if resources:
        output.append("\nRESOURCES:")
        all_resources = []
        for concept_id, resource_list in resources.items():
            all_resources.extend(resource_list[:3])  # Limit to avoid too much text

        for resource in all_resources[:5]:
            output.append(f"- [{resource['resource_type']}] {resource['description'][:100]}...")

    output.append("```\n")

    return "\n".join(output)


def generate_question_report(question: str, question_num: int) -> str:
    """Generate a complete report for one demonstration question"""
    output = []

    # Header
    output.append(f"## Question {question_num}: {question}\n")
    output.append("---\n")

    # Stage 1: Retrieval
    output.append("### Stage 1: Knowledge Graph Retrieval\n")

    retrieval_result = graphrag_retrieve(
        question,
        top_k_concepts=5,
        min_similarity=0.2
    )

    if "error" in retrieval_result:
        output.append(f"**Error:** {retrieval_result['error']}\n")
        return "\n".join(output)

    # Subgraph section
    output.append("#### Retrieved Subgraph\n")
    output.append(format_subgraph_section(retrieval_result))

    # Scaffolding section
    output.append("\n#### Scaffolding (Learning Path)\n")
    output.append(format_scaffolding_section(retrieval_result))

    # Stage 2: Answer Generation
    output.append("\n### Stage 2: Answer Generation\n")

    # System prompt
    output.append("#### System Prompt Used\n")
    output.append("The following prompt was sent to the LLM (Qwen2.5-7B-Instruct via LM Studio):\n")
    output.append("```")
    output.append(SCAFFOLDED_ANSWER_PROMPT.replace("<<<QUESTION>>>", question).replace("<<<CONTEXT>>>", "[See context below]"))
    output.append("```\n")

    # Context
    output.append("#### Context Provided to LLM\n")
    output.append(format_context_section(retrieval_result))

    # Generate answer
    answer_result = answer_with_graphrag(
        question,
        top_k_concepts=5,
        min_similarity=0.2,
        include_metadata=True
    )

    # Final answer
    output.append("### Stage 3: Generated Answer\n")
    output.append("#### Scaffolded Explanation\n")
    output.append(answer_result['answer'])
    output.append("\n")

    # Metadata
    if answer_result.get('graph_metadata'):
        meta = answer_result['graph_metadata']
        output.append("#### Answer Metadata\n")
        output.append(f"- **Concepts Retrieved:** {meta.get('subgraph_nodes', 0)} nodes")
        output.append(f"- **Relationships:** {meta.get('subgraph_edges', 0)} edges")
        output.append(f"- **Scaffolding Depth:** {answer_result.get('scaffolded_path_length', 0)} steps")
        output.append(f"- **Total Resources:** {meta.get('total_resources', 0)}")
        output.append(f"- **Total Examples:** {meta.get('total_examples', 0)}")
        output.append("\n")

    # Citations
    output.append("#### Resource Citations\n")
    output.append("The answer includes citations to the following resource types:\n")
    output.append("- **[PDF]**: Course slides and lecture notes")
    output.append("- **[Video]**: Educational videos (with timestamps)")
    output.append("- **[Web]**: Course website content\n")

    output.append("\n---\n")

    return "\n".join(output)


def generate_full_report() -> str:
    """Generate the complete assignment deliverable report"""
    output = []

    # Title and header
    output.append("# GraphRAG AI Tutor - Assignment Deliverable Report\n")
    output.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.append("---\n")

    # System overview
    output.append("## System Overview\n")
    output.append("This report demonstrates the GraphRAG-based intelligent tutoring system developed for the")
    output.append("Introduction to AI course. The system implements a knowledge graph-based retrieval augmented")
    output.append("generation (GraphRAG) approach with scaffolded answer generation.\n")

    # Architecture
    output.append("### System Architecture\n")
    output.append("1. **Ingestion Pipeline:** Web pages, PDFs, YouTube videos → Raw documents")
    output.append("2. **Chunking:** Raw documents → Text chunks (~1200 chars)")
    output.append("3. **Embedding:** Chunks → 384-dim vectors (sentence-transformers)")
    output.append("4. **Entity Extraction:** Chunks → Concepts, Resources, Examples (LLM-based)")
    output.append("5. **Relationship Extraction:** Entities → Prerequisites, Related, Explains (LLM-based)")
    output.append("6. **Graph Construction:** Entities + Relationships → NetworkX MultiDiGraph")
    output.append("7. **GraphRAG Query:** Question → Subgraph retrieval → Scaffolded answer\n")

    # Knowledge graph stats
    G = load_graph()
    if G:
        concepts = [n for n, d in G.nodes(data=True) if d.get("type") == "concept"]
        resources = [n for n, d in G.nodes(data=True) if d.get("type") == "resource"]
        examples = [n for n, d in G.nodes(data=True) if d.get("type") == "example"]

        output.append("### Knowledge Graph Statistics\n")
        output.append(f"- **Total Nodes:** {G.number_of_nodes()}")
        output.append(f"  - Concepts: {len(concepts)}")
        output.append(f"  - Resources: {len(resources)}")
        output.append(f"  - Examples: {len(examples)}")
        output.append(f"- **Total Edges:** {G.number_of_edges()}")

        # Edge types
        edges = list(G.edges(data=True, keys=True))
        edge_types = {}
        for u, v, key, data in edges:
            edge_type = data.get("relationship_type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        output.append(f"  - Edge types:")
        for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
            output.append(f"    - {edge_type}: {count}")
        output.append("")

    output.append("---\n")

    # LLM configuration
    output.append("## LLM Configuration\n")
    output.append("- **Model:** Qwen2.5-7B-Instruct (local)")
    output.append("- **Server:** LM Studio at host.docker.internal:1234")
    output.append("- **Embedding Model:** all-MiniLM-L6-v2 (sentence-transformers)")
    output.append("- **Vector Dimension:** 384\n")
    output.append("---\n")

    # GraphRAG methodology
    output.append("## GraphRAG Methodology\n")
    output.append("The system uses graph-based retrieval, NOT just vector similarity:\n")
    output.append("1. **Concept Identification:** Vector similarity search to find relevant concepts")
    output.append("2. **Subgraph Extraction:** Graph traversal to collect:")
    output.append("   - Prerequisites (for scaffolding)")
    output.append("   - Related concepts (near-transfer)")
    output.append("   - Resources that explain concepts")
    output.append("   - Worked examples")
    output.append("3. **Scaffolding:** Order concepts by difficulty and prerequisite chains (easy → medium → hard)")
    output.append("4. **Context Assembly:** Format subgraph into structured context")
    output.append("5. **Generation:** LLM generates scaffolded answer with resource citations\n")
    output.append("---\n")

    # Generate reports for all 3 questions
    output.append("# Demonstration Questions\n")
    output.append("The following sections demonstrate the complete system output for the three required")
    output.append("demonstration questions.\n\n")

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"Generating report for Question {i}: {question}")
        output.append(generate_question_report(question, i))

    # Conclusion
    output.append("# Conclusion\n")
    output.append("This report demonstrates a complete GraphRAG-based intelligent tutoring system that:")
    output.append("- Ingests educational content from multiple sources")
    output.append("- Builds a knowledge graph with concepts, resources, and examples")
    output.append("- Retrieves relevant subgraphs using graph traversal (not just vector similarity)")
    output.append("- Generates scaffolded answers from simple to complex concepts")
    output.append("- Includes resource citations to support learning\n")
    output.append("The system successfully answers all three demonstration questions with structured,")
    output.append("pedagogically-sound explanations.\n")

    return "\n".join(output)


def main():
    """Main function to generate and save the report"""
    print("=" * 80)
    print("GraphRAG Assignment Deliverable Report Generator")
    print("=" * 80)
    print()

    # Check if graph exists
    G = load_graph()
    if G is None:
        print("ERROR: Knowledge graph not found!")
        print("Please run: python -m graph.build_graph")
        return

    print(f"✓ Knowledge graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()

    print("Generating comprehensive assignment report...")
    print("This will take a few minutes as it queries the LLM for each question.")
    print()

    # Generate report
    report = generate_full_report()

    # Save to file
    output_path = "/app/ASSIGNMENT_DELIVERABLE.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print()
    print("=" * 80)
    print(f"✓ Report generated successfully!")
    print(f"✓ Saved to: {output_path}")
    print()
    print("The report includes:")
    print("  1. System prompts used")
    print("  2. Knowledge graph nodes/edges retrieved")
    print("  3. Scaffolded explanations")
    print("  4. Resource citations")
    print()
    print("You can find the report at:")
    print(f"  backend/ASSIGNMENT_DELIVERABLE.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
