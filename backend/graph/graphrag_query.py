"""
GraphRAG Query Module: Answer questions using knowledge graph retrieval.
Implements scaffolded generation (simple → complex) with resource citations.
"""
from typing import Dict, Any, List
from graph.graphrag_retrieval import graphrag_retrieve
from llm.qwenclient import call_llm
import json


SCAFFOLDED_ANSWER_PROMPT = """You are an expert AI tutor. Answer the student's question using the provided context.

IMPORTANT INSTRUCTIONS:
1. Structure your answer from SIMPLE to COMPLEX concepts (follow the scaffolded learning path)
2. Start with foundational concepts, then build up to more advanced ones
3. Include specific CITATIONS to resources (e.g., "[PDF, page 5]", "[Video, 12:34]")
4. Use worked examples when available
5. Explain prerequisites before explaining dependent concepts

Context (organized from simple → complex):
<<<CONTEXT>>>

Student Question: <<<QUESTION>>>

Provide a clear, scaffolded explanation that builds understanding step by step. Include resource citations in your answer."""


def answer_with_graphrag(
    question: str,
    top_k_concepts: int = 3,
    min_similarity: float = 0.3,
    include_metadata: bool = False
) -> Dict[str, Any]:
    """
    Answer a question using GraphRAG retrieval and scaffolded generation.

    Args:
        question: Student's question
        top_k_concepts: Number of concepts to retrieve
        min_similarity: Minimum similarity threshold
        include_metadata: Whether to include graph metadata in response

    Returns:
        Dict with:
            - question: original question
            - answer: scaffolded answer with citations
            - graph_metadata: retrieved graph structure (if include_metadata=True)
            - system_prompt: prompt used for generation
    """
    # Step 1: Retrieve relevant subgraph using GraphRAG
    retrieval_result = graphrag_retrieve(
        question,
        top_k_concepts=top_k_concepts,
        min_similarity=min_similarity
    )

    # Check for errors
    if "error" in retrieval_result:
        return {
            "question": question,
            "answer": f"Error: {retrieval_result['error']}",
            "graph_metadata": {}
        }

    # Check if we found anything
    if not retrieval_result["relevant_concepts"]:
        return {
            "question": question,
            "answer": "I couldn't find any relevant information in the knowledge base to answer this question. Please rephrase or ask about topics covered in the course materials.",
            "graph_metadata": {}
        }

    # Step 2: Build prompt with scaffolded context
    context = retrieval_result["context_summary"]
    prompt = SCAFFOLDED_ANSWER_PROMPT.replace("<<<CONTEXT>>>", context)
    prompt = prompt.replace("<<<QUESTION>>>", question)

    # Step 3: Generate answer with LLM
    answer = call_llm(prompt)

    # Step 4: Prepare response
    response = {
        "question": question,
        "answer": answer,
        "num_concepts": len(retrieval_result["relevant_concepts"]),
        "scaffolded_path_length": len(retrieval_result["scaffolded_path"])
    }

    if include_metadata:
        response["graph_metadata"] = {
            "relevant_concepts": retrieval_result["relevant_concepts"],
            "scaffolded_path": retrieval_result["scaffolded_path"],
            "subgraph_nodes": len(retrieval_result["subgraph"].get("nodes", [])),
            "subgraph_edges": len(retrieval_result["subgraph"].get("edges", [])),
            "prerequisites": retrieval_result["subgraph"].get("prerequisites", {}),
            "resources_used": retrieval_result["subgraph"].get("resources", {}),
            "examples_used": retrieval_result["subgraph"].get("examples", {})
        }
        response["system_prompt"] = prompt

    return response


def generate_demonstration_answers():
    """
    Generate answers for the three required demonstration questions.
    Saves output with complete metadata for assignment deliverables.
    """
    demonstration_questions = [
        "Explain attention mechanisms in transformers",
        "What are applications of CLIP?",
        "Explain variational bounds and Jensen's inequality"
    ]

    results = {}

    print("=== Generating Demonstration Answers ===\n")

    for i, question in enumerate(demonstration_questions, 1):
        print(f"Question {i}: {question}")

        result = answer_with_graphrag(
            question,
            top_k_concepts=5,
            min_similarity=0.2,
            include_metadata=True
        )

        results[f"question_{i}"] = result

        print(f"  Answer length: {len(result['answer'])} chars")
        print(f"  Concepts used: {result['num_concepts']}")
        print(f"  Scaffolding depth: {result['scaffolded_path_length']}")
        print()

    # Save to file
    output_file = "/app/data/demonstration_answers.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Demonstration answers saved to {output_file}")

    # Also create a readable markdown version
    md_output = "/app/data/demonstration_answers.md"
    with open(md_output, 'w') as f:
        f.write("# Erica AI Tutor - Demonstration Answers\n\n")

        for i, question in enumerate(demonstration_questions, 1):
            result = results[f"question_{i}"]
            f.write(f"## Question {i}: {question}\n\n")

            f.write("### Answer\n\n")
            f.write(result['answer'])
            f.write("\n\n")

            f.write("### Metadata\n\n")
            f.write(f"- **Concepts Retrieved**: {result['num_concepts']}\n")
            f.write(f"- **Scaffolded Path Length**: {result['scaffolded_path_length']}\n")
            f.write(f"- **Subgraph Size**: {result['graph_metadata']['subgraph_nodes']} nodes, {result['graph_metadata']['subgraph_edges']} edges\n\n")

            if result['graph_metadata']['relevant_concepts']:
                f.write("**Main Concepts Used**:\n")
                for concept in result['graph_metadata']['relevant_concepts']:
                    f.write(f"- {concept['title']} (similarity: {concept['similarity']:.3f}, difficulty: {concept['difficulty']})\n")
                f.write("\n")

            if result['graph_metadata']['scaffolded_path']:
                f.write("**Scaffolded Learning Path** (Simple → Complex):\n")
                for j, concept in enumerate(result['graph_metadata']['scaffolded_path'], 1):
                    f.write(f"{j}. {concept['title']} [{concept['difficulty']}]\n")
                f.write("\n")

            f.write("### System Prompt\n\n")
            f.write("```\n")
            f.write(result['system_prompt'])
            f.write("\n```\n\n")
            f.write("---\n\n")

    print(f"✓ Markdown report saved to {md_output}")

    return results


def interactive_graphrag_tutor():
    """
    Interactive CLI for GraphRAG-based tutoring.
    """
    print("=" * 60)
    print("Erica AI Tutor - GraphRAG Mode")
    print("=" * 60)
    print("Ask questions about course material.")
    print("The system uses knowledge graph traversal for retrieval.")
    print("Type 'demo' to generate demonstration answers.")
    print("Type 'quit' or 'exit' to end.\n")

    while True:
        try:
            question = input("\n🎓 Your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! Happy learning! 📚")
                break

            if question.lower() == "demo":
                generate_demonstration_answers()
                continue

            if not question:
                continue

            print("\n🤔 Retrieving from knowledge graph...")

            result = answer_with_graphrag(
                question,
                top_k_concepts=3,
                min_similarity=0.2,
                include_metadata=True
            )

            print(f"\n💡 Answer:\n{result['answer']}\n")
            print(f"📊 Used {result['num_concepts']} concepts with {result['scaffolded_path_length']}-step scaffolding")

            # Show graph info
            if result.get('graph_metadata'):
                meta = result['graph_metadata']
                print(f"📈 Graph: {meta['subgraph_nodes']} nodes, {meta['subgraph_edges']} edges")

                if meta.get('relevant_concepts'):
                    print("\n📚 Main concepts:")
                    for concept in meta['relevant_concepts'][:3]:
                        print(f"   • {concept['title']} (similarity: {concept['similarity']:.2f})")

        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy learning! 📚")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        generate_demonstration_answers()
    else:
        interactive_graphrag_tutor()
