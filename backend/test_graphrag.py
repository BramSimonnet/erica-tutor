"""
GraphRAG Test Script
Demonstrates GraphRAG pipeline with detailed output at each stage. Checks that this thing works.
Run with python -m test_graphrag
"""
import json
from typing import Dict, Any
from graph.graphrag_retrieval import graphrag_retrieve
from graph.graphrag_query import answer_with_graphrag
from graph.build_graph import load_graph


class Colors:
    """color codes for terminal formatting, convenience"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Prints section header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}\n")


def print_subheader(text: str):
    """Prints subsection header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}▶ {text}{Colors.END}")
    print(f"{Colors.CYAN}{'-'*80}{Colors.END}")


def print_success(text: str):
    """Prints success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_info(text: str):
    """Prints info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_warning(text: str):
    """Prints warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text: str):
    """Prints error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def display_graph_stats():
    """Display stats about the knowledge graph"""
    print_subheader("Knowledge Graph Statistics")

    G = load_graph()
    if G is None:
        print_error("Knowledge graph not found! Run 'python -m graph.build_graph' first.")
        return False

    # Counts nodes by type
    concepts = [n for n, d in G.nodes(data=True) if d.get("type") == "concept"]
    resources = [n for n, d in G.nodes(data=True) if d.get("type") == "resource"]
    examples = [n for n, d in G.nodes(data=True) if d.get("type") == "example"]

    # Counts edges by type
    edges = list(G.edges(data=True, keys=True))
    edge_types = {}
    for u, v, key, data in edges:
        edge_type = data.get("relationship_type", "unknown")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print(f"{Colors.BOLD}Nodes:{Colors.END}")
    print(f"  • Concepts: {Colors.GREEN}{len(concepts)}{Colors.END}")
    print(f"  • Resources: {Colors.BLUE}{len(resources)}{Colors.END}")
    print(f"  • Examples: {Colors.YELLOW}{len(examples)}{Colors.END}")
    print(f"  • Total: {Colors.BOLD}{G.number_of_nodes()}{Colors.END}")

    print(f"\n{Colors.BOLD}Edges:{Colors.END}")
    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {edge_type}: {count}")
    print(f"  • Total: {Colors.BOLD}{G.number_of_edges()}{Colors.END}")

    # Concept difficulty distribution
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    for node_id in concepts:
        difficulty = G.nodes[node_id].get("difficulty", "medium")
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    print(f"\n{Colors.BOLD}Concept Difficulty Distribution:{Colors.END}")
    print(f"  • Easy: {Colors.GREEN}{difficulty_counts['easy']}{Colors.END}")
    print(f"  • Medium: {Colors.YELLOW}{difficulty_counts['medium']}{Colors.END}")
    print(f"  • Hard: {Colors.RED}{difficulty_counts['hard']}{Colors.END}")

    print_success(f"Graph loaded successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return True


def display_retrieval_results(result: Dict[str, Any]):
    """Displays retrieval results, details"""
    print_subheader("Stage 1: Concept Identification (Vector Similarity)")

    if not result.get("relevant_concepts"):
        print_warning("No relevant concepts found")
        return

    print(f"Found {Colors.BOLD}{len(result['relevant_concepts'])}{Colors.END} relevant concepts:\n")
    for i, concept in enumerate(result['relevant_concepts'], 1):
        similarity = concept['similarity']

        # Color code by similarity strength
        if similarity > 0.7:
            sim_color = Colors.GREEN
        elif similarity > 0.5:
            sim_color = Colors.YELLOW
        else:
            sim_color = Colors.RED

        print(f"{Colors.BOLD}{i}. {concept['title']}{Colors.END}")
        print(f"   Similarity: {sim_color}{similarity:.4f}{Colors.END}")
        print(f"   Difficulty: {concept['difficulty']}")
        print(f"   Degree: {concept['degree']} (centrality in graph)")
        if concept['definitions']:
            print(f"   Definition: {concept['definitions'][0][:150]}...")
        print()

    print_subheader("Stage 2: Subgraph Retrieval (Graph Traversal)")

    subgraph = result.get('subgraph', {})
    print(f"Subgraph size: {Colors.BOLD}{len(subgraph.get('nodes', []))} nodes{Colors.END}, "
          f"{Colors.BOLD}{len(subgraph.get('edges', []))} edges{Colors.END}\n")

    # Prerequisites
    prerequisites = subgraph.get('prerequisites', {})
    if prerequisites:
        print(f"{Colors.BOLD}Prerequisites Found:{Colors.END}")
        for concept_id, prereq_list in prerequisites.items():
            if prereq_list:
                print(f"  • {len(prereq_list)} prerequisite(s) for concept")
        print()

    # Related concepts
    related = subgraph.get('related_concepts', [])
    if related:
        print(f"{Colors.BOLD}Related Concepts:{Colors.END} {len(related)} near-transfer concepts found")
        print()

    # Resources
    resources = subgraph.get('resources', {})
    if resources:
        print(f"{Colors.BOLD}Resources Found:{Colors.END}")
        total_resources = sum(len(r) for r in resources.values())
        print(f"  • {total_resources} resources explaining concepts")

        # Show first few resources
        shown = 0
        for concept_id, resource_list in resources.items():
            for resource in resource_list[:2]:  # Show max 2 per concept
                if shown < 3:  # Show max 3 total
                    print(f"    [{Colors.BLUE}{resource['resource_type']}{Colors.END}] "
                          f"{resource['description'][:80]}...")
                    shown += 1
        print()

    # Examples
    examples = subgraph.get('examples', {})
    if examples:
        total_examples = sum(len(e) for e in examples.values())
        print(f"{Colors.BOLD}Worked Examples:{Colors.END} {total_examples} examples found")
        print()

    print_subheader("Stage 3: Scaffolding (Simple → Complex Ordering)")

    scaffolded_path = result.get('scaffolded_path', [])
    if scaffolded_path:
        print(f"Learning path with {Colors.BOLD}{len(scaffolded_path)} concepts{Colors.END} "
              f"ordered by difficulty and prerequisites:\n")

        for i, concept in enumerate(scaffolded_path, 1):
            difficulty = concept['difficulty']

            # Color code by difficulty
            if difficulty == 'easy':
                diff_color = Colors.GREEN
                diff_icon = '●'
            elif difficulty == 'medium':
                diff_color = Colors.YELLOW
                diff_icon = '●●'
            else:
                diff_color = Colors.RED
                diff_icon = '●●●'

            print(f"{i}. {Colors.BOLD}{concept['title']}{Colors.END}")
            print(f"   {diff_color}{diff_icon} {difficulty.upper()}{Colors.END}")
            if concept['definitions']:
                print(f"   {concept['definitions'][0][:120]}...")
            print()
    else:
        print_warning("No scaffolded path generated")


def display_answer_with_metadata(result: Dict[str, Any]):
    """Display final answer w full metadata"""
    print_subheader("Stage 4: Answer Generation (LLM with Context)")

    print(f"{Colors.BOLD}Question:{Colors.END} {result['question']}\n")

    if result.get('graph_metadata'):
        meta = result['graph_metadata']
        print(f"{Colors.BOLD}Retrieval Metrics:{Colors.END}")
        print(f"  • Concepts used: {meta.get('subgraph_nodes', 0)} nodes")
        print(f"  • Relationships: {meta.get('subgraph_edges', 0)} edges")
        print(f"  • Scaffolding depth: {result.get('scaffolded_path_length', 0)} steps")
        print()

    print(f"{Colors.BOLD}Generated Answer:{Colors.END}")
    print(f"{Colors.CYAN}{'─'*80}{Colors.END}")
    print(result['answer'])
    print(f"{Colors.CYAN}{'─'*80}{Colors.END}")

    if result.get('graph_metadata'):
        meta = result['graph_metadata']

        if meta.get('scaffolded_path'):
            print(f"\n{Colors.BOLD}Learning Path Used:{Colors.END}")
            for i, concept in enumerate(meta['scaffolded_path'], 1):
                print(f"  {i}. {concept['title']} [{concept['difficulty']}]")


def test_single_query(question: str, detailed: bool = True):
    """Test GraphRAG with a single question"""
    print_header(f"Testing GraphRAG Pipeline")
    print(f"{Colors.BOLD}Query:{Colors.END} {question}\n")

    # Stage 1-3: Retrieval
    print_info("Running GraphRAG retrieval...")
    retrieval_result = graphrag_retrieve(
        question,
        top_k_concepts=5,
        min_similarity=0.2
    )

    if "error" in retrieval_result:
        print_error(f"Retrieval failed: {retrieval_result['error']}")
        return None

    if detailed:
        display_retrieval_results(retrieval_result)

    # Stage 4: Answer generation
    print_info("Generating scaffolded answer with LLM...")
    answer_result = answer_with_graphrag(
        question,
        top_k_concepts=5,
        min_similarity=0.2,
        include_metadata=True
    )

    display_answer_with_metadata(answer_result)

    return answer_result


def test_multiple_queries():
    """Test with multiple predefined queries"""
    print_header("GraphRAG Multi-Query Test Suite")

    test_queries = [
        "What is attention in transformers?",
        "Explain gradient descent",
        "What are applications of CLIP?"
    ]

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n{Colors.BOLD}[Test {i}/{len(test_queries)}]{Colors.END}")
        result = test_single_query(query, detailed=False)

        if result:
            results.append(result)
            print_success(f"Test {i} completed")
        else:
            print_error(f"Test {i} failed")

        if i < len(test_queries):
            print(f"\n{Colors.CYAN}{'─'*80}{Colors.END}\n")

    # Summary
    print_header("Test Summary")
    print(f"Completed {len(results)}/{len(test_queries)} tests\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['question']}")
        print(f"   Answer length: {len(result['answer'])} chars")
        print(f"   Concepts: {result['num_concepts']}, Scaffolding: {result['scaffolded_path_length']} steps")
        print()


def interactive_test():
    """Interactive testing mode"""
    print_header("GraphRAG Interactive Testing")
    print("Enter questions to test the GraphRAG system.")
    print("Commands: 'quit' to exit, 'stats' for graph statistics")
    print(f"{Colors.CYAN}{'─'*80}{Colors.END}\n")

    while True:
        try:
            question = input(f"{Colors.BOLD}Question: {Colors.END}").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.GREEN}Testing complete!{Colors.END}\n")
                break

            if question.lower() == 'stats':
                display_graph_stats()
                continue

            if not question:
                continue

            test_single_query(question, detailed=True)
            print(f"\n{Colors.CYAN}{'─'*80}{Colors.END}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}Testing complete!{Colors.END}\n")
            break
        except Exception as e:
            print_error(f"Error: {e}")
            import traceback
            traceback.print_exc()


def save_test_results(results: Dict[str, Any], filename: str = "graphrag_test_results.json"):
    """Save test results to file"""
    output_path = f"/app/data/{filename}"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_success(f"Results saved to {output_path}")


def main():
    import sys

    print_header("GraphRAG Test Suite")
    print(f"{Colors.BOLD}Erica AI Tutor - Knowledge Graph Testing{Colors.END}\n")

    # Check graph exists
    if not display_graph_stats():
        print_error("\nCannot proceed without knowledge graph!")
        print_info("Run these commands first:")
        print("  1. python -m graph.extract_entities")
        print("  2. python -m graph.extract_relationships")
        print("  3. python -m graph.build_graph")
        return

    print(f"\n{Colors.CYAN}{'─'*80}{Colors.END}\n")

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            interactive_test()
        elif sys.argv[1] == "--multi" or sys.argv[1] == "-m":
            test_multiple_queries()
        elif sys.argv[1] == "--demo":
            # Test with demonstration questions
            demo_questions = [
                "Explain attention mechanisms in transformers",
                "What are applications of CLIP?",
                "Explain variational bounds and Jensen's inequality"
            ]
            for question in demo_questions:
                test_single_query(question, detailed=True)
                print(f"\n{Colors.CYAN}{'='*80}{Colors.END}\n")
        else:
            # Treat as single query
            question = " ".join(sys.argv[1:])
            test_single_query(question, detailed=True)
    else:
        # Default: single test query
        default_query = "What is attention in transformers?"
        print_info(f"Running default test query: '{default_query}'")
        print_info("Use --interactive for interactive mode, --multi for multiple tests")
        print(f"\n{Colors.CYAN}{'─'*80}{Colors.END}\n")
        test_single_query(default_query, detailed=True)


if __name__ == "__main__":
    main()
