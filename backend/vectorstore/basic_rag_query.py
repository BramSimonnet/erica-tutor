"""
Basic RAG (Retrieval Augmented Generation) query module.
Combines vector search with LLM to answer questions based on ingested content.

NOTE: This is a BASIC RAG implementation using only vector similarity.
The assignment requires GRAPHRAG (graph/graphrag_query.py) which uses
knowledge graph traversal for retrieval. Use this file only for comparison
or simple vector-based retrieval tasks.

For the assignment, use: graph.graphrag_query instead.
"""
from typing import Dict, Any
from vectorstore.retrieval import retrieve_context_for_query
from llm.qwenclient import call_llm


RAG_PROMPT_TEMPLATE = """You are a helpful AI tutor. Answer the student's question using ONLY the information provided in the context below.

If the context doesn't contain enough information to answer the question, say so honestly.

Context:
{context}

Student Question: {question}

Answer:"""


def answer_question(
    question: str,
    top_k: int = 5,
    min_similarity: float = 0.3,
    include_sources: bool = True
) -> Dict[str, Any]:
    """
    Answer a question using RAG (Retrieval Augmented Generation).

    1. Retrieve relevant chunks from vector store
    2. Build context from retrieved chunks
    3. Send to LLM with context
    4. Return answer with sources

    Args:
        question: Student's question
        top_k: Number of relevant chunks to retrieve
        min_similarity: Minimum similarity threshold for retrieval
        include_sources: Whether to include source information in response

    Returns:
        Dict with:
            - question: original question
            - answer: LLM's answer
            - sources: list of source chunks used (if include_sources=True)
            - num_sources: number of sources retrieved
    """
    # Step 1: Retrieve relevant context
    retrieval_result = retrieve_context_for_query(
        question,
        top_k=top_k,
        min_similarity=min_similarity
    )

    if retrieval_result["num_results"] == 0:
        return {
            "question": question,
            "answer": "I couldn't find any relevant information in the knowledge base to answer this question.",
            "sources": [],
            "num_sources": 0
        }

    # Step 2: Build prompt with context
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=retrieval_result["context_text"],
        question=question
    )

    # Step 3: Get answer from LLM
    answer = call_llm(prompt)

    # Step 4: Prepare response
    response = {
        "question": question,
        "answer": answer,
        "num_sources": retrieval_result["num_results"]
    }

    if include_sources:
        response["sources"] = [
            {
                "text": r["chunk_data"]["text"][:200] + "...",  # Truncate for brevity
                "source_type": r["chunk_data"]["source_type"],
                "similarity": r["similarity"]
            }
            for r in retrieval_result["results"]
        ]

    return response


def interactive_tutor():
    """
    Interactive CLI for asking questions to the AI tutor.
    """
    print("=" * 60)
    print("AI Tutor - Interactive Mode")
    print("=" * 60)
    print("Ask questions about the course material.")
    print("Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            question = input("\n🎓 Your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! Happy learning! 📚")
                break

            if not question:
                continue

            print("\n🤔 Thinking...")

            result = answer_question(question, top_k=3, min_similarity=0.2)

            print(f"\n💡 Answer:\n{result['answer']}\n")

            if result['num_sources'] > 0:
                print(f"📚 Based on {result['num_sources']} source(s)")
                if result.get('sources'):
                    print("\nSources:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. [{source['source_type']}] (similarity: {source['similarity']:.2f})")
                        print(f"     {source['text']}")
            else:
                print("⚠️  No relevant sources found")

        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy learning! 📚")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    interactive_tutor()
