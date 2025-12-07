# GraphRAG Test - Example Output

This file shows what the output looks like when running `python -m test_graphrag`

## Running the Test

```bash
# Start Docker services
docker-compose up -d

# Enter backend container
docker exec -it erica-backend bash

# Run the test
python -m test_graphrag
```

---

## Expected Output

```
================================================================================
                            GraphRAG Test Suite
================================================================================

Erica AI Tutor - Knowledge Graph Testing

▶ Knowledge Graph Statistics
────────────────────────────────────────────────────────────────────────────────
Nodes:
  • Concepts: 45
  • Resources: 23
  • Examples: 12
  • Total: 80

Edges:
  • prerequisite: 38
  • explains: 56
  • related: 24
  • example_of: 18
  • Total: 136

Concept Difficulty Distribution:
  • Easy: 15
  • Medium: 20
  • Hard: 10

✓ Graph loaded successfully with 80 nodes and 136 edges

────────────────────────────────────────────────────────────────────────────────

ℹ Running default test query: 'What is attention in transformers?'
ℹ Use --interactive for interactive mode, --multi for multiple tests

────────────────────────────────────────────────────────────────────────────────

================================================================================
                         Testing GraphRAG Pipeline
================================================================================

Query: What is attention in transformers?

▶ Stage 1: Concept Identification (Vector Similarity)
────────────────────────────────────────────────────────────────────────────────
Found 5 relevant concepts:

1. Attention Mechanism
   Similarity: 0.8542
   Difficulty: medium
   Degree: 12 (centrality in graph)
   Definition: A technique that allows neural networks to focus on specific parts of the input when producing output...

2. Self-Attention
   Similarity: 0.7821
   Difficulty: medium
   Degree: 8 (centrality in graph)
   Definition: A type of attention mechanism where a sequence attends to itself to compute representations...

3. Transformer Architecture
   Similarity: 0.7234
   Difficulty: hard
   Degree: 15 (centrality in graph)
   Definition: A neural network architecture based entirely on attention mechanisms, dispensing with recurrence...

4. Multi-Head Attention
   Similarity: 0.6891
   Difficulty: hard
   Degree: 7 (centrality in graph)
   Definition: An extension of attention that allows the model to jointly attend to information from different...

5. Query-Key-Value Mechanism
   Similarity: 0.6234
   Difficulty: medium
   Degree: 5 (centrality in graph)
   Definition: The core computation in attention where queries are matched against keys to determine values...

▶ Stage 2: Subgraph Retrieval (Graph Traversal)
────────────────────────────────────────────────────────────────────────────────
Subgraph size: 18 nodes, 35 edges

Prerequisites Found:
  • 3 prerequisite(s) for concept
  • 2 prerequisite(s) for concept

Related Concepts: 4 near-transfer concepts found

Resources Found:
  • 8 resources explaining concepts
    [pdf] Attention Is All You Need (Vaswani et al., 2017), pages 3-5...
    [video] The Illustrated Transformer by Jay Alammar, timestamp 08:45...
    [web] Understanding Attention Mechanisms - Towards Data Science...

Worked Examples: 3 examples found

▶ Stage 3: Scaffolding (Simple → Complex Ordering)
────────────────────────────────────────────────────────────────────────────────
Learning path with 8 concepts ordered by difficulty and prerequisites:

1. Neural Networks
   ● EASY
   Neural networks are computing systems inspired by biological neural networks that learn from data...

2. Sequence Modeling
   ● EASY
   The task of predicting the next item in a sequence based on previous items...

3. Embedding Representations
   ●● MEDIUM
   Dense vector representations that capture semantic meaning of discrete inputs...

4. Dot Product Similarity
   ●● MEDIUM
   A measure of similarity between vectors computed as the sum of element-wise products...

5. Query-Key-Value Mechanism
   ●● MEDIUM
   The core computation in attention where queries are matched against keys to determine values...

6. Attention Mechanism
   ●● MEDIUM
   A technique that allows neural networks to focus on specific parts of the input when producing output...

7. Self-Attention
   ●● MEDIUM
   A type of attention mechanism where a sequence attends to itself to compute representations...

8. Multi-Head Attention
   ●●● HARD
   An extension of attention that allows the model to jointly attend to information from different...

ℹ Generating scaffolded answer with LLM...

▶ Stage 4: Answer Generation (LLM with Context)
────────────────────────────────────────────────────────────────────────────────
Question: What is attention in transformers?

Retrieval Metrics:
  • Concepts used: 18 nodes
  • Relationships: 35 edges
  • Scaffolding depth: 8 steps

Generated Answer:
────────────────────────────────────────────────────────────────────────────────
Let me explain attention mechanisms in transformers by building from foundational
concepts to more advanced ones.

**Starting with the Basics**

First, understand that neural networks process information in layers, transforming
input data into useful representations [Neural Networks, easy]. When working with
sequences like text or time series, we need models designed for sequence modeling
that can capture dependencies between elements [Sequence Modeling, easy].

**The Core Idea: Query-Key-Value**

Attention mechanisms work through a query-key-value (QKV) system [PDF, page 3].
Think of it like searching a library:
- The **query** is your search question
- The **keys** are the catalog entries for each book
- The **values** are the actual books

The mechanism computes how relevant each key is to your query using dot product
similarity [Dot Product Similarity, medium], then uses these relevance scores to
create a weighted combination of the values.

**Self-Attention: The Heart of Transformers**

In transformers, self-attention means the sequence attends to itself [Video, 08:45].
Each position in the input sequence creates a query, and compares it against keys
from all positions (including itself) to determine which positions are most relevant.
This allows the model to capture relationships between words regardless of their
distance in the sequence.

For example, in "The animal didn't cross the street because it was too tired,"
self-attention helps the model understand that "it" refers to "animal," not "street."

**Multi-Head Attention: Parallel Processing**

To capture different types of relationships, transformers use multi-head attention
[PDF, page 4]. Instead of one attention operation, they run multiple attention
mechanisms in parallel, each potentially focusing on different aspects:
- One head might focus on syntactic relationships
- Another on semantic similarity
- Another on positional dependencies

These multiple "heads" are then combined to create a richer representation
[Worked Example: Transformer encoder processing "Hello world"].

**Why This Matters**

Unlike recurrent models that process sequences step-by-step, attention allows
transformers to:
1. Process all positions simultaneously (parallelization)
2. Capture long-range dependencies directly
3. Learn which parts of the input are relevant for each output

This architecture has proven highly effective for tasks like machine translation,
text generation, and language understanding [Web: Understanding Attention Mechanisms].

**Resources for Further Learning**
- Original paper: "Attention Is All You Need" (Vaswani et al., 2017) [PDF, pages 3-5]
- Visual explanation: "The Illustrated Transformer" by Jay Alammar [Video, 08:45-12:30]
────────────────────────────────────────────────────────────────────────────────

Learning Path Used:
  1. Neural Networks [easy]
  2. Sequence Modeling [easy]
  3. Embedding Representations [medium]
  4. Dot Product Similarity [medium]
  5. Query-Key-Value Mechanism [medium]
  6. Attention Mechanism [medium]
  7. Self-Attention [medium]
  8. Multi-Head Attention [hard]
```

---

## Other Testing Modes

### Interactive Mode
```bash
python -m test_graphrag --interactive

# Example session:
Question: What is gradient descent?
[Full GraphRAG pipeline output...]

Question: stats
[Shows graph statistics...]

Question: Explain backpropagation
[Full GraphRAG pipeline output...]

Question: quit
Testing complete!
```

### Multi-Query Test
```bash
python -m test_graphrag --multi

# Tests multiple queries in sequence:
# - "What is attention in transformers?"
# - "Explain gradient descent"
# - "What are applications of CLIP?"
```

### Demonstration Questions
```bash
python -m test_graphrag --demo

# Tests all 3 assignment demonstration questions:
# - "Explain attention mechanisms in transformers"
# - "What are applications of CLIP?"
# - "Explain variational bounds and Jensen's inequality"
```

### Custom Single Query
```bash
python -m test_graphrag "Your custom question here"
```

---

## What Each Stage Shows

### Stage 1: Concept Identification
- Uses vector embeddings to find concepts similar to the query
- Scores are boosted by graph centrality (well-connected concepts rank higher)
- Shows similarity score, difficulty level, and graph degree

### Stage 2: Subgraph Retrieval
- Traverses the graph starting from relevant concepts
- Collects prerequisites (for scaffolding)
- Finds related concepts (near-transfer learning)
- Gathers resources (PDFs, videos, web pages with specific spans)
- Includes worked examples

### Stage 3: Scaffolding
- Orders concepts from simple → complex
- Uses prerequisite chains and difficulty levels
- Creates a progressive learning path

### Stage 4: Answer Generation
- LLM receives the scaffolded context
- Generates explanation following the learning path
- Includes citations to specific resources
- Builds understanding step-by-step

---

## Next Steps

1. **Start Docker services**: `docker-compose up -d`
2. **Enter container**: `docker exec -it erica-backend bash`
3. **Run test**: `python -m test_graphrag --interactive`
4. **Try different queries** to see how GraphRAG retrieves and scaffolds information
5. **Check graph statistics** with the `stats` command in interactive mode
