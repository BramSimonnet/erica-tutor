# GraphRAG Testing Guide

## Overview

The `test_graphrag.py` script provides comprehensive testing for the GraphRAG pipeline with detailed visual output showing each stage of processing.

## Features

- **Color-coded terminal output** for easy reading
- **Stage-by-stage visualization** of the GraphRAG pipeline:
  1. Concept Identification (vector similarity)
  2. Subgraph Retrieval (graph traversal)
  3. Scaffolding (simple → complex ordering)
  4. Answer Generation (LLM with context)
- **Graph statistics** display
- **Multiple testing modes** (single query, multi-query, interactive, demo)

## Running the Tests

### From Docker Container

```bash
# Enter the backend container
docker exec -it erica-backend bash

# Run with default test query
python -m test_graphrag

# Interactive mode (ask multiple questions)
python -m test_graphrag --interactive
python -m test_graphrag -i

# Multi-query test suite
python -m test_graphrag --multi
python -m test_graphrag -m

# Test with assignment demonstration questions
python -m test_graphrag --demo

# Single custom query
python -m test_graphrag "What is attention in transformers?"
```

### Prerequisites

The knowledge graph must be built before testing:

```bash
# 1. Extract entities from chunks
python -m graph.extract_entities

# 2. Extract relationships between entities
python -m graph.extract_relationships

# 3. Build the NetworkX graph
python -m graph.build_graph

# 4. (Optional) Detect communities
python -m graph.communities
```

## Output Explanation

### Graph Statistics

Shows the current state of the knowledge graph:
- Node counts by type (concepts, resources, examples)
- Edge counts by relationship type
- Concept difficulty distribution

### Stage 1: Concept Identification

Displays concepts found via vector similarity:
- **Similarity score** (0-1): How well concept matches query
- **Difficulty**: easy/medium/hard
- **Degree**: Graph centrality (how connected the concept is)

### Stage 2: Subgraph Retrieval

Shows graph traversal results:
- **Prerequisites**: Concepts needed to understand target concepts
- **Related concepts**: Near-transfer relationships
- **Resources**: PDFs, videos, web pages explaining concepts
- **Examples**: Worked examples demonstrating concepts

### Stage 3: Scaffolding

Learning path ordered simple → complex:
- Concepts organized by difficulty and prerequisite chains
- Enables progressive understanding

### Stage 4: Answer Generation

Final LLM-generated answer with:
- Full scaffolded explanation
- Resource citations
- Metadata about retrieval

## Understanding the Color Codes

- 🟢 **Green**: Success, high similarity, easy difficulty
- 🟡 **Yellow**: Warning, medium similarity, medium difficulty
- 🔴 **Red**: Error, low similarity, hard difficulty
- 🔵 **Blue**: Information, resources
- 🟣 **Purple**: Headers

## Testing Strategies

### Quick Test
```bash
python -m test_graphrag "What is CLIP?"
```

### Thorough Testing
```bash
python -m test_graphrag --multi
```

### Development/Debugging
```bash
python -m test_graphrag --interactive
# Then type: stats
# Ask questions interactively
```

### Assignment Deliverables
```bash
python -m test_graphrag --demo
# Tests all 3 required demonstration questions
```

## Troubleshooting

### "Knowledge graph not found"

Run the graph build pipeline:
```bash
python -m graph.build_graph
```

### "No relevant concepts found"

- Check if entities were extracted: look in MongoDB `entity_nodes` collection
- Lower the similarity threshold in the code
- Ensure embeddings are generated for chunks

### LLM Connection Issues

- Ensure LM Studio is running on host machine at `localhost:1234`
- Check `llm/qwenclient.py` configuration
- Test LLM connection: `curl http://localhost:1234/v1/models`

## Example Output

```
================================================================================
                         Testing GraphRAG Pipeline
================================================================================

Query: What is attention in transformers?

▶ Stage 1: Concept Identification (Vector Similarity)
────────────────────────────────────────────────────────────────────────────────
Found 3 relevant concepts:

1. Attention Mechanism
   Similarity: 0.8542
   Difficulty: medium
   Degree: 12 (centrality in graph)
   Definition: A technique that allows models to focus on specific parts...

▶ Stage 2: Subgraph Retrieval (Graph Traversal)
────────────────────────────────────────────────────────────────────────────────
Subgraph size: 8 nodes, 15 edges

Prerequisites Found:
  • 2 prerequisite(s) for concept

Resources Found:
  • 5 resources explaining concepts
    [pdf] Attention Is All You Need paper...
    [video] Illustrated Transformer tutorial...

▶ Stage 3: Scaffolding (Simple → Complex Ordering)
────────────────────────────────────────────────────────────────────────────────
Learning path with 4 concepts ordered by difficulty and prerequisites:

1. Neural Networks
   ● EASY
   Neural networks are computing systems inspired by biological...

2. Sequence Models
   ●● MEDIUM
   Models designed to process sequential data...

3. Attention Mechanism
   ●● MEDIUM
   A technique that allows models to focus on specific parts...

4. Transformer Architecture
   ●●● HARD
   A neural network architecture based entirely on attention...

▶ Stage 4: Answer Generation (LLM with Context)
────────────────────────────────────────────────────────────────────────────────
Question: What is attention in transformers?

Generated Answer:
────────────────────────────────────────────────────────────────────────────────
[Scaffolded explanation with citations appears here]
────────────────────────────────────────────────────────────────────────────────
```

## Customization

Edit `test_graphrag.py` to modify:
- Default queries in `test_multiple_queries()`
- Number of concepts retrieved (`top_k_concepts`)
- Similarity threshold (`min_similarity`)
- Output formatting and colors
