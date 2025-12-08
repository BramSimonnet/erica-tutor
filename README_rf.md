# Erica AI Tutor - GraphRAG Knowledge System

An intelligent tutoring system that uses GraphRAG (Graph-based Retrieval Augmented Generation) to provide scaffolded answers with code examples and resource citations.

## Project Overview

This system implements a complete GraphRAG pipeline for the Introduction to AI course, featuring:
- Multi-source content ingestion (PDFs, web pages, YouTube videos)
- Knowledge graph construction with concepts, resources, and examples
- Graph-based retrieval (not just vector similarity)
- Scaffolded answer generation (simple → complex)
- Automatic code example generation
- Resource citations and worked examples

## Architecture

```
┌─────────────┐
│  Raw Docs   │ (PDFs, Web, YouTube)
└──────┬──────┘
       │ Ingestion
       ▼
┌─────────────┐
│   Chunks    │ (~1200 chars)
└──────┬──────┘
       │ Parallel Processing
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│Embeddings│   │ Entities │   │  Edges   │
│(384-dim) │   │ (LLM)    │   │  (LLM)   │
└──────────┘   └─────┬────┘   └─────┬────┘
                     │              │
                     └──────┬───────┘
                            ▼
                   ┌─────────────────┐
                   │ Knowledge Graph │
                   │   (NetworkX)    │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  GraphRAG Query │
                   │   + Scaffolding │
                   └─────────────────┘
```

## Prerequisites

- **Docker Desktop** (required for containerization)
- **LM Studio** (required for local LLM)
  - Download from: https://lmstudio.ai/
  - Model: Qwen2.5-7B-Instruct (or similar)
  - Must be running on `localhost:1234`

## Quick Start

### 1. Start Docker Services

```bash
# Clone the repository
git clone <your-repo-url>
cd erica-tutor

# Start MongoDB and backend containers
docker-compose up -d

# Verify containers are running
docker ps
```

You should see:
- `erica-backend` (running on port 8000)
- `erica-mongo` (running on port 27017)

### 2. Start LM Studio

1. Launch LM Studio
2. Download and load **Qwen2.5-7B-Instruct** (or similar model)
3. Start the local server on port `1234`
4. Verify it's running: `curl http://localhost:1234/v1/models`

### 3. Run the Complete Pipeline

```bash
# Enter the backend container
docker exec -it erica-backend bash

# Step 1: Ingest content (web pages, PDFs)
python -m ingestion.ingest_web
python -m ingestion.ingest_pdf

# Step 2: Convert to chunks
python -m ingestion.raw_to_chunks

# Step 3: Generate embeddings
python -m vectorstore.embed_chunks

# Step 4: Extract entities (concepts, resources, examples)
python -m graph.extract_entities

# Step 5: Extract relationships (prerequisite, related, explains)
python -m graph.extract_relationships

# Step 6: Build knowledge graph
python -m graph.build_graph

# Step 7: Detect communities
python -m graph.communities

# Exit container
exit
```

**Or run everything at once:**

```bash
docker exec erica-backend bash -c "
python -m ingestion.ingest_web &&
python -m ingestion.ingest_pdf &&
python -m ingestion.raw_to_chunks &&
python -m vectorstore.embed_chunks &&
python -m graph.extract_entities &&
python -m graph.extract_relationships &&
python -m graph.build_graph &&
python -m graph.communities &&
echo '✓ Pipeline complete!'
"
```

### 4. Generate Assignment Deliverable

```bash
# Generate the complete assignment report
docker exec erica-backend python -m generate_assignment_report
```

This creates: `data/ASSIGNMENT_DELIVERABLE.md` containing:
- System prompts used
- Knowledge graph nodes/edges retrieved
- Scaffolded explanations with code examples
- Resource citations
- Complete output for all 3 demonstration questions

### 5. Create Visualizations

```bash
# Generate knowledge graph visualizations
docker exec erica-backend python -m visualize_knowledge_graph
```

This creates 11+ visualizations in `data/visualizations/`:
- `graph_overview.png` - Full graph (216 nodes, 157 edges)
- `concept_*.png` - Top concepts by centrality
- `demo_attention_*.png` - Attention mechanisms
- `demo_clip_*.png` - CLIP applications
- `demo_variational_*.png` - Variational bounds

## Testing the System

### Interactive Testing

```bash
# Run interactive GraphRAG testing
run_test.bat --interactive

# Or from inside the container:
docker exec -it erica-backend python -m test_graphrag --interactive
```

### Demo Questions

```bash
# Test all 3 required demonstration questions
run_test.bat --demo

# Questions tested:
# 1. Explain attention mechanisms in transformers
# 2. What are applications of CLIP?
# 3. Explain variational bounds and Jensen's inequality
```

### Single Query

```bash
# Test with a custom question
run_test.bat "What is backpropagation?"

# Or:
docker exec -it erica-backend python -m test_graphrag "What is backpropagation?"
```

## Project Structure

```
erica-tutor/
├── backend/
│   ├── app.py                          # FastAPI application
│   ├── ingestion/
│   │   ├── ingest_pdf.py              # PDF ingestion
│   │   ├── ingest_web.py              # Web scraping
│   │   ├── ingest_youtube.py          # YouTube transcript extraction
│   │   └── raw_to_chunks.py           # Chunking strategy
│   ├── vectorstore/
│   │   ├── embeddings.py              # Sentence-transformers embeddings
│   │   ├── storage.py                 # MongoDB vector storage
│   │   └── embed_chunks.py            # Batch embedding generation
│   ├── graph/
│   │   ├── extract_entities.py        # LLM-based entity extraction
│   │   ├── extract_relationships.py   # LLM-based relationship extraction
│   │   ├── build_graph.py             # NetworkX graph construction
│   │   ├── communities.py             # Louvain community detection
│   │   ├── graphrag_retrieval.py      # Graph traversal retrieval
│   │   └── graphrag_query.py          # Scaffolded answer generation
│   ├── llm/
│   │   └── qwenclient.py              # LM Studio integration
│   ├── generate_assignment_report.py   # Report generator
│   ├── visualize_knowledge_graph.py    # Graph visualization
│   └── test_graphrag.py                # Testing framework
├── data/
│   ├── pdfs/                           # Source PDF documents
│   ├── visualizations/                 # Generated graph images
│   ├── knowledge_graph.pkl             # Serialized NetworkX graph
│   ├── communities.json                # Community detection results
│   └── ASSIGNMENT_DELIVERABLE.md       # Final report
├── docker-compose.yml                   # Docker orchestration
├── run_test.bat                         # Windows test runner
├── run_test.sh                          # Linux/Mac test runner
└── README.md                            # This file
```

## MongoDB Collections

The system uses MongoDB for persistent storage:

- **`raw_documents`**: Original ingested content
  - Fields: `id`, `type`, `raw_text`, `metadata`, `url`/`file_path`

- **`chunk_documents`**: Text chunks (~1200 chars)
  - Fields: `id`, `raw_document_id`, `chunk_index`, `text`, `metadata`

- **`chunk_embeddings`**: Vector embeddings (384-dim)
  - Fields: `chunk_id`, `embedding`, `metadata`

- **`entity_nodes`**: Knowledge graph nodes
  - Types: `concept`, `resource`, `example`
  - Concepts: `title`, `definitions`, `difficulty`, `aliases`
  - Resources: `resource_type`, `span`, `description`
  - Examples: `text`, `concepts`

- **`entity_relationships`**: Knowledge graph edges
  - Types: `prerequisite`, `related`, `explains`, `example_of`
  - Fields: `source_id`, `target_id`, `relationship_type`, `metadata`

## Knowledge Graph Structure

### Node Types

- **Concepts** (91 nodes)
  - Educational concepts with difficulty levels (easy/medium/hard)
  - Definitions, aliases, and prerequisite chains

- **Resources** (67 nodes)
  - PDFs, videos, web pages
  - Includes spans (page numbers, timestamps)

- **Examples** (58 nodes)
  - Worked examples demonstrating concepts
  - Code snippets and explanations

### Edge Types

- **prerequisite** (20 edges): Concept A → Concept B
  - Enables scaffolding from simple to complex

- **explains** (55 edges): Resource → Concept
  - Links resources to concepts they explain

- **related** (22 edges): Concept ↔ Concept
  - Near-transfer relationships

- **example_of** (60 edges): Example → Concept
  - Links worked examples to concepts

## Adding New Content

### Add PDFs

1. Place PDF files in `data/pdfs/`
2. Update `backend/ingestion/ingest_pdf.py` with the file path
3. Run: `docker exec erica-backend python -m ingestion.ingest_pdf`

### Add Web Pages

1. Update `backend/ingestion/ingest_web.py` with URLs
2. Run: `docker exec erica-backend python -m ingestion.ingest_web`

### Add YouTube Videos

1. Update `backend/ingestion/ingest_youtube.py` with video URLs
2. Run: `docker exec erica-backend python -m ingestion.ingest_youtube`

After adding new content, re-run the pipeline from Step 2 onwards.

## Customization

### Adjust Chunking Strategy

Edit `backend/ingestion/chunk.py`:
```python
max_chars = 1200  # Adjust chunk size
```

### Change Embedding Model

Edit `backend/vectorstore/embeddings.py`:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')  # Try different models
```

### Modify LLM Prompts

Edit `backend/graph/graphrag_query.py`:
```python
SCAFFOLDED_ANSWER_PROMPT = """..."""  # Customize prompt
```

### Adjust Retrieval Parameters

When querying:
```python
answer_with_graphrag(
    question,
    top_k_concepts=5,      # Number of concepts to retrieve
    min_similarity=0.2     # Minimum similarity threshold
)
```

## Troubleshooting

### "Knowledge graph not found"

Run the graph building pipeline:
```bash
docker exec erica-backend python -m graph.build_graph
```

### "No relevant concepts found"

- Lower the similarity threshold
- Check if entities were extracted: `docker exec erica-backend python -c "from pymongo import MongoClient; print(MongoClient('mongodb://mongo:27017').erica.entity_nodes.count_documents({}))"`
- Ensure embeddings exist

### LLM Connection Issues

- Verify LM Studio is running: `curl http://localhost:1234/v1/models`
- Check the model is loaded in LM Studio
- Restart LM Studio server

### Docker Issues

```bash
# Restart containers
docker-compose down
docker-compose up -d

# View logs
docker logs erica-backend
docker logs erica-mongo

# Rebuild containers
docker-compose up --build
```

### MongoDB Issues

```bash
# Connect to MongoDB shell
docker exec -it erica-mongo mongosh

# View collections
use erica
show collections
db.entity_nodes.countDocuments({})
```

## Performance Notes

- **Entity Extraction**: ~1-2 minutes per chunk (LLM-based)
- **Relationship Extraction**: ~1-2 minutes per chunk (LLM-based)
- **Graph Building**: < 1 second
- **Query Response**: ~2-5 seconds (includes LLM generation)

For faster processing:
- Use a faster LLM (smaller model)
- Reduce chunk count (larger chunks)
- Use GPU acceleration in LM Studio

## Development

### Run Backend Locally (without Docker)

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Add New Scripts

All pipeline scripts should be runnable as modules:
```bash
python -m your_module_name
```

Include a `__main__` block:
```python
if __name__ == "__main__":
    main()
```

## Assignment Deliverables

The system generates all required assignment components:

✅ **M1: Environment Setup**
- Docker Compose configuration
- MongoDB integration
- LLM integration (Qwen via LM Studio)

✅ **M2: Data Ingestion**
- 11 documents ingested (9 web + 2 PDFs)
- All URLs documented in database

✅ **M3: Knowledge Graph Construction**
- 216 nodes (91 concepts, 67 resources, 58 examples)
- 157 edges (20 prerequisites, 55 explains, 22 related, 60 example_of)
- NetworkX MultiDiGraph with community detection
- Proper difficulty levels and prerequisite chains

✅ **M4: Query and Generation**
- GraphRAG-based retrieval (graph traversal, not just vector)
- Scaffolded answer generation (simple → complex)
- Resource citations with page numbers/timestamps
- Code examples for technical questions

### Demonstration Questions

The system answers all 3 required questions:

1. **"Explain attention mechanisms in transformers"**
   - ✅ Scaffolded explanation
   - ✅ Python code examples showing attention computation
   - ✅ Resource citations

2. **"What are applications of CLIP?"**
   - ✅ Scaffolded explanation
   - ✅ Application examples
   - ✅ Resource citations

3. **"Explain variational bounds and Jensen's inequality"**
   - ✅ Scaffolded explanation
   - ✅ Mathematical context
   - ✅ Resource citations

## License

Academic project for Introduction to AI course.

## Contact

For issues or questions about this implementation, please refer to the assignment documentation at:
https://pantelis.github.io/aiml-common/projects/nlp/ai-tutor/
