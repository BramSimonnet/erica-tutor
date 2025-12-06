# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Erica is an AI-powered tutor system that ingests educational content from multiple sources (PDFs, web pages, YouTube videos), processes them into chunks, extracts structured knowledge entities, and stores them in MongoDB for retrieval and tutoring.

## Assignment Requirements

This project implements a GraphRAG-based intelligent tutoring system for the Introduction to AI course. Full assignment details: https://pantelis.github.io/aiml-common/projects/nlp/ai-tutor/

### Project Milestones

**M1: Environment Setup**
- Docker compose configuration for development environment
- Status: ✓ Complete (docker-compose.yml with backend + MongoDB)

**M2: Data Ingestion**
- Ingest course materials: website content, YouTube videos, slides/PDFs
- Document all ingested URLs in database
- Status: ✓ Complete (ingestion modules for PDF, web, YouTube)

**M3: Knowledge Graph Construction**
- Build knowledge graph with NetworkX
- Required node types:
  - **Concepts**: title, definitions, difficulty (easy/medium/hard), aliases
  - **Resources**: type (pdf/slide/web/video), span (page/section/timestamp), description
  - **Worked Examples**: text, related concepts
- Required edge types:
  - Prerequisite relationships between concepts
  - Resource-to-concept explanations
  - Example-to-concept links
  - "Near transfer" relationships (related concepts)
- Community clustering for concept organization
- Status: Partially implemented (entity extraction done, graph construction needed)

**M4: Query and Generation**
- Convert user queries to concept candidates
- Retrieve relevant subgraphs from knowledge graph
- Generate scaffolded answers (simple → complex)
- Include resource citations in responses
- Status: RAG implementation complete, graph-based retrieval needed

### Required Deliverables

Three demonstration questions with complete system outputs:
1. "Explain attention mechanisms in transformers"
2. "What are applications of CLIP?"
3. "Explain variational bounds and Jensen's inequality"

Each answer must include:
- System prompts used
- Knowledge graph nodes/edges retrieved
- Generated explanation with scaffolding
- Resource references with citations

### Technical Requirements

**LLM Configuration**
- Local LLM via Ollama or LM Studio (Qwen2.5 preferred)
- Remote API fallback via OpenRouter for hardware limitations
- Current implementation: LM Studio at `host.docker.internal:1234`

**GraphRAG vs Basic RAG**
- The assignment requires knowledge graph-based retrieval, not just vector similarity
- Current implementation has vector-based RAG; needs graph traversal integration
- Graph should support community detection and prerequisite chains

## Architecture

The system follows a **GraphRAG pipeline** with six stages:

1. **Ingestion**: Raw content is extracted from various sources and stored in MongoDB's `raw_documents` collection
2. **Chunking**: Raw documents are split into manageable chunks (~1200 chars) and stored in `chunk_documents` collection
3. **Embedding**: Chunks are converted to dense vector embeddings (384-dimensional) using sentence-transformers and stored in `chunk_embeddings` collection
4. **Entity Extraction**: LLM analyzes each chunk to extract structured entities (concepts, resources, examples) stored in `entity_nodes` collection
5. **Relationship Extraction**: LLM identifies relationships between entities (prerequisites, related concepts, explanations) stored in `entity_relationships` collection
6. **Graph Construction**: NetworkX builds a directed multigraph enabling graph-based retrieval and community detection

**GraphRAG vs Basic RAG**: The system uses knowledge graph traversal (not just vector similarity) for retrieval. This enables prerequisite-aware scaffolding, community-based organization, and citation of specific resources.

### MongoDB Collections

- `raw_documents`: Original text from PDFs, web pages, or YouTube transcripts
  - Fields: `id`, `type` (pdf/web/youtube), `raw_text`, `metadata`, `file_path`/`url`
- `chunk_documents`: Text chunks from raw documents
  - Fields: `id`, `raw_document_id`, `chunk_index`, `text`, `source_type`, `metadata`
- `chunk_embeddings`: Vector embeddings for semantic search
  - Fields: `chunk_id`, `embedding` (384-dim float array), `metadata`
  - Uses all-MiniLM-L6-v2 model from sentence-transformers
- `entity_nodes`: Extracted knowledge entities
  - Fields: `id`, `type` (concept/resource/example), `source_chunk`, plus type-specific fields
  - Concepts: `title`, `definitions`, `difficulty`, `aliases`
  - Resources: `resource_type`, `span`, `description`
  - Examples: `text`, `concepts` (list of concept titles)
- `entity_relationships`: Edges in the knowledge graph
  - Fields: `id`, `source_id`, `target_id`, `relationship_type`, `metadata`
  - Relationship types: `prerequisite`, `related`, `explains`, `example_of`

### LLM Integration

The system uses a local Qwen2.5-7B-Instruct model running in LM Studio, accessed via `llm/qwenclient.py`. The LM Studio server must be running at `http://localhost:1234` on the host machine (accessed as `host.docker.internal:1234` from containers).

Entity extraction uses a structured prompt in `graph/extract_entities.py` that enforces JSON output with specific schema for concepts, resources, and examples.

## Development Commands

### Docker Environment

Start services (backend + MongoDB):
```bash
docker-compose up --build
```

Stop services:
```bash
docker-compose down
```

The backend runs on `http://localhost:8000` and MongoDB on `localhost:27017`.

### Running Pipeline Scripts

All pipeline scripts are designed to be run as standalone modules from the backend container:

```bash
# Enter backend container
docker exec -it erica-backend bash

# Ingest content
python -m ingestion.ingest_pdf
python -m ingestion.ingest_web
python -m ingestion.ingest_youtube

# Process raw documents into chunks
python -m ingestion.raw_to_chunks

# Generate embeddings for vector search (run after chunking)
python -m vectorstore.embed_chunks

# Extract entities from chunks (concepts, resources, examples)
python -m graph.extract_entities

# Extract relationships between entities (prerequisites, related, etc.)
python -m graph.extract_relationships

# Build the knowledge graph with NetworkX
python -m graph.build_graph

# Detect communities in the knowledge graph
python -m graph.communities

# Interactive GraphRAG tutor (uses graph traversal + scaffolding)
python -m graph.graphrag_query

# Generate demonstration answers for assignment deliverables
python -m graph.graphrag_query --demo

# Basic vector RAG (for comparison only, not for assignment)
python -m vectorstore.basic_rag_query
```

Each script has a `__main__` block with example usage that should be updated with actual content sources.

### Backend API

The FastAPI app in `app.py` is minimal - currently only has a health check endpoint. To add new endpoints, import and use the `app` FastAPI instance.

Start backend locally (without Docker):
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Key Implementation Details

### Chunking Strategy

`ingestion/chunk.py` uses paragraph-based chunking with a max_chars limit (default 1200). Paragraphs are concatenated until the limit is reached, then a new chunk starts. This preserves semantic boundaries.

### LLM Response Parsing

`graph/extract_entities.py` includes robust JSON parsing in `_parse_llm_json()` that handles:
- Code fence removal (```json blocks)
- Extraction of JSON from surrounding text
- Validation of required structure

This is necessary because LLMs sometimes add explanatory text or formatting despite instructions.

### MongoDB Connection

MongoDB is accessed directly via pymongo. The connection string is hardcoded to `mongodb://mongo:27017` (the Docker service name). For local development outside Docker, update to `mongodb://localhost:27017`.

Database name: `erica`

## Knowledge Graph & GraphRAG

### Graph Module Structure

The `graph/` module implements the GraphRAG system:

- `extract_entities.py`: LLM-based extraction of concepts, resources, and examples from chunks
- `extract_relationships.py`: LLM-based extraction of relationships (prerequisites, related, explains, example_of)
- `build_graph.py`: NetworkX graph construction and management
- `communities.py`: Community detection using Louvain algorithm for topic clustering
- `graphrag_retrieval.py`: Graph traversal-based retrieval (hybrid: vector similarity + graph structure)
- `graphrag_query.py`: Scaffolded answer generation with resource citations

### Knowledge Graph Structure

**Graph Type**: NetworkX MultiDiGraph (directed, allows multiple edges between nodes)

**Nodes**:
- **Concepts**: Educational concepts with difficulty levels (easy/medium/hard), definitions, aliases
- **Resources**: Learning materials (PDF/slide/web/video) with spans (page numbers, timestamps)
- **Examples**: Worked examples demonstrating concepts

**Edges**:
- **prerequisite**: Concept A must be understood before Concept B (enables scaffolding)
- **related**: Near-transfer relationships between similar concepts
- **explains**: Resource/example explains a concept
- **example_of**: Example demonstrates a concept

### GraphRAG Retrieval Pipeline

The system uses **graph traversal**, not just vector similarity:

1. **Concept Identification**: User query → vector embedding → find relevant concepts via cosine similarity (boosted by graph centrality)
2. **Subgraph Retrieval**: Starting from relevant concepts, traverse graph to collect:
   - Prerequisites (for scaffolding from simple → complex)
   - Related concepts (near-transfer)
   - Resources that explain concepts
   - Worked examples
3. **Scaffolding**: Order concepts by difficulty and prerequisite chains (easy → medium → hard)
4. **Context Assembly**: Format subgraph into structured context with learning path
5. **Generation**: LLM generates scaffolded answer with resource citations

### Community Detection

Communities represent topic clusters in the knowledge graph:
- Uses Louvain algorithm on undirected projection of concept relationships
- Edges weighted by type (prerequisites weighted higher)
- Enables topic-based navigation and curriculum organization
- Stored in `data/communities.json`

### Vector Store (Supporting Module)

The `vectorstore/` module provides vector embeddings for hybrid retrieval:

- `embeddings.py`: Sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- `storage.py`: MongoDB storage for embeddings
- `retrieval.py`: Cosine similarity search (used by GraphRAG for initial concept finding)
- `basic_rag_query.py`: Basic vector RAG (for comparison only, NOT for assignment)

**Important**: The assignment requires GraphRAG, not basic RAG. Use `graph.graphrag_query`, not `vectorstore.basic_rag_query`.

## Data Directory

Place source materials in `data/pdfs/` before ingestion. Update the example paths in ingestion scripts' `__main__` blocks to point to your actual files.
