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

The system follows a four-stage pipeline:

1. **Ingestion**: Raw content is extracted from various sources and stored in MongoDB's `raw_documents` collection
2. **Chunking**: Raw documents are split into manageable chunks (~1200 chars) and stored in `chunk_documents` collection
3. **Embedding**: Chunks are converted to dense vector embeddings (384-dimensional) using sentence-transformers and stored in `chunk_embeddings` collection
4. **Entity Extraction**: LLM analyzes each chunk to extract structured entities (concepts, resources, examples) stored in `entity_nodes` collection

The vector embeddings enable semantic search and RAG (Retrieval Augmented Generation) for answering questions based on course content.

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

# Verify embeddings
python -m vectorstore.embed_chunks --verify

# Embed only new chunks (incremental)
python -m vectorstore.embed_chunks --new-only

# Extract entities from chunks
python -m graph.extract_entities

# Interactive AI tutor (RAG-based Q&A)
python -m vectorstore.rag_query
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

## Vector Store & RAG

### Vector Database Architecture

The `vectorstore/` module provides semantic search capabilities:

- `embeddings.py`: Generates 384-dimensional vectors using sentence-transformers (all-MiniLM-L6-v2 model)
- `storage.py`: Stores embeddings in MongoDB with efficient indexing
- `retrieval.py`: Performs cosine similarity search (brute-force for simplicity)
- `embed_chunks.py`: Batch processing script to embed all chunks
- `rag_query.py`: RAG implementation combining retrieval + LLM generation

### Semantic Search

The retrieval system uses cosine similarity on dense vectors. For large datasets (>10k chunks), consider:
- MongoDB Atlas Vector Search
- FAISS for in-memory search
- Dedicated vector databases (Pinecone, Weaviate, Qdrant)

### RAG Pipeline

1. User asks a question
2. Question is embedded into a vector
3. Top-k most similar chunks are retrieved via cosine similarity
4. Retrieved chunks are formatted as context
5. Context + question sent to LLM
6. LLM generates answer based on context

The `rag_query.py` module provides both programmatic API (`answer_question()`) and interactive CLI (`interactive_tutor()`).

## Data Directory

Place source materials in `data/pdfs/` before ingestion. Update the example paths in ingestion scripts' `__main__` blocks to point to your actual files.
