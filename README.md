
This project implements a full **retrieval-augmented generation (RAG)** pipeline, but instead of using only vector search, we use a **graph-based retrieval engine** that extracts concepts, prerequisites, examples, and resources from course documents — creating structured, scaffolded answers for tutoring.

Authors:
* **Bram Simonnet**
* **Ryan Fleshman**

# **Features**

### **Graph-based retrieval (GraphRAG)**

Automatically retrieves concepts, relationships, examples, and prerequisite chains from a custom knowledge graph.

### **Scaffolded tutoring answers**

Answers are generated **from simple → complex**, following prerequisite paths.

### **Uses course PDFs, YouTube transcripts, and slides**

All course material is chunked, embedded, and converted into a multi-relational knowledge graph.

### **Local LLM execution with LM Studio**

Supports running models such as:

* Qwen2.5-7B-Instruct
* Mistral-7B
* Llama 3.8B

### **Web UI “Erica — Your AI Tutor”**

A simple front-end chat UI that connects to the FastAPI backend.

---

# **System Architecture**

```
data/
    ├── pdfs/
    ├── visualizations/
    ├── knowledge_graph.pkl
    ├── communities.json

backend/
    ├── app.py               <-- FastAPI server
    ├── llm.py               <-- LM Studio API client
    ├── graph/
         ├── extract_entities.py
         ├── extract_relationships.py
         ├── build_graph.py
         ├── graphrag_retrieval.py
         ├── graphrag_query.py
    ├── static/
         └── chat.html       <-- UI

docker-compose.yml
README.md
```

The pipeline has 7 major stages:

---

# **1. Ingestion**

Raw documents → processed text
Sources include:

* Course lectures
* PDFs (ex: “The Learning Problem – Engineering AI Agents”)
* YouTube videos and transcripts
* Web pages from the instructor’s site

---

# **2. Chunking**

Documents are split into ~1200-character chunks for easier processing.

---

# **3. Embedding**

Each chunk is embedded using:

```
all-MiniLM-L6-v2  (384-dim vectors)
```

Stored and used to:

* find relevant concepts
* map similar content
* support retrieval

---

# **4. Entity Extraction**

Each chunk is sent to an LLM to identify:

* Concepts
* Resources
* Examples

Example output:

```
Concept: Variational Lower Bound (ELBO)
Resource: "The Learning Problem" PDF (page 5)
Example: VAE reconstruction loss
```

---

# **5. Relationship Extraction**

The system infers graph edges:

* **prerequisite**
* **explains**
* **related**
* **example_of**

Result: a **multi-relational knowledge graph** (NetworkX MultiDiGraph).

---

# **6. Graph Construction**

Nodes + edges are merged into:

```
knowledge_graph.pkl
```

with full metadata:

* type (Concept / Resource / Example)
* definition
* difficulty
* centrality scores
* etc.

The real graph contains:

| Type      | Count |
| --------- | ----- |
| Concepts  | 91    |
| Resources | 67    |
| Examples  | 58    |

Edges: 157 (all 4 relationship types)

---

# **7. GraphRAG Query Engine**

When a student asks a question:

1. Vector search identifies relevant concepts

2. Graph traversal retrieves:

   * prerequisites
   * related concepts
   * explanations
   * example nodes

3. A **scaffolded learning path** is built (easy → medium → hard)

4. A structured context is assembled

5. The LLM receives a **custom, pedagogically-designed system prompt**, including:

   * ordered concepts
   * definitions
   * examples
   * resources
   * instructions for citations
   * instructions for code blocks

6. The final answer is produced.

---

# **Demo Questions Included**

The system includes three demonstration outputs required by the assignment:

### **1. Explain attention mechanisms in transformers**

* 5 concepts retrieved
* scaffolded learning path
* Python code examples
* resource citations

### **2. What are applications of CLIP?**

* concepts + examples
* pipeline explanation
* sample code

### **3. Explain variational bounds and Jensen’s inequality**

* 9 concepts retrieved
* ELBO derivation
* VAE loss explanation
* full mathematical breakdown

These outputs appear in the assignment deliverable document.

---

# **Running the System**

### **1. Start LM Studio**

Load a compatible model (e.g., **Qwen2.5-7B-Instruct**, used in the building of this system)
Make sure the local server is reachable at:

```
http://127.0.0.1:1234
```

### **2. Launch backend**

```
docker compose up --build
```

Backend now runs at:

```
http://localhost:8000
```

### **3. Open UI**

Go to:

```
http://localhost:8000
```

You can now chat with **Erica**.

---

# **Tech Stack**

* Python 3.10
* FastAPI
* NetworkX
* Sentence Transformers
* LM Studio (OpenAI-compatible API)
* Docker
* HTML/CSS (simple chat UI)

---

# Contributors

* **Bram Simonnet**
* **Ryan Fleshman**
