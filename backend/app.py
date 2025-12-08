from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph.graphrag_retrieval import GraphRAG

rag = GraphRAG()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("static/chat.html")


class Query(BaseModel):
    question: str


@app.post("/query")
def query_endpoint(request: Query):
    try:
        result = rag.run(request.question)

        return {
            "answer": result.get("final_response", "No answer."),
            "nodes_used": result.get("nodes_used", []),
            "resources_used": result.get("resources_used", {}),
            "system_prompt": result.get("system_prompt", "")
        }

    except Exception as e:
        return {
            "answer": f"Backend error: {str(e)}",
            "nodes_used": [],
            "resources_used": {},
            "system_prompt": ""
        }
