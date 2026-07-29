"""
main.py — FastAPI wrapper around the LangGraph RAG flow.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860

Then:
    curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \\
         -d '{"query": "What is your delivery fee?"}'
"""

from fastapi import FastAPI

from graph import run_query
from schemas import AskRequest, AskResponse

app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = run_query(request.query)
    return AskResponse(**result)


@app.get("/")
def health():
    return {"status": "ok", "service": "zepto-support-assistant"}
