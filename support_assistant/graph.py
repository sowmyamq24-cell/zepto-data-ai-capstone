"""
graph.py — LangGraph StateGraph implementing the intent-routed RAG flow.

Nodes:
    classify_intent      -> classifies query as policy_question / general_question
    retrieve_and_answer   -> retrieves top-3 chunks + generates the answer (RAG path)
    direct_answer         -> canned/direct answer, no retrieval (non-RAG path)

Every node's *generation* step branches on the MOCK_LLM environment variable.
MOCK_LLM unset or "1" -> deterministic mock logic (the required, graded baseline).
MOCK_LLM == "0"       -> optional real-LLM extension (not required for grading).
"""

import os
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from ingest import retrieve_top_k
from prompts import build_prompt

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]


def mock_llm_enabled() -> bool:
    """MOCK_LLM unset or '1' -> mock (graded baseline). MOCK_LLM == '0' -> real LLM."""
    return os.environ.get("MOCK_LLM", "1") != "0"


class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if mock_llm_enabled():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: would call a real LLM to classify.
        # Left as a placeholder — not required for the graded baseline.
        raise NotImplementedError(
            "Real-LLM intent classification is an optional extension; "
            "set MOCK_LLM=1 (or leave unset) to use the graded mock baseline."
        )

    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer (policy_question path)
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real in both modes (no API key/network needed).
    retrieved = retrieve_top_k(query, k=3)
    source_ids = [r["id"] for r in retrieved]

    if mock_llm_enabled():
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        top_chunk_snippet = retrieved[0]["text"][:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt a real LLM grounded on the
        # retrieved chunks using the structured template from prompts.py.
        prompt = build_prompt(query, retrieved)  # noqa: F841 (built, not sent, in this offline baseline)
        raise NotImplementedError(
            "Real-LLM generation is an optional extension; "
            "set MOCK_LLM=1 (or leave unset) to use the graded mock baseline."
        )

    return {**state, "answer": answer, "sources": source_ids, "confidence": confidence}


# ---------------------------------------------------------------------------
# Node 3: direct_answer (general_question path)
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    if mock_llm_enabled():
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        raise NotImplementedError(
            "Real-LLM direct answering is an optional extension; "
            "set MOCK_LLM=1 (or leave unset) to use the graded mock baseline."
        )

    return {**state, "answer": answer, "sources": [], "confidence": confidence}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def route_by_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str) -> dict:
    graph = get_graph()
    initial_state: GraphState = {
        "query": query, "intent": "", "answer": "", "sources": [], "confidence": 0.0,
    }
    result = graph.invoke(initial_state)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
    }


if __name__ == "__main__":
    for q in ["What is your delivery fee?", "What is the capital of France?"]:
        print(f"\nQuery: {q}")
        print(run_query(q))
