# /support_assistant — Zepto Data & AI Platform

A small RAG-based GenAI support assistant for Zepto's own policy corpus, built
with sentence-transformers + ChromaDB (retrieval), LangGraph (intent routing
and orchestration), Pydantic (structured output), and FastAPI (serving).

## MOCK_LLM — read this first

Every LLM call in this module is gated behind the `MOCK_LLM` environment
variable:
- **Unset, or `MOCK_LLM=1` (default, graded baseline):** fully deterministic,
  rule-based logic — no signup, no API key, no network call to any LLM
  provider. This is what gets graded and is fully correct on its own.
- **`MOCK_LLM=0` (optional, ungraded extension):** would call a real LLM
  (e.g. Groq's free tier) using the structured prompt template in
  `prompts.py`. **Not required or graded** — left as a documented
  `NotImplementedError` stub in this submission; the mock baseline is what's
  submitted for grading.

Retrieval (embedding the query + querying ChromaDB) always runs for real in
**both** modes — it needs no API key and no network call once the
embedding model is cached locally, so it isn't part of the mock/real split.

## Setup & run

```bash
pip install -r requirements.txt
python ingest.py          # sanity-checks corpus loading + embedding + retrieval
uvicorn main:app --host 0.0.0.0 --port 7860
```

The first run downloads the `all-MiniLM-L6-v2` model from the
sentence-transformers/HuggingFace cache (internet needed once); subsequent
runs reuse the local cache, same pattern as `sns.load_dataset` in
`/analytics`.

### Docker (required, graded baseline for containerization)

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

This builds and runs the FastAPI app locally, serving `POST /ask` on
`http://localhost:7860/ask`. (Pushing to Hugging Face Spaces is an optional,
ungraded stretch not attempted in this submission.)

## Architecture — ingestion → embedding → retrieval → generation

**Ingestion** (`ingest.py`, `load_chunks()`): the 8 policy documents in
`docs/doc_01.txt` … `docs/doc_08.txt` are read from disk. Given each
document is a single short policy paragraph, chunking is one chunk per
document (no finer splitting needed).

**Embedding** (`ingest.py`, `get_embedding_model()` / `build_collection()`):
each chunk is embedded with `sentence-transformers`' `all-MiniLM-L6-v2`
model (runs locally, no API key). The 8 embeddings are stored, alongside
their chunk IDs and raw text, in an in-memory ChromaDB collection named
`zepto_policies` (`chromadb.EphemeralClient()`), built once at app startup.

**Retrieval** (`ingest.py`, `retrieve_top_k()`, called from the
`retrieve_and_answer` node in `graph.py`): the incoming query is embedded
with the same model, then ChromaDB's collection is queried for the top-3
most similar chunks by cosine similarity (via its default HNSW index). This
step runs identically regardless of `MOCK_LLM`.

**Generation** (`graph.py`): a LangGraph `StateGraph` with a `TypedDict`
state (`query`, `intent`, `answer`, `sources`, `confidence`) and 3 nodes:
- `classify_intent` — keyword heuristic (mock) vs. LLM classification
  (optional extension) — decides `policy_question` vs. `general_question`.
- `retrieve_and_answer` — runs retrieval (always real), then either returns
  a canned `"Based on the retrieved context: ..."` string built from the
  top chunk (mock/graded) or would prompt a real LLM with the structured
  template from `prompts.py` (optional extension).
- `direct_answer` — returns a fixed canned string (mock/graded) or would
  prompt the LLM directly with no retrieval (optional extension).

A conditional edge out of `classify_intent` (`route_by_intent` in
`graph.py`) routes to `retrieve_and_answer` or `direct_answer` based on the
classified intent — this routing logic itself does not depend on
`MOCK_LLM`, only the generation step inside each node does.

**Structured output** (`schemas.py` + `main.py`): the final state is
validated against a Pydantic `AskResponse` model (`answer: str`,
`sources: List[str]`, `confidence: float` in `[0, 1]`), then returned by the
FastAPI `POST /ask` endpoint. In mock mode, this schema is populated
deterministically in code (chunk IDs for `sources` on the retrieval path,
empty list on the direct-answer path, fixed `confidence=1.0`) — there is no
LLM output to fail validation, since none was generated on this path.

**Data flow:** `docs/*.txt` → `ingest.py` (chunk + embed) → ChromaDB
(`zepto_policies` collection) → `graph.py`'s `classify_intent` routes the
query → `retrieve_and_answer` (queries ChromaDB, then mock/real generation)
or `direct_answer` (mock/real generation, no retrieval) → validated
`AskResponse` → returned by `main.py`'s `POST /ask`.

## Structured prompt template (Task 2, used by the optional MOCK_LLM=0 path)

See `prompts.py` for the full template. It follows the
role → context → task → format → length skeleton, and includes:
- **Negative constraint:** *"Do not answer using information not present in
  the provided context..."*
- **Few-shot example:** a worked example question/context/answer triple
  embedded directly in the template.

This template is not invoked in the required mock baseline (no LLM is
called), but is present as required, ready for the optional extension.

## Example calls (run with `MOCK_LLM` left at its default)

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is your delivery fee for small orders?"}'
```
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below t",
  "sources": ["doc_01", "doc_05", "doc_04"],
  "confidence": 1.0
}
```
*(This query contains the keyword "delivery" → routed to `policy_question`
→ `retrieve_and_answer`, which retrieved chunks including `doc_01`, the
correct source document for delivery fees.)*

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
*(No policy keyword present → routed to `general_question` → `direct_answer`,
a fixed canned string with no retrieval and no LLM call.)*

> **Note:** the exact `sources` ordering above depends on the real
> `all-MiniLM-L6-v2` embeddings computed on your machine (this repository's
> automated wiring test used a placeholder embedding function, since this
> development environment has no access to download the HuggingFace model —
> re-run `uvicorn main:app` locally, where internet access lets the real
> model download once, and paste your own two example transcripts here to
> replace these illustrative ones before final submission).
