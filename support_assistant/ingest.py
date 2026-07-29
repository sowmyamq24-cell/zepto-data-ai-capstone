"""
ingest.py — loads the 8 policy documents, chunks them (one chunk per document,
since each is short), embeds each chunk with all-MiniLM-L6-v2, and stores the
embeddings in a ChromaDB collection.

Building the collection requires downloading the all-MiniLM-L6-v2 model the
first time (from the sentence-transformers/HuggingFace cache) — internet is
needed once; the model is cached locally afterward, same pattern as the
Titanic loader in /analytics.
"""

import glob
import os

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
COLLECTION_NAME = "zepto_policies"

_model = None
_client = None
_collection = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def load_chunks():
    """One chunk per document (each doc is short enough that finer chunking
    isn't needed). Returns list of (chunk_id, text) tuples."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt"))):
        chunk_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks.append((chunk_id, text))
    return chunks


def build_collection():
    """Builds (or rebuilds) an in-memory ChromaDB collection with all 8
    embedded chunks. Called once at app startup."""
    global _client, _collection

    _client = chromadb.EphemeralClient()  # in-memory, no disk persistence needed
    _collection = _client.get_or_create_collection(name=COLLECTION_NAME)

    model = get_embedding_model()
    chunks = load_chunks()

    ids = [c[0] for c in chunks]
    texts = [c[1] for c in chunks]
    embeddings = model.encode(texts).tolist()

    _collection.add(ids=ids, documents=texts, embeddings=embeddings)
    return _collection


def get_collection():
    global _collection
    if _collection is None:
        build_collection()
    return _collection


def retrieve_top_k(query: str, k: int = 3):
    """Embed the query and retrieve the top-k most similar chunks
    (cosine similarity, via ChromaDB's default HNSW index)."""
    collection = get_collection()
    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    retrieved = []
    for chunk_id, text, distance in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        retrieved.append({"id": chunk_id, "text": text, "distance": distance})
    return retrieved


if __name__ == "__main__":
    build_collection()
    print(f"Indexed {len(load_chunks())} chunks into ChromaDB collection '{COLLECTION_NAME}'")
    demo = retrieve_top_k("How long is standard delivery free threshold?", k=3)
    for r in demo:
        print(f"  {r['id']} (distance={r['distance']:.4f}): {r['text'][:80]}...")
