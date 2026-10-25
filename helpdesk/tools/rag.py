"""
helpdesk/tools/rag.py - Retrieval over the ChromaDB corpus (nomic-embed-text, cosine distance).
"""

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import chromadb
import ollama
from google.adk.tools.tool_context import ToolContext

from helpdesk import config
from helpdesk.blackboard import note_sources

_collection = None
_ollama_client = None


def get_chroma_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = client.get_or_create_collection(name=config.COLLECTION_NAME,
                                                       metadata={"hnsw:space": "cosine"})
    return _collection


def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=config.OLLAMA_API_BASE)
    return _ollama_client


@lru_cache(maxsize=1)
def supersession() -> Dict[str, list]:
    """doc_id -> doc_ids it supersedes, from the `supersedes` column of ingest/manifest.csv."""
    manifest = Path(config.BASE_DIR) / "ingest" / "manifest.csv"
    with open(manifest, encoding="utf-8") as f:
        return {r["doc_id"]: [d for d in r.get("supersedes", "").split(";") if d] for r in csv.DictReader(f)}


def apply_supersession(results: list) -> list:
    """
    Argumentation by recency: when a newer circular and an older regulation it amends are retrieved
    together, the circular attacks (defeats) the older passage. Put the newer document first and mark
    the older passages so the agent answers from the circular.
    """
    sup = supersession()
    present = {r["doc_id"] for r in results}
    for r in results:
        newer = [d for d in present if r["doc_id"] in sup.get(d, [])]
        if newer:
            r["superseded_by"] = newer
            r["note"] = (f"Older rule: superseded on this topic by {', '.join(f'[{d}]' for d in newer)}. "
                         "Answer from the newer document.")
    return sorted(results, key=lambda r: ("superseded_by" in r, -bool(sup.get(r["doc_id"])), r["distance"]))


def search_documents(query: str, top_k: int = config.DEFAULT_TOP_K,
                     tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
    """
    Search the official IIT Jodhpur academic documents (regulations, circulars, curricula,
    academic calendar, slot system rules).

    Args:
        query: The question or keywords to look up.
        top_k: Number of passages to return (default 5).

    Returns:
        Passages with their doc_id, page, document date and a ready-made citation such as
        [UG_REG_2019, p.12]. When passages disagree, the more recent document (date) prevails.
        status is 'no_relevant_docs' when nothing is close enough; then do not answer.
    """
    try:
        emb = get_ollama_client().embed(model=config.EMBEDDING_MODEL, input=f"search_query: {query.strip()}")
        query_emb = emb["embeddings"][0]
    except Exception as e:
        return {"status": "error", "message": f"Embedding failed: {e}"}

    res = get_chroma_collection().query(query_embeddings=[query_emb], n_results=int(top_k),
                                        include=["documents", "metadatas", "distances"])
    if not res["ids"] or not res["ids"][0]:
        return {"status": "no_relevant_docs", "message": "The document index is empty or has no match."}

    tau = config.DEFAULT_TAU_THRESHOLD  # read at call time so the eval can sweep it
    hits = [(d, m, t) for d, m, t in zip(res["distances"][0], res["metadatas"][0], res["documents"][0]) if d <= tau]
    if not hits:
        return {
            "status": "no_relevant_docs",
            "message": f"No passage is within the relevance threshold (best distance "
                       f"{res['distances'][0][0]:.3f} > {tau}). Do not answer; log the question.",
        }

    results, source_map = [], {}
    for i, (dist, meta, text) in enumerate(hits, start=1):
        doc_id, page = meta.get("doc_id", "UNKNOWN_DOC"), meta.get("page", 1)
        source_map[f"S{i}"] = doc_id
        results.append({
            "cite_as": f"[{doc_id}, p.{page}]",
            "doc_id": doc_id,
            "title": meta.get("title", ""),
            "date": meta.get("date", ""),
            "page": page,
            "distance": round(float(dist), 4),
            "text": text,
        })

    results = apply_supersession(results)
    note_sources(tool_context, *[r["doc_id"] for r in results], source_map=source_map)
    return {"status": "success", "query": query, "count": len(results), "results": results}
