"""
Meal & Health Agent — rag.py
Retrieval-Augmented Generation over a local nutrition knowledge base.

How it works:
  1. Load nutrition_kb.json (20 entries of food/nutrient facts)
  2. Embed each entry once using OpenAI text-embedding-3-small, cache to disk
  3. At query time, embed the query and return top-k entries by cosine similarity
  4. The retrieved entries are injected into the meal planner prompt

This grounds the LLM's meal recommendations in actual nutritional facts
rather than relying purely on parametric knowledge.
"""

import os
import json
import numpy as np
from openai import OpenAI

KB_PATH         = os.path.join(os.path.dirname(__file__), "nutrition_kb.json")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "nutrition_kb_embeddings.json")
EMBED_MODEL     = "text-embedding-3-small"


def _load_kb() -> list[dict]:
    with open(KB_PATH) as f:
        return json.load(f)


def _embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _load_or_build_embeddings(client: OpenAI) -> tuple[list[dict], list[list[float]]]:
    """
    Load cached embeddings from disk, or build and cache them if missing.
    Embeddings are only computed once — subsequent runs load from JSON.
    """
    kb = _load_kb()

    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH) as f:
            cached = json.load(f)
        # Validate cache matches current KB
        if len(cached) == len(kb) and cached[0]["id"] == kb[0]["id"]:
            embeddings = [entry["embedding"] for entry in cached]
            return kb, embeddings

    # Build embeddings
    print("  [rag] Building knowledge base embeddings (one-time setup)...")
    texts = [entry["text"] for entry in kb]
    embeddings = _embed(texts, client)

    # Cache to disk
    cache = [{"id": kb[i]["id"], "embedding": embeddings[i]} for i in range(len(kb))]
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(cache, f)

    return kb, embeddings


def retrieve(query: str, top_k: int = 4) -> list[str]:
    """
    Retrieve the top-k most relevant knowledge base entries for a query.
    Returns a list of text strings ready to inject into a prompt.

    Args:
        query:  Natural language description of what's needed
                e.g. "iron deficiency, lactose intolerant, muscle gain"
        top_k:  Number of entries to return (default 4)

    Returns:
        List of relevant fact strings from the knowledge base.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    kb, kb_embeddings = _load_or_build_embeddings(client)

    # Embed the query
    query_embedding = _embed([query], client)[0]

    # Score all entries
    scored = [
        (i, _cosine_similarity(query_embedding, kb_embeddings[i]))
        for i in range(len(kb))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = [kb[i]["text"] for i, _ in scored[:top_k]]
    return top


def format_context(retrieved: list[str]) -> str:
    """Format retrieved entries as a numbered context block for a prompt."""
    if not retrieved:
        return ""
    lines = ["Relevant nutrition knowledge:"]
    for i, entry in enumerate(retrieved, 1):
        lines.append(f"{i}. {entry}")
    return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    results = retrieve("low iron, lactose intolerant, muscle gain", top_k=3)
    print(format_context(results))
