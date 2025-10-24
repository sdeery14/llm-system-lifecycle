"""
rag_agent_faiss_min.py

Minimal RAG with Faiss + OpenAI Agents SDK (function tools).
- Seeds a tiny KB
- Exposes a `search_kb` function tool via @function_tool
- Runs two smoke tests when executed directly

Install (Poetry or pip):
    poetry add openai-agents faiss-cpu sentence-transformers numpy
or:
    pip install openai-agents faiss-cpu sentence-transformers numpy

Env:
    export OPENAI_API_KEY=sk-...

Docs (function tools): https://openai.github.io/openai-agents-python/tools/
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Agents SDK imports per docs
from agents import Agent, Runner, function_tool  # <-- function tools per docs

# ------------------------------
# Vector store (Faiss, minimal)
# ------------------------------
@dataclass
class VSConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

class FaissVectorStore:
    def __init__(self, cfg: VSConfig = VSConfig()):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        # Cosine similarity via inner product on L2-normalized vectors
        self.index = faiss.IndexFlatIP(self.dim)
        self.docs: List[str] = []
        self.metas: List[Dict[str, Any]] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype="float32")

    def add_texts(self, texts: Iterable[str], metadatas: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        texts = list(texts)
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        assert len(texts) == len(metadatas), "texts and metadatas length mismatch"
        embs = self._embed(texts)
        self.index.add(embs)
        self.docs.extend(texts)
        self.metas.extend(metadatas)

    def search(self, query: str, k: int = 5, filter_fn: Optional[Any] = None) -> List[Tuple[str, Dict[str, Any], float]]:
        qv = self._embed([query])
        scores, idxs = self.index.search(qv, k=k * 3)  # overfetch a bit
        out: List[Tuple[str, Dict[str, Any], float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            doc, meta = self.docs[i], self.metas[i]
            if (filter_fn is None) or filter_fn(meta):
                out.append((doc, meta, float(score)))
            if len(out) >= k:
                break
        return out

# ---------------------------------
# Seed a tiny customer-service KB
# ---------------------------------
def seed_demo_store() -> FaissVectorStore:
    store = FaissVectorStore()
    docs = [
        "Refunds are available within 30 days of purchase if the item is in original condition.",
        "Orders ship within 2 business days. Expedited shipping is available at checkout.",
        "To change your account email, go to Settings > Account > Email, then verify the new address.",
        "Live chat support is available Monday through Friday, 9am–6pm ET.",
        "Refunds are issued to the original payment method within 5–7 business days after approval.",
    ]
    metas = [
        {"type": "policy", "topic": "refunds"},
        {"type": "policy", "topic": "shipping"},
        {"type": "howto", "topic": "account"},
        {"type": "support", "topic": "hours"},
        {"type": "policy", "topic": "refunds"},
    ]
    store.add_texts(docs, metas)
    return store

# ---------------------------------
# Function tool: search_kb
# ---------------------------------
# Per docs, decorate a plain function; schema & docstrings get auto-parsed. :contentReference[oaicite:1]{index=1}
_store_singleton: Optional[FaissVectorStore] = None  # simple module-level handle

def _get_store() -> FaissVectorStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = seed_demo_store()
    return _store_singleton

@function_tool  # optional: (name_override="search_kb")
def search_kb(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Semantic search over the customer-service KB.

    Args:
        query: The natural-language query.
        k: Number of top results to return.

    Returns:
        A list of {"text": str, "metadata": dict, "score": float}
    """
    store = _get_store()
    hits = store.search(query, k=k)
    return [{"text": d, "metadata": m, "score": s} for (d, m, s) in hits]

# -------------------------------
# Build the Agent
# -------------------------------
def build_agent() -> Agent:
    system_prompt = (
        "You are a concise and accurate customer service assistant.\n"
        "Use the `search_kb` tool to retrieve policies and how-tos when helpful.\n"
        "Cite specifics. If unsure, say you don't know.\n"
    )
    agent = Agent(
        name="CustomerServiceAgent",
        instructions=system_prompt,
        tools=[search_kb],          # pass the decorated function as a tool
        model="gpt-4o-mini",        # pick any model configured in your env
    )
    return agent

# -------------------------------
# Minimal demo run
# -------------------------------
async def demo_conversation() -> str:
    """
    One-turn conversation; model should call search_kb then answer.
    """
    agent = build_agent()
    result = await Runner.run(agent, "Can I get a refund after I buy something?")
    return result.final_output

# -------------------------------
# Simple tests (run on __main__)
# -------------------------------
def test_vector_search_basic():
    store = seed_demo_store()
    hits = store.search("refund timeline", k=2)
    assert len(hits) >= 1, "Expected at least one refund-related hit"
    texts = [h[0].lower() for h in hits]
    assert any("refund" in t for t in texts)

def test_agent_tool_integration_smoke():
    """
    Runs only if OPENAI_API_KEY is present.
    Checks that the agent returns a non-empty response.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("SKIP test_agent_tool_integration_smoke: OPENAI_API_KEY not set.")
        return
    import asyncio
    out = asyncio.run(demo_conversation())
    assert isinstance(out, str) and len(out.strip()) > 0, "Agent response should be non-empty"
    print("Agent said:", out)

if __name__ == "__main__":
    print("Running local tests...")
    test_vector_search_basic()
    test_agent_tool_integration_smoke()
    print("All done.")
