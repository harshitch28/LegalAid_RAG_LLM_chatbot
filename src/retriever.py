from __future__ import annotations
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

from config import VECTOR_DIR, KB_COLLECTION, EMB_MODEL

class Retriever:
    def __init__(self, top_k: int = 5):
        self.client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        self.kb = self.client.get_or_create_collection(KB_COLLECTION)
        self.embedder = SentenceTransformer(EMB_MODEL)
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        q_emb = self.embedder.encode([query])[0]
        res = self.kb.query(query_embeddings=[q_emb], n_results=k)
        docs = res.get("documents", [[]])[0]
        mets = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        items = []
        for d, m, dist in zip(docs, mets, dists):
            sim = 1.0 - float(dist)
            items.append({"content": d, "meta": m, "score": sim})
        items.sort(key=lambda x: x["score"], reverse=True)
        return items
