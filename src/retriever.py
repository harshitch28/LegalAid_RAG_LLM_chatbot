from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import VECTOR_DIR, KB_COLLECTION, EMB_MODEL

class Retriever:
    def __init__(self, top_k: int = 5):
        self.client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        self.kb = self.client.get_or_create_collection(KB_COLLECTION)
        self.embedder = SentenceTransformer(EMB_MODEL)
        self.top_k = top_k

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        k = top_k or self.top_k
        q_emb = self.embedder.encode([query])[0]
        res = self.kb.query(query_embeddings=[q_emb], n_results=k)
        hits = []
        docs = res.get("documents", [[]])[0]
        mets = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for d, m, dist in zip(docs, mets, dists):
            hits.append({
                "content": d,
                "metadata": m,
                "score": 1 - dist  # cosine similarity proxy
            })
        return hits
