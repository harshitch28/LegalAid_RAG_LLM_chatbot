from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime, timezone
import math, uuid

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    VECTOR_DIR, MEMORY_COLLECTION, EMB_MODEL,
    SIM_WEIGHT, RECENCY_WEIGHT, ROLE_WEIGHT, SESSION_WEIGHT,
    RECENCY_HALFLIFE_HOURS, ROLE_SCORES, SAME_SESSION_BONUS, DIFFERENT_SESSION_BONUS,
    MAX_MEMORY_CANDIDATES
)

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class MemoryStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        try:
            self.col = self.client.get_collection(MEMORY_COLLECTION)
        except:
            self.col = self.client.create_collection(MEMORY_COLLECTION, metadata={"hnsw:space":"cosine"})
        self.embedder = SentenceTransformer(EMB_MODEL)

    # ---- write ----
    def save_message(self, *, user_id: str, session_id: str, role: str, content: str) -> str:
        emb = self.embedder.encode([content])[0]
        mid = str(uuid.uuid4())
        self.col.add(
            documents=[content],
            embeddings=[emb],
            metadatas=[{
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "timestamp": utcnow_iso()
            }],
            ids=[mid]
        )
        return mid

    # ---- read + score ----
    def _recency_score(self, iso_ts: str) -> float:
        try:
            t = datetime.fromisoformat(iso_ts.replace("Z","")).astimezone(timezone.utc)
        except Exception:
            return 0.5
        age_hours = (datetime.now(timezone.utc) - t).total_seconds() / 3600.0
        # exponential decay to [0..1], 1 when fresh, ~0 as it gets old
        if RECENCY_HALFLIFE_HOURS <= 0:
            return 0.5
        decay = 0.5 ** (age_hours / RECENCY_HALFLIFE_HOURS)
        return max(0.0, min(1.0, decay))

    def _role_score(self, role: str) -> float:
        return ROLE_SCORES.get(role, 0.8)

    def _session_bonus(self, mem_session_id: str, current_session_id: str) -> float:
        return SAME_SESSION_BONUS if mem_session_id == current_session_id else DIFFERENT_SESSION_BONUS

    def search_relevant(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        n_candidates: int = MAX_MEMORY_CANDIDATES
    ) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([query])[0]
        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=n_candidates,
            where={"user_id": user_id},
        )
        docs = res.get("documents", [[]])[0]
        mets = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]  # cosine distance [0..2]
        items = []
        for d, m, dist in zip(docs, mets, dists):
            # Convert distance to similarity in [0..1] (approx)
            sim = 1.0 - float(dist)
            rec = self._recency_score(m.get("timestamp", ""))
            rscore = self._role_score(m.get("role", "user"))
            sbonus = self._session_bonus(m.get("session_id", ""), session_id)

            final = (
                SIM_WEIGHT * sim +
                RECENCY_WEIGHT * rec +
                ROLE_WEIGHT * rscore +
                SESSION_WEIGHT * sbonus
            )
            items.append({
                "content": d,
                "meta": m,
                "scores": {"sim": sim, "recency": rec, "role": rscore, "session": sbonus, "final": final}
            })

        # sort by final score descending
        items.sort(key=lambda x: x["scores"]["final"], reverse=True)
        return items
    
        # Delete all memory for a user (GDPR-style)
    def delete_user(self, user_id: str):
        res = self.col.get(where={"user_id": user_id})
        ids = res.get("ids", [])
        if ids:
            self.col.delete(ids=ids)

    # Trim oldest messages beyond a cap
    def trim_user(self, user_id: str, keep_last: int = 500):
        res = self.col.get(where={"user_id": user_id})
        ids = res.get("ids", [])
        mets = res.get("metadatas", [])
        if not ids: return
        # sort by timestamp ascending, remove oldest
        pairs = sorted(zip(ids, mets), key=lambda x: x[1].get("timestamp", ""))
        if len(pairs) > keep_last:
            drop = [pid for pid, _ in pairs[:-keep_last]]
            self.col.delete(ids=drop)

