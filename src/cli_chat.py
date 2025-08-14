from __future__ import annotations
import uuid
from src.rag_pipeline import RAGEngine

def new_session_id() -> str:
    return str(uuid.uuid4())

if __name__ == "__main__":
    print("RAG + Memory CLI (type 'new' for new session, 'su <user>' to switch user, 'q' to quit)")
    engine = RAGEngine()
    user_id = "user_001"
    session_id = new_session_id()
    print(f"Current user: {user_id} | session: {session_id}")

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if q.lower().startswith("new"):
            session_id = new_session_id()
            print(f"→ started new session: {session_id}")
            continue
        if q.lower().startswith("su "):
            user_id = q.split(" ", 1)[1].strip() or user_id
            print(f"→ switched to user: {user_id}")
            continue

        out = engine.answer(user_id=user_id, session_id=session_id, query=q)
        print("\nBot:", out["answer"])
