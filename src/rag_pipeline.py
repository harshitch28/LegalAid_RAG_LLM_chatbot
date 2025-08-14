from __future__ import annotations
import os
from typing import Dict, Any, List

import google.generativeai as genai
from dotenv import load_dotenv

from retriever import Retriever
from memory import MemoryStore
from context_builder import build_context_blocks, render_prompt
from config import TOP_KB_SNIPPETS, TOP_MEMORY_AFTER_SCORE

# Configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class RAGEngine:
    def __init__(self, kb_top: int = TOP_KB_SNIPPETS, mem_top: int = TOP_MEMORY_AFTER_SCORE):
        self.retriever = Retriever(top_k=kb_top)
        self.memory = MemoryStore()
        self.kb_top = kb_top
        self.mem_top = mem_top
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def answer(self, *, user_id: str, session_id: str, query: str) -> Dict[str, Any]:
        # 1) Retrieve
        kb_hits = self.retriever.search(query, top_k=self.kb_top)
        mem_hits_scored = self.memory.search_relevant(user_id=user_id, session_id=session_id, query=query)

        # 2) Build context blocks under budget
        blocks = build_context_blocks(kb_hits=kb_hits, mem_hits=mem_hits_scored)

        # 3) Render prompt & query Gemini
        prompt = render_prompt(query, blocks)
        resp = self.model.generate_content(prompt)
        text = getattr(resp, "text", "").strip()

        # 4) Persist Q&A to memory
        self.memory.save_message(user_id=user_id, session_id=session_id, role="user", content=query)
        self.memory.save_message(user_id=user_id, session_id=session_id, role="assistant", content=text)

        # 5) Return structured result
        return {
            "answer": text,
            "used_kb": kb_hits[:self.kb_top],
            "used_memory": mem_hits_scored[:self.mem_top],
            "prompt_chars": len(prompt)
        }
