from __future__ import annotations
from typing import List, Dict, Any

from src.config import CONTEXT_CHAR_BUDGET, TOP_KB_SNIPPETS, TOP_MEMORY_AFTER_SCORE

def _clip_char_budget(blocks: List[Dict[str, str]], budget: int) -> List[Dict[str, str]]:
    total = 0
    out = []
    for b in blocks:
        c = b["content"]
        if total + len(c) <= budget:
            out.append(b)
            total += len(c)
        else:
            # take a slice if nothing has been added from this source
            remaining = max(0, budget - total)
            if remaining > 0:
                out.append({"type": b["type"], "content": c[:remaining], "meta": b.get("meta", {})})
                total += remaining
            break
    return out

def build_context_blocks(
    *,
    kb_hits: List[Dict[str, Any]],
    mem_hits: List[Dict[str, Any]],
    kb_top: int = TOP_KB_SNIPPETS,
    mem_top: int = TOP_MEMORY_AFTER_SCORE,
    budget_chars: int = CONTEXT_CHAR_BUDGET
) -> List[Dict[str, Any]]:
    # Select top-N from each source
    kb_sel = [{"type":"kb", "content": h["content"], "meta": h["meta"]} for h in kb_hits[:kb_top]]
    mem_sel = [{"type":"memory", "content": h["content"], "meta": h["meta"]} for h in mem_hits[:mem_top]]

    # Interleave: KB first (authoritative), then Memory (personalization)
    blocks = kb_sel + mem_sel

    # Enforce char budget
    return _clip_char_budget(blocks, budget_chars)

def render_prompt(query: str, blocks: List[Dict[str, Any]]) -> str:
    # Compose a grounded, citation-friendly prompt
    header = (
        "You are a helpful legal assistant specialized in Indian law.\n"
        "Use ONLY the provided context blocks (laws + prior conversation) to answer.\n"
        "If the answer is not present in the context, say you cannot answer based on the given sources.\n"
        "Cite Act name and section/article when possible.\n\n"
    )

    parts = []
    for i, b in enumerate(blocks, 1):
        meta = b.get("meta", {})
        if b["type"] == "kb":
            title = f"{meta.get('act','Law')} &{meta.get('section_number','')} {meta.get('section_title','')}".strip()
            parts.append(f"[KB {i}] {title}\n{b['content']}\n")
        else:
            who = meta.get("role", "user")
            parts.append(f"[MEM {i}] ({who}) {b['content']}\n")

    ctx = "\n".join(parts)
    user = f"User question: {query}\n"

    instructions = (
        "\nResponse rules:\n"
        "1) Be concise and accurate.\n"
        "2) Quote section/article numbers where relevant.\n"
        "3) If unsure, say so and suggest contacting the appropriate Legal Services Authority.\n"
        "4) Include a brief list of which [KB x] blocks you used.\n"
    )

    return header + ctx + "\n" + user + instructions
