import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import os, uuid, time
from dotenv import load_dotenv
import gradio as gr
from rag_pipeline import RAGEngine
from memory import MemoryStore

load_dotenv()

engine = RAGEngine()
memory = MemoryStore()

def get_sessions(user_id):
    sessions = memory.list_sessions(user_id=user_id)
    session_ids = [s["session_id"] for s in sessions]
    session_labels = [f"{sid[:8]} (msgs:{s['count']})" for sid, s in zip(session_ids, sessions)]
    return session_ids, session_labels

def start_new_session(user_id):
    session_id = str(uuid.uuid4())
    return session_id

def chat_fn(history, user_id, session_id, include_memory, include_kb, show_citations, query):
    if not query:
        return history, gr.update(value=""), ""
    start = time.time()
    try:
        result = engine.answer(user_id=user_id, session_id=session_id, query=query)
    except Exception as e:
        return history, gr.update(value=""), f"Error: {e}"
    elapsed = time.time() - start
    answer = result.get("answer", "").strip()
    # Append messages as dicts for Gradio 'messages' format
    history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer}
    ]

    citations = ""
    if show_citations:
        used_kb = result.get("used_kb", [])
        if used_kb:
            citations += "**Citations / KB blocks used:**\n"
            for i, b in enumerate(used_kb, 1):
                meta = b.get("meta", {})
                title = f"{meta.get('act','') } §{meta.get('section_number','')} {meta.get('section_title','')}"
                citations += f"- [{i}] {title} — source: `{meta.get('source_file','')}`\n"
        else:
            citations += "_No KB snippets used._\n"

    used_mem = result.get("used_memory", [])
    citations += "\n**Memory snippets considered:**\n"
    if used_mem:
        for m in used_mem:
            who = m["meta"].get("role", "user")
            ts = m["meta"].get("timestamp","")
            citations += f"- ({who}) `{ts}` — {m['content'][:180]}...\n"
    else:
        citations += "_No memory snippets used._\n"

    citations += f"\nResponse generation latency: {elapsed:.2f}s, prompt chars: {result.get('prompt_chars', 'n/a')}"
    return history, gr.update(value=""), citations

def export_conversation(history, user_id, session_id):
    if not history:
        return None
    # Extract content from dicts
    pairs = []
    for i in range(0, len(history), 2):
        user_msg = history[i] if i < len(history) and history[i].get("role") == "user" else None
        bot_msg = history[i+1] if i+1 < len(history) and history[i+1].get("role") == "assistant" else None
        if user_msg and bot_msg:
            pairs.append(f"You: {user_msg['content']}\nBot: {bot_msg['content']}")
        elif user_msg:
            pairs.append(f"You: {user_msg['content']}")
        elif bot_msg:
            pairs.append(f"Bot: {bot_msg['content']}")
    txt = "\n\n".join(pairs)
    filename = f"conversation_{user_id}_{session_id[:8]}.txt"
    return gr.File.update(value=(filename, txt.encode("utf-8")))

def load_recent_memory(user_id, session_id):
    mem_list = memory.get_recent_memory(user_id=user_id, session_id=session_id, limit=50)
    if not mem_list:
        return "No memory for this user/session yet."
    out = ""
    for item in mem_list:
        who = item["meta"].get("role","user")
        ts = item["meta"].get("timestamp","")
        out += f"- ({who}) `{ts}` — {item['content'][:300]} ...\n"
    return out

with gr.Blocks(title="LegalAid Chatbot (RAG+Memory)") as demo:
    gr.Markdown("# 🇮🇳 LegalAid — Conversational RAG with Memory (Gemini + Chroma)")
    with gr.Row():
        with gr.Column(scale=1):
            user_id = gr.Textbox(label="User ID", value=f"user_{str(uuid.uuid4())[:8]}", interactive=True)
            session_ids, session_labels = get_sessions(user_id.value)
            session_id = gr.Textbox(label="Session ID", value=str(uuid.uuid4()), interactive=True)
            new_session_btn = gr.Button("🆕 Start new session")
            session_dropdown = gr.Dropdown(
                label="Choose existing session",
                choices=session_ids or ["(none)"],
                value=session_ids[0] if session_ids else "(none)",
                interactive=True
            )
            include_memory = gr.Checkbox(label="Include user memory", value=True)
            include_kb = gr.Checkbox(label="Include legal KB", value=True)
            show_citations = gr.Checkbox(label="Show citations in response", value=True)
            export_btn = gr.Button("Export conversation (.txt)")
            export_file = gr.File(label="Download conversation")
            clear_btn = gr.Button("Clear local history (UI only)")
            disclaimer = gr.Markdown("Disclaimer: This is an educational aid, not legal advice. See app disclaimer in README.")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", value=[], show_label=True, type="messages")
            query = gr.Textbox(label="Ask your legal question:")
            send_btn = gr.Button("Send")
            citations_box = gr.Markdown()
            memory_viewer_btn = gr.Button("Load recent memory")
            memory_viewer = gr.Markdown()

    # Button logic
    new_session_btn.click(lambda: str(uuid.uuid4()), None, session_id)
    session_dropdown.change(lambda sid: sid if sid else session_id.value, session_dropdown, session_id)
    send_btn.click(chat_fn, [chatbot, user_id, session_id, include_memory, include_kb, show_citations, query], [chatbot, query, citations_box])
    export_btn.click(export_conversation, [chatbot, user_id, session_id], export_file)
    clear_btn.click(lambda: [], None, chatbot)
    memory_viewer_btn.click(load_recent_memory, [user_id, session_id], memory_viewer)

demo.launch()
