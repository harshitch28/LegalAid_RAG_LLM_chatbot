# src/app.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import os, time, uuid
from dotenv import load_dotenv
import streamlit as st

from rag_pipeline import RAGEngine
from memory import MemoryStore

load_dotenv()
# ---- UI / App config ----
st.set_page_config(page_title="LegalAid Chatbot (RAG+Memory)", layout="wide")
st.title("🇮🇳 LegalAid — Conversational RAG with Memory (Gemini + Chroma)")

# Instantiate engine & memory as cached resources (singletons)
@st.cache_resource
def get_engine():
    return RAGEngine()

@st.cache_resource
def get_memory():
    return MemoryStore()

engine = get_engine()
memory = get_memory()

# --- Left column: user/session controls ---
col1, col2 = st.columns([1, 3])
with col1:
    st.header("Session Controls")

    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"
    # simple user_id input (no auth) — for demo only
    user_id = st.text_input("User ID:", st.session_state.user_id)
    st.session_state.user_id = user_id.strip() or st.session_state.user_id

    # list sessions for the user
    sessions = memory.list_sessions(user_id=st.session_state.user_id)
    session_ids = [s["session_id"] for s in sessions]
    session_label_map = {s["session_id"]: f"{s['session_id'][:8]} (msgs:{s['count']})" for s in sessions}

    if "session_id" not in st.session_state:
        # start a brand new session
        st.session_state.session_id = str(uuid.uuid4())

    # Create new session button
    if st.button("🆕 Start new session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.success(f"New session: {st.session_state.session_id[:8]}")

    # Session selector (if sessions exist)
    if session_ids:
        sel = st.selectbox("Choose existing session:", options=["(current)"] + session_ids, format_func=lambda x: session_label_map.get(x, "(new)"))
        if sel != "(current)":
            st.session_state.session_id = sel

    st.markdown("**Session id:**")
    st.code(st.session_state.session_id[:64])
    st.divider()

    # Toggle options
    st.header("Options")
    include_memory = st.checkbox("Include user memory", value=True, help="Use past conversations from this user as context")
    include_kb = st.checkbox("Include legal KB", value=True, help="Retrieve law text from KB")
    show_citations = st.checkbox("Show citations in response", value=True)

    st.divider()
    st.header("Actions")
    if st.button("Export conversation (.txt)"):
        # export from session_state
        hist = st.session_state.get("history", [])
        if hist:
            txt = "\n\n".join([f"You: {q}\nBot: {a}" for q, a in hist])
            st.download_button("Download conversation", data=txt, file_name=f"conversation_{st.session_state.user_id}_{st.session_state.session_id[:8]}.txt")
        else:
            st.info("No conversation to export.")

    if st.button("Clear local history (UI only)"):
        st.session_state.history = []
        st.success("Cleared UI history (backend memory still persists).")

    st.caption("Disclaimer: This is an educational aid, not legal advice. See app disclaimer in README.")

# --- Right column: chat area ---
with col2:
    st.header("Chat")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Input box for user queries
    query = st.text_input("Ask your legal question:", key="query_input")
    send = st.button("Send")

    # handle send
    if send and query:
        # show spinner while waiting
        with st.spinner("Retrieving context and generating answer..."):
            start = time.time()
            # call engine; engine.answer returns structured dict
            try:
                result = engine.answer(user_id=st.session_state.user_id,
                                       session_id=st.session_state.session_id,
                                       query=query)
            except Exception as e:
                st.error(f"Error from backend: {e}")
                result = None

            elapsed = time.time() - start

        if result:
            answer = result.get("answer", "").strip()
            st.session_state.history.append((query, answer))

            # Persist local UI history (display)
            st.success("Answer generated")
            st.write("**Answer:**")
            st.info(answer)

            # Show citations (KB blocks used)
            if show_citations:
                st.markdown("**Citations / KB blocks used:**")
                used_kb = result.get("used_kb", [])
                if used_kb:
                    for i, b in enumerate(used_kb, 1):
                        meta = b.get("meta", {})
                        title = f"{meta.get('act','') } §{meta.get('section_number','')} {meta.get('section_title','')}"
                        st.write(f"- [{i}] {title} — source: `{meta.get('source_file','')}`")
                else:
                    st.write("_No KB snippets used._")

            # Show memory snippets used
            st.markdown("**Memory snippets considered:**")
            used_mem = result.get("used_memory", [])
            if used_mem:
                for m in used_mem:
                    who = m["meta"].get("role", "user")
                    ts = m["meta"].get("timestamp","")
                    st.write(f"- ({who}) `{ts}` — {m['content'][:180]}...")
            else:
                st.write("_No memory snippets used._")

            st.caption(f"Response generation latency: {elapsed:.2f}s, prompt chars: {result.get('prompt_chars', 'n/a')}")

    # Render conversation history
    st.divider()
    st.subheader("Conversation")
    if st.session_state.history:
        for q,a in st.session_state.history[::-1]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")
    else:
        st.write("No messages yet. Start by typing a question and clicking Send.")

    # Memory viewer (inspect user memory)
    st.divider()
    st.subheader("Memory Viewer (backend)")
    if st.button("Load recent memory"):
        mem_list = memory.get_recent_memory(user_id=st.session_state.user_id, session_id=st.session_state.session_id, limit=50)
        if not mem_list:
            st.write("No memory for this user/session yet.")
        else:
            for item in mem_list:
                who = item["meta"].get("role","user")
                ts = item["meta"].get("timestamp","")
                st.write(f"- ({who}) `{ts}` — {item['content'][:300]} ...")
