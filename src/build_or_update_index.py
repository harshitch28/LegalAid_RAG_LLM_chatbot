import json, glob
from pathlib import Path
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import DATA_PROCESSED, NOTES_DIR, VECTOR_DIR, KB_COLLECTION, EMB_MODEL, MAX_CHARS, OVERLAP, STATE_DIR
from utils import soft_clean, sliding_chunks, sha256_text
from state_registry import StateRegistry

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_notes():
    for p in NOTES_DIR.glob("*.md"):
        yield {
            "act": "Procedural Guide",
            "section_number": "",
            "section_title": p.stem.replace("_", " ").title(),
            "content": p.read_text(encoding="utf-8"),
            "source_file": str(p)
        }

def make_docs():
    # All law sections from jsonl
    for jf in sorted(DATA_PROCESSED.glob("*.jsonl")):
        for rec in load_jsonl(jf):
            # Expect keys: act, section_number, section_title, content (from Step 2)
            act = rec.get("act", jf.stem)
            sec = (rec.get("section_number") or "").strip()
            title = (rec.get("section_title") or "").strip()
            content = soft_clean(rec.get("content") or rec.get("text",""))

            if not content:
                continue

            base_meta = {
                "act": act,
                "section_number": sec,
                "section_title": title,
                "source_file": jf.name
            }
            # Sub-chunk if extremely long
            for idx, chunk in enumerate(sliding_chunks(content, MAX_CHARS, OVERLAP)):
                yield {
                    "text": chunk,
                    "meta": {**base_meta, "sub_index": idx}
                }

    # Add manual notes
    for rec in load_notes():
        content = soft_clean(rec["content"])
        base_meta = {
            "act": rec["act"],
            "section_number": rec["section_number"],
            "section_title": rec["section_title"],
            "source_file": rec["source_file"]
        }
        for idx, chunk in enumerate(sliding_chunks(content, MAX_CHARS, OVERLAP)):
            yield {
                "text": chunk,
                "meta": {**base_meta, "sub_index": idx}
            }

def main():
    # Persistent vector client
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    kb = client.get_or_create_collection(
        name=KB_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # State registry
    state = StateRegistry(STATE_DIR / "kb_chunks.json")

    # Embedder
    model = SentenceTransformer(EMB_MODEL)

    # Gather new docs
    new_docs, new_metas, new_ids = [], [], []
    now = datetime.utcnow().isoformat()

    for doc in tqdm(make_docs(), desc="Scanning docs"):
        text = doc["text"].strip()
        meta = doc["meta"]

        # Hash for dedup
        chunk_sha = sha256_text(text)
        if state.has(chunk_sha):
            continue

        # Compose ID from sha (stable & unique)
        doc_id = f"sha:{chunk_sha[:32]}"

        new_docs.append(text)
        new_metas.append({**meta, "chunk_sha": chunk_sha, "added_at": now})
        new_ids.append(doc_id)

        # Add to state memory (we’ll persist after successful add)
        state.add(chunk_sha, {"meta": meta, "added_at": now})

    if not new_docs:
        print("No new/changed chunks to index. You're up-to-date.")
        return

    print(f"Encoding {len(new_docs)} new chunks ...")
    embs = model.encode(new_docs, batch_size=64, show_progress_bar=True)

    print("Adding to Chroma ...")
    kb.add(documents=new_docs, embeddings=embs, metadatas=new_metas, ids=new_ids)

    state.save()
    print(f"✓ Indexed {len(new_docs)} chunks into '{KB_COLLECTION}' at {VECTOR_DIR}/")

if __name__ == "__main__":
    main()
