import os, json, glob, pathlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = pathlib.Path("data/processed")
NOTES_DIR = pathlib.Path("data/manual_notes")
VSTORE_DIR = "vectorstore"

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def load_notes():
    records = []
    for p in NOTES_DIR.glob("*.md"):
        text = p.read_text(encoding="utf-8")
        records.append({
            "act": "Procedural Guide",
            "section_number": "",
            "section_title": p.stem.replace("_", " ").title(),
            "content": text
        })
    return records

if __name__ == "__main__":
    model = SentenceTransformer(EMB_MODEL)

    client = chromadb.PersistentClient(path=VSTORE_DIR)
    # Knowledge base collection (Acts + guides)
    kb = client.get_or_create_collection(
        name="kb_india_law",
        metadata={"hnsw:space": "cosine"}
    )

    docs, ids, metas = [], [], []
    uid = 0

    # Acts
    for jf in tqdm(sorted(DATA_DIR.glob("*.jsonl"))):
        act_records = list(load_jsonl(jf))
        for r in act_records:
            ids.append(f"doc-{uid}"); uid += 1
            docs.append(r["content"])
            metas.append({
                "act": r["act"],
                "section_number": r["section_number"],
                "section_title": r["section_title"],
                "source_file": jf.name
            })

    # Procedural notes
    for r in load_notes():
        ids.append(f"doc-{uid}"); uid += 1
        docs.append(r["content"])
        metas.append({
            "act": r["act"],
            "section_number": r["section_number"],
            "section_title": r["section_title"],
            "source_file": "manual_note"
        })

    print(f"Encoding {len(docs)} chunks ...")
    embs = model.encode(docs, batch_size=64, show_progress_bar=True)

    print("Adding to Chroma ...")
    kb.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
    print(f"✓ Indexed {len(docs)} chunks into collection 'kb_india_law' at {VSTORE_DIR}/")
