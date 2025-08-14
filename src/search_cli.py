from src.retriever import Retriever

if __name__ == "__main__":
    r = Retriever(top_k=5)
    print("Semantic search ready. Type your query (q to quit).")
    while True:
        q = input("\nQuery: ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        hits = r.search(q, top_k=5)
        for i, h in enumerate(hits, 1):
            meta = h["metadata"]
            heading = f"{meta.get('act')} &{meta.get('section_number','')} {meta.get('section_title','')}".strip()
            print(f"\n[{i}] {heading}")
            print(f"Score: {h['score']:.3f} | Source: {meta.get('source_file')}")
            print(h["content"][:1000], "...")
