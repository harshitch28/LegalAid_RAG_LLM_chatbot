import hashlib, re

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def soft_clean(s: str) -> str:
    # Light cleanup; heavy cleaning was done in Step 2
    s = s.replace("\x00", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sliding_chunks(text: str, max_chars=1800, overlap=150):
    if len(text) <= max_chars:
        return [text]
    chunks, i = [], 0
    step = max_chars - overlap
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += step
    return chunks
