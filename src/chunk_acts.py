import re, json, pathlib
from tqdm import tqdm

PROC = pathlib.Path("data/processed")

SECTION_PATTERN = re.compile(
    r"(?m)^\s*(\d{1,3}[A-Z]?)\.\s+([A-Z][^\n]+?)\s*$"
)
# Matches lines like "356. Defamation." or "27A. XYZ"

def split_into_sections(text: str):
    sections = []
    # Find all section headings
    matches = list(SECTION_PATTERN.finditer(text))
    for i, m in enumerate(matches):
        sec_no = m.group(1).strip()
        title = m.group(2).strip().rstrip(".")
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append({
                "section_number": sec_no,
                "section_title": title,
                "text": body
            })
    return sections

def to_jsonl(sections, act_name, outpath):
    with open(outpath, "w", encoding="utf-8") as f:
        for s in sections:
            rec = {
                "act": act_name,
                "section_number": s["section_number"],
                "section_title": s["section_title"],
                "content": s["text"]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    for txt in tqdm(PROC.glob("*.txt")):
        act_name = txt.stem.replace("_", " ")
        text = txt.read_text(encoding="utf-8")
        secs = split_into_sections(text)
        out = PROC / (txt.stem + ".jsonl")
        to_jsonl(secs, act_name, out)
        print(f"{act_name}: {len(secs)} sections → {out}")
