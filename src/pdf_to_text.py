import re, json, pathlib
from pypdf import PdfReader
from tqdm import tqdm

RAW = pathlib.Path("data/raw")
PROC = pathlib.Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def read_pdf(path):
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def clean_text(s: str) -> str:
    s = s.replace("\x00", "")
    # Normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    # Drop common page headers/footers like page numbers (heuristic)
    s = re.sub(r"\n\d+\s*/\s*\d+\s*\n", "\n", s)
    return s.strip()

def pdf_to_txt(infile: pathlib.Path, outfile: pathlib.Path):
    raw = read_pdf(infile)
    cleaned = clean_text(raw)
    outfile.write_text(cleaned, encoding="utf-8")

if __name__ == "__main__":
    for pdf in tqdm(sorted(RAW.glob("*.pdf"))):
        txt = PROC / (pdf.stem + ".txt")
        pdf_to_txt(pdf, txt)
        print("Saved:", txt)
