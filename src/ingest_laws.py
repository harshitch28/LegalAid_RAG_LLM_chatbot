# src/ingest_laws.py
import os, pathlib, re, json, requests
from pypdf import PdfReader
from tqdm import tqdm

# 1. Define your download sources:
LAWS = {
    "Constitution.pdf": "https://www.indiacode.nic.in/bitstream/123456789/19150/1/constitution_of_india.pdf",
    "IPC.pdf": "https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf",
    "CrPC.pdf": "https://www.indiacode.nic.in/bitstream/123456789/15272/1/the_code_of_criminal_procedure%2C_1973.pdf",
    "IEA1872.pdf":"https://www.indiacode.nic.in/bitstream/123456789/15351/1/iea_1872.pdf",
    "CodeofCivilProcedure.pdf":"https://www.indiacode.nic.in/bitstream/123456789/13813/1/the_code_of_civil_procedure%2C_1908.pdf",
    "ContractAct1872.pdf" :"https://www.indiacode.nic.in/bitstream/123456789/2187/2/A187209.pdf",
    "CompaniesAct2013.pdf":"https://www.indiacode.nic.in/bitstream/123456789/2114/5/A2013-18.pdf",
    "ITAct2000.pdf":"https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf",
    "ConsumerProtectionAct2019.pdf" : "https://www.indiacode.nic.in/bitstream/123456789/16939/1/a2019-35.pdf",
    "RTIAct2005.pdf" :"https://www.indiacode.nic.in/bitstream/123456789/19840/1/right_yo_information_act.pdf",
    "MotorVehiclesAct1988.pdf" :"https://www.indiacode.nic.in/bitstream/123456789/9460/1/a1988-59.pdf",
    "Hindu&SpecialMarriageActs.pdf":"https://www.indiacode.nic.in/bitstream/123456789/15480/1/special_marriage_act.pdf",
    "DowryProhibitionAct1961.pdf" : "https://www.indiacode.nic.in/bitstream/123456789/5556/1/dowry_prohibition.pdf",
    "DomesticViolenceAct2005.pdf" : "https://www.indiacode.nic.in/bitstream/123456789/15436/1/protection_of_women_from_domestic_violence_act%2C_2005.pdf",
    "NarcoticDrugsAct1985.pdf" : "https://www.indiacode.nic.in/bitstream/123456789/18974/1/narcotic-drugs-and-psychotropic-substances-act-1985.pdf",
    "LabourLaws.pdf":"https://www.indiacode.nic.in/bitstream/123456789/1687/1/a1988-------51.pdf" 
    # Add other PDF URLs likewise...
}

RAW = pathlib.Path("data/raw"); RAW.mkdir(exist_ok=True, parents=True)
PROC = pathlib.Path("data/processed"); PROC.mkdir(exist_ok=True, parents=True)

def download_pdfs():
    for fname, url in LAWS.items():
        dest = RAW / fname
        if dest.exists():
            print(f"✓ {fname} already downloaded")
        else:
            print(f"↓ Downloading {fname}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"✓ Saved {fname}")

def pdf_to_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def clean_text(s):
    s = s.replace("\x00", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"\n\d+\s*/\s*\d+\s*\n", "\n", s)
    return s.strip()

SEP_PATTERN = re.compile(r"(?m)^(Section|Article)\s+(\d+[A-Za-z\-]*)\b.*")

def split_sections(text):
    # Find all headings
    headings = list(SEP_PATTERN.finditer(text))
    sections = []
    for i, m in enumerate(headings):
        start = m.start()
        end = headings[i+1].start() if i+1 < len(headings) else len(text)
        sec_text = text[start:end].strip()
        sections.append(sec_text)
    return sections

def save_jsonl(sections, outf):
    with open(outf, "w", encoding="utf-8") as w:
        for sec in sections:
            w.write(json.dumps({"text": sec}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    download_pdfs()
    for pdf in tqdm(sorted(RAW.glob("*.pdf"))):
        txt = clean_text(pdf_to_text(pdf))
        secs = split_sections(txt)
        out = PROC / (pdf.stem + ".jsonl")
        save_jsonl(secs, out)
        print(f"➡ {pdf.name}: {len(secs)} sections → {out}")
