import os, pathlib, requests

RAW = pathlib.Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

URLS = {
  "BNS_2023.pdf": "https://www.indiacode.nic.in/bitstream/123456789/20062/1/a2023-45.pdf",
  "BNSS_2023.pdf": "https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf",
  "BSA_2023.pdf": "https://www.indiacode.nic.in/bitstream/123456789/20063/1/a2023-47.pdf",
}

for fname, url in URLS.items():
    dest = RAW / fname
    if dest.exists():
        print(f"✓ Exists: {dest}")
        continue
    print(f"↓ Downloading {fname} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"✓ Saved: {dest}")
