from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
NOTES_DIR = ROOT / "data" / "manual_notes"
VECTOR_DIR = ROOT / "vectorstore"
STATE_DIR = ROOT / "state"

# Collections
KB_COLLECTION = "kb_india_law"

# Embedding model (swap later if needed)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking safeguards (applied to very long sections)
MAX_CHARS = 1800
OVERLAP = 150
