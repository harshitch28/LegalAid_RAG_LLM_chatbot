from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
NOTES_DIR = ROOT / "data" / "manual_notes"
VECTOR_DIR = ROOT / "vectorstore"
STATE_DIR = ROOT / "state"

# Collections
KB_COLLECTION = "kb_india_law"
MEMORY_COLLECTION = "conversation_memory"

# Embedding model (swap later if needed)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking safeguards (applied to very long sections)
MAX_CHARS = 1800
OVERLAP = 150


# -------- Step 4 additions --------
# Context window budget (characters) for Gemini 1.5 Flash prompts
# (Gemini handles large contexts; we still budget to keep prompts fast)
CONTEXT_CHAR_BUDGET = 24_000

# Memory scoring knobs
SIM_WEIGHT = 0.60       # semantic similarity weight
RECENCY_WEIGHT = 0.20   # freshness
ROLE_WEIGHT = 0.10      # prioritize assistant answers slightly
SESSION_WEIGHT = 0.10   # boost same-session continuity

# Recency decay (half-life in hours): newer messages score higher
RECENCY_HALFLIFE_HOURS = 24.0

# Role weights
ROLE_SCORES = {
    "assistant": 1.00,
    "user": 0.90
}

# Session continuity bonus
SAME_SESSION_BONUS = 1.00
DIFFERENT_SESSION_BONUS = 0.60

# Limits
MAX_MEMORY_CANDIDATES = 30   # pull this many from Chroma before scoring trim
TOP_MEMORY_AFTER_SCORE = 8   # final memory snippets to include in prompt
TOP_KB_SNIPPETS = 6          # legal chunks to include