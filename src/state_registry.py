import json
from pathlib import Path

class StateRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"chunks": {}}  # {chunk_sha: {source_id, added_at}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def has(self, chunk_sha: str) -> bool:
        return chunk_sha in self.data["chunks"]

    def add(self, chunk_sha: str, meta: dict):
        self.data["chunks"][chunk_sha] = meta

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
