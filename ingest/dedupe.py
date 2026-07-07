"""Content-hash manifest: deduplication and incremental state.

corpus/manifest.json records every chunk hash ever ingested plus a ledger
of runs. A hash seen in any earlier run is skipped, so pointing the CLI at
old exports plus a new drop appends only genuinely new content. A full
rebuild of one source (--full) drops that source's hashes and chunk file,
leaving the rest of the corpus untouched.
"""

from __future__ import annotations

import json
import os

MANIFEST_VERSION = "1.0"


class Manifest:
    def __init__(self, path: str):
        self.path = path
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "manifest_version": MANIFEST_VERSION,
                "hashes": {},
                "runs": [],
            }

    def seen(self, content_hash: str) -> bool:
        return content_hash in self.data["hashes"]

    def add(self, content_hash: str, chunk_id: str, source: str) -> None:
        self.data["hashes"][content_hash] = {"id": chunk_id, "source": source}

    def drop_source(self, source: str) -> int:
        """Remove all hashes owned by one source; returns how many."""
        doomed = [
            h for h, meta in self.data["hashes"].items()
            if meta.get("source") == source
        ]
        for h in doomed:
            del self.data["hashes"][h]
        return len(doomed)

    def record_run(self, report: dict) -> None:
        self.data["runs"].append(report)

    def count(self) -> int:
        return len(self.data["hashes"])

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
            f.write("\n")
        os.replace(tmp, self.path)
