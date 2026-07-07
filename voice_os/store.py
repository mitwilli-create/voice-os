"""Read-only access to the local JSONL chunk store.

voice_os stays independent of the ingest package; this is the one place
the runtime reads corpus/chunks/*.jsonl. Malformed lines are skipped
rather than raised: the chunk store is optional input everywhere it is
consumed, and one corrupt line must not break a query or an evaluation
run (graceful-degradation contract).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator


def iter_chunks(chunks_dir: str) -> Iterator[dict]:
    for path in sorted(glob.glob(os.path.join(chunks_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    yield chunk
