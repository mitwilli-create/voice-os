"""Render the provenance-rich JSONL store into the plain-text corpus format
that voice_os.corpus.parse_corpus consumes.

Traceability is preserved by the chunk store; this module is a lossy view
of it (text + date + channel + audience) shaped exactly for score.py and
pipeline.py. Undated chunks are skipped: they cannot be tiered honestly,
and tier 4 content carries zero generation weight anyway.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator


def iter_chunk_dicts(corpus_dir: str) -> Iterator[dict]:
    for path in sorted(glob.glob(os.path.join(corpus_dir, "chunks", "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def render_entry(chunk: dict) -> str | None:
    ts = chunk["provenance"].get("timestamp")
    if not ts or len(ts) < 10:
        return None
    date = ts[:10]
    channel = chunk["context"]["channel"]
    audience = chunk["context"]["audience"]
    # A body line that looks like an entry header would corrupt parsing;
    # a leading space defeats the anchored header regex.
    body = "\n".join(
        (" " + line) if line.startswith("---") else line
        for line in chunk["text"].split("\n")
    )
    return f"--- {date} | {channel} | {audience} ---\n{body}\n"


def export_corpus(
    corpus_dir: str,
    out_path: str,
    min_words: int = 8,
    channels: tuple | None = None,
) -> dict:
    """Write the rendered corpus; returns a small report dict."""
    entries = []
    skipped_undated = 0
    skipped_short = 0
    for chunk in iter_chunk_dicts(corpus_dir):
        if channels and chunk["context"]["channel"] not in channels:
            continue
        if len(chunk["text"].split()) < min_words:
            skipped_short += 1
            continue
        rendered = render_entry(chunk)
        if rendered is None:
            skipped_undated += 1
            continue
        ts = chunk["provenance"]["timestamp"]
        entries.append((ts, rendered))

    entries.sort(key=lambda pair: pair[0])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, rendered in entries:
            f.write(rendered)
            f.write("\n")

    return {
        "out_path": out_path,
        "entries": len(entries),
        "skipped_short": skipped_short,
        "skipped_undated": skipped_undated,
    }
