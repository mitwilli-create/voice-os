"""Queryable historical voice evolution over the chunk store.

Groups the dated history by half-year window, year, or tier and
returns axis means plus the pattern profile per group, optionally
sliced by context (audience, medium, goal). Uses ALL dated chunks
including tiers 3 and 4, unweighted, for the same reason mine/drift.py
does: temporal analysis needs the history the tier weights would
erase. Tier 4 remains zero-weight for GENERATION; this surface is
analysis only.

Stdlib-only and offline-deterministic. Design: docs/evolution.md.
"""

from __future__ import annotations

import os

from ..axes import AXES, score_text
from ..drift import window_key
from ..store import iter_chunks
from .patterns import DEFAULT_MARKERS, extract_pattern_profile

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_CHUNKS_DIR = os.path.join(REPO_ROOT, "corpus", "chunks")

GROUP_BYS = ("window", "year", "tier")


def _chunks_dir(chunks_dir: str | None) -> str:
    return chunks_dir or DEFAULT_CHUNKS_DIR


def _group_key(chunk: dict, group_by: str) -> str | None:
    if group_by == "tier":
        tier = chunk.get("tier")
        return f"tier-{tier}" if isinstance(tier, int) else None
    timestamp = chunk.get("provenance", {}).get("timestamp") or ""
    if group_by == "window":
        return window_key(timestamp)
    year = timestamp[:4]
    return year if year.isdigit() else None


def _matches_slice(chunk: dict, slice_by: dict | None) -> bool:
    if not slice_by:
        return True
    context = chunk.get("context", {})
    return all(context.get(key) == value for key, value in slice_by.items())


def evolution_timeline(
    chunks_dir: str | None = None,
    *,
    group_by: str = "window",
    slice_by: dict | None = None,
    markers: tuple[str, ...] = DEFAULT_MARKERS,
    min_chunks: int = 1,
) -> list[dict]:
    """Per-group axis means + pattern profiles, chronologically sorted.

    slice_by filters on chunk context, e.g. {"audience": "leadership"}
    or {"medium": "email", "goal": "coordinate"}. Groups with fewer
    than min_chunks chunks are dropped. Returns a JSON-safe list.
    """
    if group_by not in GROUP_BYS:
        raise ValueError(
            f"unknown group_by '{group_by}'; expected one of {GROUP_BYS}"
        )

    buckets: dict[str, list[str]] = {}
    for chunk in iter_chunks(_chunks_dir(chunks_dir)):
        text = chunk.get("text")
        if not isinstance(text, str) or not text:
            continue
        if not _matches_slice(chunk, slice_by):
            continue
        key = _group_key(chunk, group_by)
        if key is None:
            continue
        buckets.setdefault(key, []).append(text)

    groups = []
    for key in sorted(buckets):
        texts = buckets[key]
        if len(texts) < min_chunks:
            continue
        score_sum = {axis: 0.0 for axis in AXES}
        for text in texts:
            scores = score_text(text)
            for axis in AXES:
                score_sum[axis] += scores[axis]
        groups.append(
            {
                "group": key,
                "n_chunks": len(texts),
                "axis_mean": {
                    axis: round(score_sum[axis] / len(texts), 4)
                    for axis in AXES
                },
                "patterns": extract_pattern_profile(texts, markers),
            }
        )
    return groups


def tier1_texts(chunks_dir: str | None = None) -> list[str]:
    """The current-voice texts (tier 1) drift runs profile, in stable
    (hash-sorted) order so extraction order never depends on file
    layout."""
    pairs = []
    for chunk in iter_chunks(_chunks_dir(chunks_dir)):
        text = chunk.get("text")
        if not isinstance(text, str) or not text:
            continue
        if chunk.get("tier") != 1:
            continue
        pairs.append((str(chunk.get("hash", "")), text))
    pairs.sort()
    return [text for _, text in pairs]
