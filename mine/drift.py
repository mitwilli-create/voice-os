"""Drift mining: windowed voice evolution over the full dated history.

Unlike the other miners this one uses ALL dated chunks, unweighted and
including tiers 3 and 4: temporal windowing is itself the recency
treatment, and evolution analysis needs the history the tier weights
would erase. The pure detection math lives in voice_os/drift.py; this
module only reads chunks and writes the drift_report artifact.
"""

from __future__ import annotations

import re
from typing import Iterable

from voice_os.axes import score_text
from voice_os.drift import (
    MIN_WINDOW_CHUNKS,
    flag_shifts,
    marker_series,
    suggest_boundaries,
    window_key,
    window_profiles,
)

from .weights import HOLDOUT_PCT, envelope, is_train

DEFAULT_MARKERS = [("yea", "yeah"), ("gonna", "going to")]


def mine_drift(
    chunks: Iterable[dict],
    markers: list[tuple[str, str]] | None = None,
    min_chunks: int = MIN_WINDOW_CHUNKS,
    holdout_pct: int = HOLDOUT_PCT,
) -> dict:
    """Returns the drift_report artifact dict (envelope included)."""
    markers = markers if markers is not None else DEFAULT_MARKERS
    forms = {form for pair in markers for form in pair}
    patterns = {
        form: re.compile(r"\b" + re.escape(form) + r"\b", re.IGNORECASE)
        for form in forms
    }

    dated_scores: list[tuple[str, dict[str, float]]] = []
    counts_by_window: dict[str, dict[str, float]] = {}
    word_totals: dict[str, float] = {}
    total = 0
    undated = 0

    for chunk in chunks:
        if not is_train(chunk, holdout_pct):
            continue
        timestamp = chunk.get("provenance", {}).get("timestamp")
        window = window_key(timestamp) if timestamp else None
        if window is None:
            undated += 1
            continue
        total += 1
        text = chunk["text"]
        dated_scores.append((timestamp, score_text(text)))

        counts = counts_by_window.setdefault(window, {})
        for form, pattern in patterns.items():
            hits = len(pattern.findall(text))
            if hits:
                counts[form] = counts.get(form, 0.0) + hits
        word_totals[window] = word_totals.get(window, 0.0) + len(text.split())

    if not dated_scores:
        raise ValueError("no dated train-split chunks; run ingestion first")

    windows = window_profiles(dated_scores, min_chunks=min_chunks)
    flags = flag_shifts(windows)
    # Marker analysis covers only the windows the axis analysis kept, so
    # crossovers are never reported from windows dropped as too sparse.
    kept = {w["window"] for w in windows}
    marker_data = marker_series(
        {k: v for k, v in counts_by_window.items() if k in kept},
        markers,
        {k: v for k, v in word_totals.items() if k in kept},
    )
    suggestions = suggest_boundaries(flags, marker_data)

    data = {
        "windows": windows,
        "flags": flags,
        "markers": marker_data,
        "suggestions": suggestions,
        "stats": {
            "dated_chunks": total,
            "undated_or_malformed_skipped": undated,
            "windows_kept": len(windows),
            "windows_dropped": len(set(counts_by_window) - kept),
        },
    }
    return envelope(
        artifact="drift_report",
        miner="mine.drift@1.0",
        params={
            "window": "half-year",
            "min_window_chunks": min_chunks,
            "markers": [list(pair) for pair in markers],
        },
        data=data,
        holdout_pct=holdout_pct,
    )
