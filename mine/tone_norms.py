"""Per-context axis and tone profiles mined from the chunk store.

Produces the context_profiles artifact: tier-weighted axis and tone
statistics per audience, per medium, per goal, and per (audience, medium)
pair. The runtime blends these mined profiles over the hand-seeded delta
tables (docs/extended-model.md).
"""

from __future__ import annotations

from typing import Iterable

from .aggregate import GroupStats
from .weights import HOLDOUT_PCT, envelope, is_train

MIN_CHUNKS = 40


def mine_context_profiles(
    chunks: Iterable[dict],
    min_chunks: int = MIN_CHUNKS,
    holdout_pct: int = HOLDOUT_PCT,
) -> dict:
    """Returns the context_profiles artifact dict (envelope included)."""
    global_stats = GroupStats()
    groups: dict[str, dict[str, GroupStats]] = {
        "audiences": {},
        "media": {},
        "goals": {},
        "pairs": {},
    }

    for chunk in chunks:
        if not is_train(chunk, holdout_pct):
            continue
        global_stats.add(chunk)
        context = chunk.get("context", {})
        audience = context.get("audience", "")
        medium = context.get("medium", "")
        goal = context.get("goal", "")
        if audience:
            groups["audiences"].setdefault(audience, GroupStats()).add(chunk)
        if medium:
            groups["media"].setdefault(medium, GroupStats()).add(chunk)
        if goal and goal != "unknown":
            groups["goals"].setdefault(goal, GroupStats()).add(chunk)
        if audience and medium:
            groups["pairs"].setdefault(f"{audience}|{medium}", GroupStats()).add(chunk)

    if global_stats.n_chunks == 0:
        raise ValueError("no train-split chunks with generation weight; run ingestion first")

    global_summary = global_stats.summary()
    data: dict = {
        "global": {
            "axis_mean": global_summary["axis_mean"],
            "axis_std": global_summary["axis_std"],
            "tone_mean": global_summary["tone_mean"],
            "tone_std": global_summary["tone_std"],
            "n_chunks": global_summary["n_chunks"],
        },
    }
    for kind, members in groups.items():
        data[kind] = {
            key: stats.summary()
            for key, stats in sorted(members.items())
            if stats.n_chunks >= min_chunks
        }
    return envelope(
        artifact="context_profiles",
        miner="mine.tone_norms@1.0",
        params={"min_chunks": min_chunks},
        data=data,
        holdout_pct=holdout_pct,
    )
