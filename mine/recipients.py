"""Per-recipient axis deltas mined from relationship-labeled chunks.

The finest grain of audience patterning: how the voice shifts for a
specific conversation partner. Groups on the normalized
context.relationship_hint, with a second aggregation level by recipient
email domain for email chunks. Only groups with enough tier-weighted
support are emitted; everything else stays covered by the audience table.
"""

from __future__ import annotations

from typing import Iterable

from .aggregate import GroupStats, axis_delta
from .weights import HOLDOUT_PCT, envelope, is_train

MIN_CHUNKS = 40
MIN_WEIGHTED_WORDS = 1500.0
DELTA_CLIP = 0.35


def _normalize_hint(hint: str) -> str:
    return " ".join(hint.strip().lower().split())


def _email_domain(hint: str) -> str | None:
    if "@" in hint:
        domain = hint.rsplit("@", 1)[-1].strip().lower()
        return domain or None
    return None


def mine_recipient_deltas(
    chunks: Iterable[dict],
    min_chunks: int = MIN_CHUNKS,
    min_weighted_words: float = MIN_WEIGHTED_WORDS,
    holdout_pct: int = HOLDOUT_PCT,
) -> dict:
    """Returns the recipient_deltas artifact dict (envelope included)."""
    global_stats = GroupStats()
    recipients: dict[str, GroupStats] = {}
    domains: dict[str, GroupStats] = {}

    for chunk in chunks:
        if not is_train(chunk, holdout_pct):
            continue
        global_stats.add(chunk)
        hint = _normalize_hint(chunk.get("context", {}).get("relationship_hint", ""))
        if not hint:
            continue
        recipients.setdefault(hint, GroupStats()).add(chunk)
        domain = _email_domain(hint)
        if domain:
            domains.setdefault(domain, GroupStats()).add(chunk)

    if global_stats.n_chunks == 0:
        raise ValueError("no train-split chunks with generation weight; run ingestion first")

    global_summary = global_stats.summary()

    def emit(groups: dict[str, GroupStats]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for key, stats in sorted(groups.items()):
            summary = stats.summary()
            if summary["n_chunks"] < min_chunks:
                continue
            if summary["weighted_words"] < min_weighted_words:
                continue
            summary["axis_delta"] = axis_delta(
                summary["axis_mean"], global_summary["axis_mean"], DELTA_CLIP
            )
            out[key] = summary
        return out

    data = {
        "global": {
            "mean": global_summary["axis_mean"],
            "std": global_summary["axis_std"],
            "n_chunks": global_summary["n_chunks"],
        },
        "recipients": emit(recipients),
        "domains": emit(domains),
    }
    return envelope(
        artifact="recipient_deltas",
        miner="mine.recipients@1.0",
        params={
            "min_chunks": min_chunks,
            "min_weighted_words": min_weighted_words,
            "delta_clip": DELTA_CLIP,
        },
        data=data,
        holdout_pct=holdout_pct,
    )
