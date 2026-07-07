"""Deterministic held-out split keyed on chunk content hash.

Miners train on the held-in side; evaluation measures on the held-out side.
Keying on the content hash (not file order or timestamps) makes the split
stable across re-ingestion runs and incremental updates.
"""

from __future__ import annotations


def is_holdout(chunk_hash: str, pct: int = 20) -> bool:
    """True when a chunk belongs to the held-out evaluation split.

    Uses the first 8 hex characters of the chunk's SHA256 content hash,
    mod 100, so the split is deterministic and roughly pct percent.
    """
    if not 0 <= pct <= 100:
        raise ValueError(f"holdout pct must be 0..100, got {pct}")
    if len(chunk_hash) < 8:
        raise ValueError(f"chunk hash too short: '{chunk_hash}'")
    prefix = chunk_hash[:8]
    try:
        bucket = int(prefix, 16)
    except ValueError:
        raise ValueError(
            f"chunk hash prefix is not hex: '{prefix}' (expected a SHA256 hex digest)"
        ) from None
    return bucket % 100 < pct
