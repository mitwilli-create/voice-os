"""Shared mining helpers: tier weighting, weighted stats, split filtering.

Tier weights come from voice_os.corpus.TIER_WEIGHTS, the single source of
truth for the temporal model. Miners see only the train side of the
holdout split so evaluation stays honest.
"""

from __future__ import annotations

from datetime import datetime

from voice_os.corpus import TIER_WEIGHTS
from voice_os.holdout import is_holdout

HOLDOUT_PCT = 20
SCHEMA_VERSION = "1.0"


def chunk_weight(chunk: dict) -> float:
    """Tier weight for a chunk; unknown tiers weigh zero."""
    return TIER_WEIGHTS.get(int(chunk.get("tier", 4)), 0.0)


def is_train(chunk: dict, holdout_pct: int = HOLDOUT_PCT) -> bool:
    return not is_holdout(chunk["hash"], holdout_pct)


def weighted_mean_std(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Weighted mean and population std over (value, weight) pairs."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0, 0.0
    mean = sum(v * w for v, w in pairs) / total
    variance = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    return round(mean, 4), round(variance ** 0.5, 4)


def envelope(
    artifact: str,
    miner: str,
    params: dict,
    data: dict,
    holdout_pct: int = HOLDOUT_PCT,
) -> dict:
    """Shared artifact envelope (docs/extended-model.md)."""
    return {
        "artifact": artifact,
        "version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "miner": miner,
        "train_split": {"method": "hash_prefix_mod100", "holdout_pct": holdout_pct},
        "tier_weights": {str(k): v for k, v in TIER_WEIGHTS.items()},
        "params": params,
        "data": data,
    }
