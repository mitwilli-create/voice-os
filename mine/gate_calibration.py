"""Gate calibration: what the author's own text scores per cell.

Produces the gate_calibration artifact: per-(channel, audience) fidelity
percentiles of real held-in text measured against the same extended
calibrated targets the QA gate uses. A gate threshold above what real
text scores in a cell rejects drafts for being no more target-conformant
than the author himself (docs/live-alignment.md).

Follows the mine/ invariant: trains on the held-in split only, so the
held-out evaluation stays honest and the gate never learns from the
chunks the scorecard measures.
"""

from __future__ import annotations

import math
import os
from typing import Iterable

from voice_os import load_corpus
from voice_os.axes import AxisProfile, score_text
from voice_os.calibration import calibrate_extended
from voice_os.contexts import VoiceContext
from voice_os.mined import load_artifacts

from .weights import HOLDOUT_PCT, envelope, is_train

MINER = "mine.gate_calibration@1.0"

# A cell needs enough real chunks for its percentiles to mean anything;
# below this the runtime keeps the hand default threshold.
MIN_CHUNKS = 50

PERCENTILES = (25, 40, 50)


def _percentile(sorted_values: list[float], pct: int) -> float:
    """Nearest-rank percentile over an ascending list."""
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return round(sorted_values[rank - 1], 4)


def _chunk_context(chunk: dict) -> VoiceContext | None:
    """Context for a chunk, or None when its tags fail validation
    (mirrors the tolerant reader in voice_os.eval)."""
    context = chunk.get("context", {})
    try:
        ctx = VoiceContext(
            channel=context.get("channel", "email"),
            audience=context.get("audience", "peer"),
            goal=context.get("goal", "unknown"),
            medium=context.get("medium") or None,
        )
        ctx.validate()
    except (ValueError, TypeError):
        return None
    return ctx


def mine_gate_calibration(
    chunks: Iterable[dict],
    corpus_path: str = os.path.join("corpus", "voice_corpus.txt"),
    mined_dir: str | None = os.path.join("corpus", "mined"),
    min_chunks: int = MIN_CHUNKS,
    holdout_pct: int = HOLDOUT_PCT,
) -> dict:
    """Returns the gate_calibration artifact dict (envelope included)."""
    baseline = load_corpus(corpus_path)
    mined = load_artifacts(mined_dir)

    target_cache: dict[tuple, dict] = {}
    cell_values: dict[str, list[float]] = {}

    for chunk in chunks:
        if not is_train(chunk, holdout_pct):
            continue
        ctx = _chunk_context(chunk)
        if ctx is None:
            continue
        key = (ctx.channel, ctx.audience, ctx.goal, ctx.medium)
        if key not in target_cache:
            target, _ = calibrate_extended(baseline, ctx, mined=mined)
            target_cache[key] = target
        profile = AxisProfile(mean=target_cache[key], std=baseline.std)
        fidelity, _ = profile.fidelity(score_text(chunk["text"]))
        cell = f"{ctx.channel}|{ctx.audience}"
        cell_values.setdefault(cell, []).append(fidelity)

    cells = {}
    for cell, values in sorted(cell_values.items()):
        if len(values) < min_chunks:
            continue
        values.sort()
        cells[cell] = {
            "n": len(values),
            "mean": round(sum(values) / len(values), 4),
            **{f"p{pct}": _percentile(values, pct) for pct in PERCENTILES},
        }

    return envelope(
        artifact="gate_calibration",
        miner=MINER,
        params={"min_chunks": min_chunks, "percentiles": list(PERCENTILES)},
        data={"cells": cells},
        holdout_pct=holdout_pct,
    )
