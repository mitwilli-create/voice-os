"""Shared group aggregation over chunks: axis and tone statistics.

Both the recipient and context-profile miners reduce groups of chunks to
the same statistical shape; this module holds that reduction so the two
artifacts stay structurally identical.
"""

from __future__ import annotations

from voice_os.axes import AXES, score_text
from voice_os.tone import TONE_METRICS, derive_metrics

from .weights import chunk_weight, weighted_mean_std


class GroupStats:
    """Accumulates tier-weighted axis and tone statistics for one group."""

    def __init__(self) -> None:
        self.axis_pairs: dict[str, list[tuple[float, float]]] = {a: [] for a in AXES}
        self.tone_pairs: dict[str, list[tuple[float, float]]] = {m: [] for m in TONE_METRICS}
        self.n_chunks = 0
        self.weighted_words = 0.0
        self.tiers: dict[str, int] = {}
        self.labels: dict[str, dict[str, int]] = {"audience": {}, "medium": {}}

    def add(self, chunk: dict) -> None:
        weight = chunk_weight(chunk)
        if weight <= 0:
            return
        self.n_chunks += 1
        tier = str(chunk.get("tier", 4))
        self.tiers[tier] = self.tiers.get(tier, 0) + 1

        context = chunk.get("context", {})
        for label in ("audience", "medium"):
            value = context.get(label, "")
            if value:
                counts = self.labels[label]
                counts[value] = counts.get(value, 0) + 1

        scores = score_text(chunk["text"])
        for axis in AXES:
            self.axis_pairs[axis].append((scores[axis], weight))

        signals = context.get("tone_signals") or {}
        if not signals:
            from voice_os.tone import tone_signals as _tone_signals

            signals = _tone_signals(chunk["text"])
        metrics = derive_metrics(signals)
        for metric in TONE_METRICS:
            self.tone_pairs[metric].append((metrics[metric], weight))

        word_count = signals.get("word_count") or len(chunk["text"].split())
        self.weighted_words += weight * float(word_count)

    def summary(self) -> dict:
        axis_mean, axis_std = {}, {}
        for axis in AXES:
            mean, std = weighted_mean_std(self.axis_pairs[axis])
            axis_mean[axis], axis_std[axis] = mean, std
        tone_mean, tone_std = {}, {}
        for metric in TONE_METRICS:
            mean, std = weighted_mean_std(self.tone_pairs[metric])
            tone_mean[metric], tone_std[metric] = mean, std
        majority = {
            label: max(counts, key=counts.get) if counts else ""
            for label, counts in self.labels.items()
        }
        return {
            "n_chunks": self.n_chunks,
            "weighted_words": round(self.weighted_words, 1),
            "tiers": self.tiers,
            "audience": majority["audience"],
            "medium": majority["medium"],
            "axis_mean": axis_mean,
            "axis_std": axis_std,
            "tone_mean": tone_mean,
            "tone_std": tone_std,
        }


def axis_delta(group_mean: dict, global_mean: dict, clip: float = 0.35) -> dict:
    """Per-axis delta of a group from the global mean, clipped to +/- clip."""
    return {
        axis: round(max(-clip, min(clip, group_mean[axis] - global_mean[axis])), 4)
        for axis in AXES
    }
