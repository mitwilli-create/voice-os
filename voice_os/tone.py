"""Tone metrics: the emotional-calibration space parallel to the six axes.

This module is the single source of truth for tone signal computation.
ingest/enrich.py imports tone_signals from here so chunks and drafts are
measured identically. The raw signal dict keeps the exact keys the chunk
schema has always carried (including the un-normalized emoji_count and
word_count), so existing corpus/chunks/*.jsonl files stay valid.

ToneProfile mirrors the AxisProfile pattern from axes.py: per-metric mean
and spread, with a tolerance band of two standard deviations over a floor.
The floors are per metric because each metric has its own natural scale
(the 0.12 floor used for the 0..1 axes would be meaningless for
avg_sentence_words). Deviations are returned as revision-signal strings,
the same currency the QA gate and personas already exchange.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF]"
)

# The normalized metric space tone norms are mined and compared in.
TONE_METRICS = (
    "exclaim_per_100w",
    "question_ratio",
    "emoji_per_100w",
    "avg_sentence_words",
    "caps_ratio",
)

# Tolerance floors in each metric's own units.
_TOLERANCE_FLOORS = {
    "exclaim_per_100w": 0.75,
    "question_ratio": 0.10,
    "emoji_per_100w": 0.75,
    "avg_sentence_words": 3.0,
    "caps_ratio": 0.02,
}


def tone_signals(text: str) -> dict:
    """Numeric tone signals for a text. Chunk-schema compatible keys."""
    words = text.split()
    n_words = max(len(words), 1)
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    n_sentences = max(len(sentences), 1)
    letters = [c for c in text if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    return {
        "exclaim_per_100w": round(text.count("!") * 100 / n_words, 2),
        "question_ratio": round(text.count("?") / n_sentences, 2),
        "emoji_count": len(_EMOJI.findall(text)),
        "avg_sentence_words": round(n_words / n_sentences, 1),
        "caps_ratio": round(upper / max(len(letters), 1), 3),
        "word_count": len(words),
    }


def derive_metrics(signals: dict) -> dict[str, float]:
    """Normalize raw tone signals into the TONE_METRICS space.

    Accepts either a fresh tone_signals() dict or a chunk's stored
    context.tone_signals; emoji_count is length-normalized here so norms
    mined from chunks of different sizes are comparable.
    """
    n_words = max(int(signals.get("word_count", 0)), 1)
    return {
        "exclaim_per_100w": float(signals.get("exclaim_per_100w", 0.0)),
        "question_ratio": float(signals.get("question_ratio", 0.0)),
        "emoji_per_100w": round(float(signals.get("emoji_count", 0)) * 100 / n_words, 2),
        "avg_sentence_words": float(signals.get("avg_sentence_words", 0.0)),
        "caps_ratio": float(signals.get("caps_ratio", 0.0)),
    }


@dataclass
class ToneProfile:
    """Per-metric mean and spread for a communication context."""

    mean: dict[str, float]
    std: dict[str, float]

    def deviations(self, observed: dict[str, float]) -> list[str]:
        """Revision-signal strings for metrics outside the tolerance band.

        Tolerance is two standard deviations with a per-metric floor,
        mirroring AxisProfile.fidelity. Metrics missing from the profile
        are skipped, so partial mined norms degrade gracefully.
        """
        signals: list[str] = []
        for metric in TONE_METRICS:
            if metric not in self.mean:
                continue
            tolerance = max(2.0 * self.std.get(metric, 0.0), _TOLERANCE_FLOORS[metric])
            value = float(observed.get(metric, 0.0))
            deviation = value - self.mean[metric]
            if abs(deviation) > tolerance:
                direction = "reduce" if deviation > 0 else "raise"
                signals.append(
                    f"{direction} {metric}: draft {value:.2f} vs norm "
                    f"{self.mean[metric]:.2f} (tolerance {tolerance:.2f})"
                )
        return signals


def tone_guidance(profile: ToneProfile, draft: str) -> list[str]:
    """Tone deviation signals for a draft against a profile."""
    return profile.deviations(derive_metrics(tone_signals(draft)))
