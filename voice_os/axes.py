"""Canonical six-axis voice model.

These six axes are the single canonical stylistic dimension set for Voice OS.
They are used both for corpus baseline scoring (score.py) and as the target
space for register calibration deltas (calibration.py).

Historical note: docs/architecture.md previously described a second set of six
"register dimensions" (directness, structure, warmth, formality, precision,
assertiveness) used by the claude.ai Projects deployment. Those are now
re-expressed as deltas on the canonical axes via this mapping:

    directness    -> rhetorical_pace
    assertiveness -> risk_tolerance
    structure     -> sentence_rhythm
    warmth        -> escalation_pattern
    precision     -> hedging_behavior (inverse: more precision, less hedging)
    formality     -> editorial_register

Each axis is scored 0.0 to 1.0 by a deterministic heuristic. The heuristics
are intentionally simple and inspectable; when an Anthropic API key is
available the pipeline layers Claude-based judgment on top (see personas.py),
but the deterministic layer always runs so results are reproducible offline.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

AXES = (
    "rhetorical_pace",      # how quickly the writing gets to the point
    "risk_tolerance",       # willingness to make direct claims and asks
    "sentence_rhythm",      # variation in sentence length (cadence)
    "escalation_pattern",   # how intensity builds across the text
    "hedging_behavior",     # density of softeners and qualifiers
    "editorial_register",   # formality and polish
)

_HEDGES = re.compile(
    r"\b(maybe|perhaps|possibly|might|i think|i guess|i feel like|sort of|"
    r"kind of|just|a bit|somewhat|arguably|it seems|we may want to)\b",
    re.IGNORECASE,
)
_SOFTENERS = re.compile(
    r"\b(might|could|possibly|perhaps|if it makes sense|no worries if not|"
    r"whenever you get a chance|i was wondering)\b",
    re.IGNORECASE,
)
_INTENSIFIERS = re.compile(
    r"(!|\b(really|very|critical|must|now|urgent|huge|massive|absolutely)\b)",
    re.IGNORECASE,
)
_FILLER_OPENERS = re.compile(
    r"^(i hope this (email )?finds you well|i wanted to (reach out|touch base|"
    r"circle back)|just checking in|i hope you('re| are) (doing )?well|"
    r"per my last email)",
    re.IGNORECASE,
)
_CONTRACTIONS = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b")
_CASUAL = re.compile(r"\b(lol|gonna|wanna|yeah|haha|omg|btw|tbh)\b", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _per_100_words(count: int, words: int) -> float:
    return count / max(words, 1) * 100.0


def score_text(text: str) -> dict[str, float]:
    """Score a text on all six canonical axes. Deterministic."""
    sentences = _sentences(text)
    words = len(text.split())
    if not sentences or words == 0:
        return {axis: 0.5 for axis in AXES}

    # rhetorical_pace: filler openers and long wind-ups lower the score
    opener_penalty = 0.45 if _FILLER_OPENERS.search(sentences[0]) else 0.0
    first_sentence_ratio = len(sentences[0].split()) / words
    pace = _clamp(0.9 - opener_penalty - max(0.0, first_sentence_ratio - 0.5))

    # risk_tolerance: softener density lowers the score
    softeners = len(_SOFTENERS.findall(text))
    risk = _clamp(1.0 - _per_100_words(softeners, words) / 6.0)

    # sentence_rhythm: coefficient of variation of sentence lengths
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) > 1 and statistics.mean(lengths) > 0:
        cv = statistics.pstdev(lengths) / statistics.mean(lengths)
        rhythm = _clamp(cv / 0.9)
    else:
        rhythm = 0.3

    # escalation_pattern: intensity in the second half relative to the first
    mid = max(1, len(sentences) // 2)
    first_half = " ".join(sentences[:mid])
    second_half = " ".join(sentences[mid:]) or first_half
    i1 = _per_100_words(len(_INTENSIFIERS.findall(first_half)), len(first_half.split()))
    i2 = _per_100_words(len(_INTENSIFIERS.findall(second_half)), len(second_half.split()))
    escalation = _clamp(0.5 + (i2 - i1) / 10.0)

    # hedging_behavior: hedge density (higher score = more hedging)
    hedges = len(_HEDGES.findall(text))
    hedging = _clamp(_per_100_words(hedges, words) / 8.0)

    # editorial_register: formality composite
    contraction_rate = _per_100_words(len(_CONTRACTIONS.findall(text)), words)
    casual_rate = _per_100_words(len(_CASUAL.findall(text)), words)
    avg_word_len = sum(len(w) for w in text.split()) / words
    register = _clamp(
        0.5
        + (avg_word_len - 4.2) / 4.0
        - contraction_rate / 12.0
        - casual_rate / 4.0
    )

    return {
        "rhetorical_pace": round(pace, 3),
        "risk_tolerance": round(risk, 3),
        "sentence_rhythm": round(rhythm, 3),
        "escalation_pattern": round(escalation, 3),
        "hedging_behavior": round(hedging, 3),
        "editorial_register": round(register, 3),
    }


@dataclass
class AxisProfile:
    """Per-axis mean and spread, produced from a scored corpus."""

    mean: dict[str, float]
    std: dict[str, float]

    def fidelity(self, scores: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Overall fidelity of a draft's scores against this profile.

        Per-axis closeness = 1 - min(1, |draft - mean| / tolerance) where
        tolerance is two standard deviations with a 0.12 floor, so a corpus
        with natural variation on an axis is forgiving on that axis.
        Returns (overall 0..1, per-axis closeness).
        """
        per_axis: dict[str, float] = {}
        for axis in AXES:
            tolerance = max(2.0 * self.std.get(axis, 0.0), 0.12)
            deviation = abs(scores[axis] - self.mean[axis])
            per_axis[axis] = round(1.0 - min(1.0, deviation / tolerance), 3)
        overall = round(sum(per_axis.values()) / len(AXES), 3)
        return overall, per_axis
