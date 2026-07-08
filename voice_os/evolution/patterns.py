"""Pure pattern extraction and baseline diffing math.

The pattern profile is the unit of evolution tracking: greeting and
sign-off distributions matched against FIXED lexicons (so names and
personal content never enter stored artifacts), sentence-length shape,
and per-100-word marker frequencies. Everything here is stdlib-only,
RNG-free, and iterates in sorted order, per the determinism contract
(docs/determinism.md, Rules for new modules).

Design: docs/evolution.md.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

# Longest-first so "good morning" wins over "good", "hey there" over
# "hey". Unlisted openers/closers bucket to "other": the lexicon is the
# privacy boundary that keeps names out of stored baselines.
GREETING_LEXICON = (
    "good morning",
    "good afternoon",
    "good evening",
    "hey there",
    "hiya",
    "howdy",
    "hello",
    "hey",
    "hi",
    "yo",
)
SIGNOFF_LEXICON = (
    "thanks so much",
    "thank you",
    "talk soon",
    "take care",
    "thanks",
    "cheers",
    "best",
    "later",
    "ttyl",
)
# Single forms (not pairs like mine/drift.py): evolution tracks each
# form's frequency; crossover pairs stay in the axis-drift layer.
DEFAULT_MARKERS = ("yea", "yeah", "gonna", "going to", "lol", "haha")

# Diff thresholds, documented constants per docs/evolution.md:
# a family key needs MIN_SUPPORT raw occurrences on the side asserting
# presence; NEAR_ZERO is the "effectively absent" floor for shares and
# per-100w frequencies; a shift needs both a relative and an absolute
# floor so tiny bases cannot inflate relative change into noise flags.
MIN_SUPPORT = 5
NEAR_ZERO = 0.01
MIN_RELATIVE_CHANGE = 0.5
MIN_ABS_CHANGE = 0.02
SENTENCE_MEAN_TOLERANCE_WORDS = 2.0

SHORT_CHUNK_WORDS = 10

_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")
_WORD = re.compile(r"[A-Za-z0-9']+")


def _classify_line(line: str, lexicon: tuple[str, ...]) -> str:
    cleaned = " ".join(_WORD.findall(line.lower()))
    if not cleaned:
        return "other"
    for form in lexicon:
        if cleaned == form or cleaned.startswith(form + " "):
            return form
    return "other"


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def _last_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return ""


def _nearest_rank(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    rank = max(0, math.ceil(q * len(sorted_values)) - 1)
    return float(sorted_values[rank])


def extract_pattern_profile(
    texts: Iterable[str],
    markers: tuple[str, ...] = DEFAULT_MARKERS,
) -> dict:
    """The pattern profile of a text collection. Pure; JSON-safe.

    Shares are per-chunk (greetings/signoffs) or per-100-words
    (markers, exclamations); raw counts ride along so diffing can
    apply support floors. All floats rounded to 4 places, all dict
    keys emitted in sorted order.
    """
    marker_patterns = {
        form: re.compile(r"\b" + re.escape(form) + r"\b", re.IGNORECASE)
        for form in markers
    }
    n_chunks = 0
    n_words = 0
    short_chunks = 0
    exclamations = 0
    sentence_lengths: list[int] = []
    greeting_counts: dict[str, int] = {}
    signoff_counts: dict[str, int] = {}
    marker_counts = {form: 0 for form in markers}

    for text in texts:
        n_chunks += 1
        words = len(_WORD.findall(text))
        n_words += words
        if words < SHORT_CHUNK_WORDS:
            short_chunks += 1
        exclamations += text.count("!")

        greeting = _classify_line(_first_line(text), GREETING_LEXICON)
        greeting_counts[greeting] = greeting_counts.get(greeting, 0) + 1
        signoff = _classify_line(_last_line(text), SIGNOFF_LEXICON)
        signoff_counts[signoff] = signoff_counts.get(signoff, 0) + 1

        for form, pattern in marker_patterns.items():
            marker_counts[form] += len(pattern.findall(text))

        for sentence in _SENTENCE_SPLIT.split(text):
            length = len(_WORD.findall(sentence))
            if length:
                sentence_lengths.append(length)

    chunks = max(n_chunks, 1)
    words_total = max(n_words, 1)
    sentence_lengths.sort()
    mean_len = (
        sum(sentence_lengths) / len(sentence_lengths)
        if sentence_lengths
        else 0.0
    )
    return {
        "n_chunks": n_chunks,
        "n_words": n_words,
        "n_sentences": len(sentence_lengths),
        "greetings": {
            form: round(count / chunks, 4)
            for form, count in sorted(greeting_counts.items())
        },
        "greeting_counts": dict(sorted(greeting_counts.items())),
        "signoffs": {
            form: round(count / chunks, 4)
            for form, count in sorted(signoff_counts.items())
        },
        "signoff_counts": dict(sorted(signoff_counts.items())),
        "markers_per_100w": {
            form: round(count * 100 / words_total, 4)
            for form, count in sorted(marker_counts.items())
        },
        "marker_counts": dict(sorted(marker_counts.items())),
        "exclamations_per_100w": round(exclamations * 100 / words_total, 4),
        "sentence_length": {
            "mean": round(mean_len, 4),
            "p50": _nearest_rank(sentence_lengths, 0.50),
            "p90": _nearest_rank(sentence_lengths, 0.90),
            "short_rate": round(short_chunks / chunks, 4),
        },
    }


_FAMILIES = (
    # (family name, value field, count field)
    ("greetings", "greetings", "greeting_counts"),
    ("signoffs", "signoffs", "signoff_counts"),
    ("markers", "markers_per_100w", "marker_counts"),
)


def diff_profiles(
    baseline: dict,
    current: dict,
    *,
    min_support: int = MIN_SUPPORT,
    near_zero: float = NEAR_ZERO,
    min_relative: float = MIN_RELATIVE_CHANGE,
    min_abs: float = MIN_ABS_CHANGE,
) -> dict:
    """Pattern drift between two profiles. Pure; deterministic.

    emerging: effectively absent in the baseline, present with real
    support now. fading: the reverse. shifted: present on both sides
    with a change that clears both the relative and absolute floors.
    The "other" bucket is skipped (it aggregates unlisted forms and
    would flag as a pattern when it is really lexicon coverage).
    """
    emerging: list[dict] = []
    fading: list[dict] = []
    shifted: list[dict] = []

    for family, value_field, count_field in _FAMILIES:
        base_values = baseline.get(value_field, {})
        cur_values = current.get(value_field, {})
        base_counts = baseline.get(count_field, {})
        cur_counts = current.get(count_field, {})
        for key in sorted(set(base_values) | set(cur_values)):
            if key == "other":
                continue
            b = base_values.get(key, 0.0)
            c = cur_values.get(key, 0.0)
            entry = {
                "family": family,
                "key": key,
                "baseline": round(b, 4),
                "current": round(c, 4),
            }
            if b <= near_zero and c > near_zero:
                if cur_counts.get(key, 0) >= min_support:
                    emerging.append(entry)
            elif c <= near_zero and b > near_zero:
                if base_counts.get(key, 0) >= min_support:
                    fading.append(entry)
            elif b > 0 and c > 0:
                delta = abs(c - b)
                if delta >= min_abs and delta / max(b, near_zero) >= min_relative:
                    if max(base_counts.get(key, 0), cur_counts.get(key, 0)) >= min_support:
                        shifted.append(entry)

    base_mean = baseline.get("sentence_length", {}).get("mean", 0.0)
    cur_mean = current.get("sentence_length", {}).get("mean", 0.0)
    sentence_shift = None
    if abs(cur_mean - base_mean) > SENTENCE_MEAN_TOLERANCE_WORDS:
        sentence_shift = {
            "baseline_mean": round(base_mean, 4),
            "current_mean": round(cur_mean, 4),
            "delta": round(cur_mean - base_mean, 4),
        }

    flags = []
    for entry in emerging:
        flags.append(
            f"emerging {entry['family'][:-1] if entry['family'].endswith('s') else entry['family']}"
            f" '{entry['key']}' ({entry['baseline']} -> {entry['current']})"
        )
    for entry in fading:
        flags.append(
            f"fading {entry['family'][:-1] if entry['family'].endswith('s') else entry['family']}"
            f" '{entry['key']}' ({entry['baseline']} -> {entry['current']})"
        )
    if sentence_shift:
        flags.append(
            "sentence length shifted "
            f"({sentence_shift['baseline_mean']} -> "
            f"{sentence_shift['current_mean']} words/sentence)"
        )

    return {
        "emerging": emerging,
        "fading": fading,
        "shifted": shifted,
        "sentence_shift": sentence_shift,
        "flags": flags,
        "stats": {
            "baseline_chunks": baseline.get("n_chunks", 0),
            "current_chunks": current.get("n_chunks", 0),
        },
    }
