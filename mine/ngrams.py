"""n-gram anti-pattern mining: frequency diffs against generic LLM output.

Moves the banned list from hand-curated toward statistically mined
evidence. Counts 1- to 4-grams in the self corpus (train split,
tier-weighted) and in a contrast corpus of generic LLM output, normalizes
both to frequency per million tokens, and scores each n-gram with smoothed
log-odds. n-grams far more common in LLM output than in the self corpus
are emitted as banned candidates with their evidence attached, so every
mined ban is auditable.

A committed never-ban guard list keeps statistics from ever banning
common function words, no matter how skewed a small contrast corpus is.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

from .weights import HOLDOUT_PCT, chunk_weight, envelope, is_train

N_MAX = 4
MIN_CONTRAST_COUNT = 5
MIN_LOG_ODDS = 2.0
SMOOTHING = 0.5  # per million tokens
# An anti-pattern is a phrase the self corpus effectively never uses;
# anything the author writes more often than this stays allowed no matter
# how much LLMs also use it.
MAX_SELF_PER_MILLION = 20.0

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, internal apostrophes kept."""
    return _TOKEN.findall(text.lower())


def _count_ngrams(
    token_lists: Iterable[tuple[list[str], float]], n_max: int
) -> tuple[dict[str, float], float]:
    """Weighted n-gram counts and the weighted token total."""
    counts: dict[str, float] = {}
    total_tokens = 0.0
    for tokens, weight in token_lists:
        total_tokens += weight * len(tokens)
        for n in range(1, n_max + 1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i : i + n])
                counts[gram] = counts.get(gram, 0.0) + weight
    return counts, total_tokens


def load_never_ban(path: str) -> set[str]:
    """Guard list of unigrams that statistics may never ban."""
    words: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.add(line)
    return words


def mine_ngram_diffs(
    self_chunks: Iterable[dict],
    contrast_texts: list[str],
    never_ban: set[str] | None = None,
    n_max: int = N_MAX,
    min_contrast_count: int = MIN_CONTRAST_COUNT,
    min_log_odds: float = MIN_LOG_ODDS,
    max_self_per_million: float = MAX_SELF_PER_MILLION,
    holdout_pct: int = HOLDOUT_PCT,
) -> dict:
    """Returns the ngram_banned artifact dict (envelope included)."""
    if not contrast_texts:
        raise ValueError("contrast corpus is empty; nothing to diff against")
    never_ban = never_ban or set()

    self_counts, self_tokens = _count_ngrams(
        (
            (tokenize(chunk["text"]), chunk_weight(chunk))
            for chunk in self_chunks
            if is_train(chunk, holdout_pct) and chunk_weight(chunk) > 0
        ),
        n_max,
    )
    contrast_counts, contrast_tokens = _count_ngrams(
        ((tokenize(text), 1.0) for text in contrast_texts), n_max
    )
    if self_tokens <= 0:
        raise ValueError("no weighted train-split self tokens; run ingestion first")
    if contrast_tokens <= 0:
        raise ValueError("contrast corpus has no tokens")

    banned = []
    for gram, raw_count in contrast_counts.items():
        if raw_count < min_contrast_count:
            continue
        n = gram.count(" ") + 1
        tokens = gram.split(" ")
        # Guard unigrams on the list, and composites made entirely of
        # guarded words ("that our", "as we"): a small contrast corpus
        # inflates ratios on such ordinary function-word sequences.
        if never_ban and all(t in never_ban for t in tokens):
            continue
        per_million_contrast = raw_count * 1_000_000 / contrast_tokens
        per_million_self = self_counts.get(gram, 0.0) * 1_000_000 / self_tokens
        if per_million_self > max_self_per_million:
            continue
        log_odds = math.log(
            (per_million_contrast + SMOOTHING) / (per_million_self + SMOOTHING)
        )
        if log_odds < min_log_odds:
            continue
        banned.append(
            {
                "ngram": gram,
                "n": n,
                "contrast_count": int(raw_count),
                "per_million_contrast": round(per_million_contrast, 2),
                "per_million_self": round(per_million_self, 2),
                "log_odds": round(log_odds, 3),
            }
        )

    # Most distinctive first; drop sub-grams of an already banned longer
    # phrase so the list stays readable ("delve into" not also "delve").
    banned.sort(key=lambda e: (-e["log_odds"], e["ngram"]))
    kept: list[dict] = []
    phrases: list[str] = []
    for entry in sorted(banned, key=lambda e: -e["n"]):
        padded = f" {entry['ngram']} "
        if any(padded in f" {p} " for p in phrases):
            continue
        phrases.append(entry["ngram"])
        kept.append(entry)
    kept.sort(key=lambda e: (-e["log_odds"], e["ngram"]))

    data = {
        "banned": kept,
        "stats": {
            "self_tokens": round(self_tokens, 1),
            "contrast_tokens": round(contrast_tokens, 1),
            "contrast_passages": len(contrast_texts),
            "candidates_scored": len(contrast_counts),
        },
    }
    return envelope(
        artifact="ngram_banned",
        miner="mine.ngrams@1.0",
        params={
            "n_max": n_max,
            "min_contrast_count": min_contrast_count,
            "min_log_odds": min_log_odds,
            "max_self_per_million": max_self_per_million,
            "smoothing_per_million": SMOOTHING,
        },
        data=data,
        holdout_pct=holdout_pct,
    )
