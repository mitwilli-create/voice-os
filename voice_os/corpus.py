"""Corpus ingestion and tier-weighted baseline construction.

Implements the temporal weighting model from docs/architecture.md: corpus
entries are assigned to four recency tiers and the axis baseline is a
tier-weighted average, so the baseline reflects the current voice rather
than a flat average over two decades of writing.

    Tier 1 (2024+):      weight 1.00  primary generation source
    Tier 2 (2021-2023):  weight 0.60
    Tier 3 (2015-2020):  weight 0.25
    Tier 4 (pre-2015):   weight 0.00  context only, never replicated

Corpus file format (plain text, one entry per header line):

    --- 2025-11-03 | email | leadership ---
    Entry text...

Header fields: date (YYYY-MM-DD), channel, audience. Channel and audience
are optional and default to "email" / "peer".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .axes import AXES, AxisProfile, score_text

TIER_WEIGHTS = {1: 1.0, 2: 0.6, 3: 0.25, 4: 0.0}

_HEADER = re.compile(r"^---\s*(\d{4})-\d{2}-\d{2}\s*(?:\|([^|]*))?(?:\|([^|]*))?---\s*$")


def tier_for_year(year: int) -> int:
    if year >= 2024:
        return 1
    if year >= 2021:
        return 2
    if year >= 2015:
        return 3
    return 4


@dataclass
class CorpusEntry:
    year: int
    channel: str
    audience: str
    text: str
    tier: int = field(init=False)
    scores: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.tier = tier_for_year(self.year)
        self.scores = score_text(self.text)


def parse_corpus(path: str) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    current: dict | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal current, lines
        if current is not None and any(line.strip() for line in lines):
            entries.append(
                CorpusEntry(
                    year=current["year"],
                    channel=current["channel"],
                    audience=current["audience"],
                    text="\n".join(lines).strip(),
                )
            )
        current, lines = None, []

    with open(path, encoding="utf-8") as f:
        for line in f:
            # rstrip only: a header must start at column 0, so indented
            # body lines that merely look like headers never match.
            match = _HEADER.match(line.rstrip())
            if match:
                flush()
                current = {
                    "year": int(match.group(1)),
                    "channel": (match.group(2) or "email").strip(),
                    "audience": (match.group(3) or "peer").strip(),
                }
            elif current is not None:
                lines.append(line.rstrip("\n"))
    flush()

    if not entries:
        raise ValueError(f"No corpus entries found in {path} (expected '--- YYYY-MM-DD | channel | audience ---' headers)")
    return entries


def build_baseline(entries: list[CorpusEntry]) -> AxisProfile:
    """Tier-weighted axis baseline. Tier 4 entries carry zero weight."""
    weighted = [(e, TIER_WEIGHTS[e.tier]) for e in entries if TIER_WEIGHTS[e.tier] > 0]
    if not weighted:
        raise ValueError("Corpus has no Tier 1-3 entries; baseline would be empty")

    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    total = sum(w for _, w in weighted)
    for axis in AXES:
        values = [(e.scores[axis], w) for e, w in weighted]
        m = sum(v * w for v, w in values) / total
        mean[axis] = round(m, 3)
        # tier-weighted spread, consistent with the tier-weighted mean
        variance = sum(w * (v - m) ** 2 for v, w in values) / total
        std[axis] = round(math.sqrt(variance), 3)
    return AxisProfile(mean=mean, std=std)
