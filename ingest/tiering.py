"""Temporal tier assignment.

Reuses the boundaries and weights from voice_os.corpus so tier definitions
can never drift between ingestion and scoring. Undated content is assigned
tier 4, which carries weight 0.0 at scoring time: nothing without a
verifiable date can influence generation.
"""

from __future__ import annotations

from voice_os.corpus import TIER_WEIGHTS, tier_for_year

__all__ = ["TIER_WEIGHTS", "tier_for_chunk_year"]


def tier_for_chunk_year(year: int | None) -> int:
    if year is None:
        return 4
    return tier_for_year(year)
