"""Extended context vocabulary: goals, stakes, media, and VoiceContext.

Channel, audience, and situation vocabularies stay in calibration.py; this
module adds the dimensions the extended model introduces and the single
dataclass that carries a full communication context through the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .calibration import AUDIENCES, CHANNELS, SITUATIONS

# Strict superset of the ingest heuristic-v1 goal set, so chunk tags map in
# with no translation. negotiate / de-escalate / set-expectations have no
# heuristic tagger yet; they are caller-supplied until a model retag pass
# (the chunk context.inference field exists for that upgrade).
GOALS = (
    "inform",
    "connect",
    "coordinate",
    "request",
    "persuade",
    "negotiate",
    "de-escalate",
    "set-expectations",
    "unknown",
)

STAKES = ("low", "routine", "high", "critical")

# Mirrors the values of ingest.enrich.MEDIUM_BY_SOURCE.
MEDIA = ("dm", "post", "comment", "story", "email", "sms", "script", "spoken")

_CRITICAL_MARKERS = re.compile(
    r"\b(resign\w*|lawyer|legal|lawsuit|termination|final notice|escalat\w*|"
    r"unacceptable)\b",
    re.IGNORECASE,
)
_HIGH_MARKERS = re.compile(
    r"\b(deadline|urgent|asap|apolog\w*|sorry for|missed|mistake|overdue|"
    r"salary|offer letter|contract)\b",
    re.IGNORECASE,
)
_LOW_MARKERS = re.compile(
    r"\b(lol|haha|no rush|whenever|no worries)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VoiceContext:
    """Full communication context for the extended model."""

    channel: str = "email"
    audience: str = "peer"
    situation: str = "standard"
    goal: str = "unknown"
    stakes: str = "routine"
    medium: str | None = None
    recipient: str | None = None

    def validate(self) -> None:
        """Fail fast on unknown context values, mirroring calibrate()."""
        for value, valid, label in (
            (self.channel, CHANNELS, "channel"),
            (self.audience, AUDIENCES, "audience"),
            (self.situation, SITUATIONS, "situation"),
            (self.goal, GOALS, "goal"),
            (self.stakes, STAKES, "stakes"),
        ):
            if value not in valid:
                raise ValueError(
                    f"unknown {label} '{value}'; expected one of {', '.join(valid)}"
                )
        if self.medium is not None and self.medium not in MEDIA:
            raise ValueError(
                f"unknown medium '{self.medium}'; expected one of {', '.join(MEDIA)}"
            )

    def as_dict(self) -> dict:
        return asdict(self)


def infer_stakes(text: str, situation: str = "standard") -> str:
    """Deterministic stakes heuristic, in the axes.py regex style.

    Critical markers win over everything; error acknowledgment and bad news
    are at least high stakes; explicit casual markers lower to low.
    """
    if _CRITICAL_MARKERS.search(text):
        return "critical"
    if situation in ("error-ack", "bad-news") or _HIGH_MARKERS.search(text):
        return "high"
    if _LOW_MARKERS.search(text):
        return "low"
    return "routine"
