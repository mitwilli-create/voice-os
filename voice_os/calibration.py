"""Register calibration engine: channel x audience x situation -> axis deltas.

Carries forward the register calibration matrix concept from
docs/architecture.md (which expressed adjustments on the legacy register
dimensions), re-expressed on the canonical six axes per the mapping in
axes.py. Deltas are applied to the corpus baseline mean to produce the
generation target for a specific communication context.
"""

from __future__ import annotations

from .axes import AXES, AxisProfile

CHANNELS = ("email", "chat", "linkedin", "text", "doc", "social")
AUDIENCES = ("leadership", "peer", "direct-report", "external", "friend-family", "networking", "job-seeking")
SITUATIONS = ("standard", "follow-up", "error-ack", "bad-news", "request", "edge-case")

# Deltas on the canonical axes. Positive editorial_register = more formal,
# positive hedging_behavior = more hedged, positive rhetorical_pace = more
# point-first, positive risk_tolerance = bolder asks.
CHANNEL_DELTAS: dict[str, dict[str, float]] = {
    "email":    {},
    "chat":     {"editorial_register": -0.20, "sentence_rhythm": -0.10},
    "linkedin": {"editorial_register": +0.10, "escalation_pattern": +0.10},
    "text":     {"editorial_register": -0.35, "sentence_rhythm": -0.20, "hedging_behavior": -0.05},
    "doc":      {"editorial_register": +0.15, "rhetorical_pace": -0.05},
    "social":   {"editorial_register": -0.25, "escalation_pattern": +0.10},
}

AUDIENCE_DELTAS: dict[str, dict[str, float]] = {
    "leadership":    {"editorial_register": +0.15, "risk_tolerance": -0.05, "rhetorical_pace": +0.05},
    "peer":          {},
    "direct-report": {"escalation_pattern": -0.05, "hedging_behavior": -0.05},
    "external":      {"editorial_register": +0.10, "hedging_behavior": +0.05},
    "friend-family": {"editorial_register": -0.30, "hedging_behavior": -0.10},
    "networking":    {"editorial_register": +0.05, "escalation_pattern": +0.05},
    "job-seeking":   {"editorial_register": +0.15, "hedging_behavior": -0.05, "rhetorical_pace": +0.05},
}

SITUATION_DELTAS: dict[str, dict[str, float]] = {
    "standard":  {},
    "follow-up": {"rhetorical_pace": +0.10},
    "error-ack": {"editorial_register": +0.10, "risk_tolerance": -0.05, "hedging_behavior": -0.05},
    "bad-news":  {"rhetorical_pace": +0.05, "hedging_behavior": -0.10},
    "request":   {"risk_tolerance": +0.05},
    "edge-case": {"hedging_behavior": -0.05},
}


def calibrate(baseline: AxisProfile, channel: str, audience: str, situation: str) -> dict[str, float]:
    """Apply register deltas to the baseline mean; returns the target profile.

    Raises ValueError on unknown context values so callers of the package API
    fail fast instead of silently receiving an uncalibrated target.
    """
    for value, valid, label in (
        (channel, CHANNELS, "channel"),
        (audience, AUDIENCES, "audience"),
        (situation, SITUATIONS, "situation"),
    ):
        if value not in valid:
            raise ValueError(f"unknown {label} '{value}'; expected one of {', '.join(valid)}")

    target = dict(baseline.mean)
    for table, key in (
        (CHANNEL_DELTAS, channel),
        (AUDIENCE_DELTAS, audience),
        (SITUATION_DELTAS, situation),
    ):
        for axis, delta in table.get(key, {}).items():
            target[axis] = round(max(0.0, min(1.0, target[axis] + delta)), 3)
    assert set(target) == set(AXES)
    return target
