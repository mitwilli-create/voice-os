"""Register calibration engine: channel x audience x situation -> axis deltas.

Carries forward the register calibration matrix concept from
docs/architecture.md (which expressed adjustments on the legacy register
dimensions), re-expressed on the canonical six axes per the mapping in
axes.py. Deltas are applied to the corpus baseline mean to produce the
generation target for a specific communication context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .axes import AXES, AxisProfile

if TYPE_CHECKING:
    from .contexts import VoiceContext

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

# Extended-model delta tables (docs/extended-model.md). Hand-seeded values;
# mined per-context profiles blend over these once mining artifacts exist.
GOAL_DELTAS: dict[str, dict[str, float]] = {
    "inform":           {},
    "connect":          {"editorial_register": -0.05, "escalation_pattern": +0.05},
    "coordinate":       {"rhetorical_pace": +0.05},
    "request":          {"risk_tolerance": +0.05, "rhetorical_pace": +0.05},
    "persuade":         {"risk_tolerance": +0.10, "escalation_pattern": +0.10},
    "negotiate":        {"risk_tolerance": +0.10, "hedging_behavior": -0.05, "editorial_register": +0.05},
    "de-escalate":      {"escalation_pattern": -0.15, "risk_tolerance": -0.05, "hedging_behavior": +0.05},
    "set-expectations": {"rhetorical_pace": +0.10, "hedging_behavior": -0.10},
    "unknown":          {},
}

STAKES_DELTAS: dict[str, dict[str, float]] = {
    "low":      {"editorial_register": -0.05},
    "routine":  {},
    "high":     {"editorial_register": +0.05, "hedging_behavior": -0.05},
    "critical": {"editorial_register": +0.10, "hedging_behavior": -0.10, "escalation_pattern": -0.10},
}

# Medium is a finer grain than channel (a story and a post are both channel
# "social"); these deltas apply on top of the channel delta.
MEDIUM_DELTAS: dict[str, dict[str, float]] = {
    "dm":      {},
    "post":    {"escalation_pattern": +0.05},
    "comment": {"sentence_rhythm": -0.05},
    "story":   {"editorial_register": -0.10},
    "email":   {},
    "sms":     {},
    "script":  {"editorial_register": +0.05, "sentence_rhythm": +0.05},
    "spoken":  {"editorial_register": -0.10, "sentence_rhythm": +0.10},
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


def calibrate_extended(
    baseline: AxisProfile, ctx: VoiceContext
) -> tuple[dict[str, float], dict[str, str]]:
    """Apply the full extended delta stack to the baseline mean.

    Delta order: channel, medium, audience, situation, stakes, goal; each
    step clamps to [0, 1] exactly like calibrate(). Returns (target,
    sources) where sources records, per dimension, whether the applied
    adjustment came from mined data or a hand-seeded heuristic table
    ("absent" when the dimension was not supplied). Mined blending lands
    with the mining layer; until then every supplied dimension reports
    "heuristic".

    With a default VoiceContext this returns exactly what
    calibrate(baseline, "email", "peer", "standard") returns.
    """
    ctx.validate()
    target = dict(baseline.mean)
    sources: dict[str, str] = {}
    for label, table, key in (
        ("channel", CHANNEL_DELTAS, ctx.channel),
        ("medium", MEDIUM_DELTAS, ctx.medium),
        ("audience", AUDIENCE_DELTAS, ctx.audience),
        ("situation", SITUATION_DELTAS, ctx.situation),
        ("stakes", STAKES_DELTAS, ctx.stakes),
        ("goal", GOAL_DELTAS, ctx.goal),
    ):
        if key is None:
            sources[label] = "absent"
            continue
        for axis, delta in table.get(key, {}).items():
            target[axis] = round(max(0.0, min(1.0, target[axis] + delta)), 3)
        sources[label] = "heuristic"
    assert set(target) == set(AXES)
    return target, sources
