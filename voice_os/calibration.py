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
    from .mined import MinedArtifacts

# Shrinkage: a mined group delta earns full weight as its chunk support n
# grows past this constant (lam = n / (n + SHRINKAGE_N)); at zero data the
# hand-seeded tables apply unchanged.
SHRINKAGE_N = 50
MINED_DELTA_CLIP = 0.35

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


def _blended_deltas(
    hand: dict[str, float], profile: dict, global_mean: dict[str, float]
) -> dict[str, float]:
    """Shrinkage blend of a mined group profile over a hand delta table.

    The mined delta is the group's tier-weighted axis mean minus the mined
    global mean, clipped to +/- MINED_DELTA_CLIP. Blend weight
    lam = n / (n + SHRINKAGE_N) grows with the group's chunk support.
    """
    n = profile.get("n_chunks", 0)
    lam = n / (n + SHRINKAGE_N)
    group_mean = profile["axis_mean"]
    blended: dict[str, float] = {}
    for axis in AXES:
        mined_delta = group_mean[axis] - global_mean[axis]
        mined_delta = max(-MINED_DELTA_CLIP, min(MINED_DELTA_CLIP, mined_delta))
        blended[axis] = lam * mined_delta + (1.0 - lam) * hand.get(axis, 0.0)
    return blended


def calibrate_extended(
    baseline: AxisProfile,
    ctx: VoiceContext,
    mined: MinedArtifacts | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Apply the full extended delta stack to the baseline mean.

    Delta order: channel, medium, audience, recipient, situation, stakes,
    goal; each step clamps to [0, 1] exactly like calibrate(). Returns
    (target, sources) where sources records, per dimension, whether the
    applied adjustment came from mined data or a hand-seeded heuristic
    table ("absent" when the dimension was not supplied).

    When mined artifacts carry a supported profile for the medium,
    audience, or goal, the mined delta blends over the hand delta by
    shrinkage; a known recipient applies its mined delta directly.

    With a default VoiceContext and no mined artifacts this returns
    exactly what calibrate(baseline, "email", "peer", "standard") returns.
    """
    from .mined import group_profile, recipient_profile

    ctx.validate()
    context_profiles = mined.context_profiles if mined else None
    global_mean = (context_profiles or {}).get("global", {}).get("axis_mean")

    target = dict(baseline.mean)
    sources: dict[str, str] = {}

    def apply(label: str, deltas: dict[str, float], source: str) -> None:
        for axis, delta in deltas.items():
            target[axis] = round(max(0.0, min(1.0, target[axis] + delta)), 3)
        sources[label] = source

    for label, table, key, kind in (
        ("channel", CHANNEL_DELTAS, ctx.channel, None),
        ("medium", MEDIUM_DELTAS, ctx.medium, "media"),
        ("audience", AUDIENCE_DELTAS, ctx.audience, "audiences"),
        ("recipient", None, ctx.recipient, "recipient"),
        ("situation", SITUATION_DELTAS, ctx.situation, None),
        ("stakes", STAKES_DELTAS, ctx.stakes, None),
        ("goal", GOAL_DELTAS, ctx.goal, "goals"),
    ):
        if key is None:
            sources[label] = "absent"
            continue
        if kind == "recipient":
            profile = recipient_profile(mined.recipient_deltas if mined else None, key)
            if profile:
                apply(label, profile["axis_delta"], "mined")
            else:
                sources[label] = "absent"
            continue
        hand = table.get(key, {})
        profile = group_profile(context_profiles, kind, key) if kind else None
        if profile and global_mean:
            apply(label, _blended_deltas(hand, profile, global_mean), "mined")
        else:
            apply(label, hand, "heuristic")

    assert set(target) == set(AXES)
    return target, sources
