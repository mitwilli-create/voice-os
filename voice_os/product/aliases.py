"""Friendly-alias normalization in front of the fail-fast calibrator.

Callers of the product layer say things like audience="boss" or
situation="high_stakes"; the calibrator and VoiceContext demand the
canonical vocabularies. This module maps every friendly form the alias
table can save, and lets VoiceContext.validate() reject the rest with
its usual actionable ValueError.

Rules (docs/callable-layer.md):
1. Lowercase, strip, underscores to hyphens.
2. Apply the alias table.
3. A situation value that names a stakes level reroutes to stakes
   (situation falls back to "standard") but only when the caller did
   not pass explicit stakes of their own.
4. Anything still unknown fails VoiceContext.validate() downstream.

Stdlib only.
"""

from __future__ import annotations

AUDIENCE_ALIASES = {
    "boss": "leadership",
    "manager": "leadership",
    "exec": "leadership",
    "executive": "leadership",
    "coworker": "peer",
    "colleague": "peer",
    "teammate": "peer",
    "report": "direct-report",
    "client": "external",
    "customer": "external",
    "vendor": "external",
    "partner": "external",
    "friend": "friend-family",
    "family": "friend-family",
    "recruiter": "job-seeking",
    "hiring-manager": "job-seeking",
    "connection": "networking",
    "network": "networking",
}

CHANNEL_ALIASES = {
    "slack": "chat",
    "teams": "chat",
    "dm": "chat",
    "im": "chat",
    "sms": "text",
    "imessage": "text",
}

SITUATION_ALIASES = {
    "followup": "follow-up",
    "apology": "error-ack",
    "error": "error-ack",
    "mistake": "error-ack",
    "badnews": "bad-news",
}

GOAL_ALIASES = {
    "deescalate": "de-escalate",
}

# Situation values that actually name a stakes level (rule 3). Keys are
# post-rule-1 forms; "high_stakes" arrives here as "high-stakes".
STAKES_FROM_SITUATION = {
    "high-stakes": "high",
    "critical-stakes": "critical",
    "low-stakes": "low",
    "routine-stakes": "routine",
    "high": "high",
    "critical": "critical",
    "low": "low",
}

# Friendly forms for an explicitly passed stakes value.
STAKES_ALIASES = {
    "high-stakes": "high",
    "critical-stakes": "critical",
    "low-stakes": "low",
    "routine-stakes": "routine",
    "normal": "routine",
    "standard": "routine",
}


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower().replace("_", "-")


def normalize_context(
    *,
    channel: str = "email",
    audience: str = "peer",
    situation: str = "standard",
    goal: str = "unknown",
    stakes: str | None = None,
    medium: str | None = None,
) -> dict:
    """Map friendly context forms to the canonical vocabularies.

    Returns canonical kwargs for VoiceContext. stakes is always resolved
    to a canonical string here: explicit value wins, a stakes-shaped
    situation reroutes, otherwise "routine". Values the table cannot
    save pass through for VoiceContext.validate() to reject.
    """
    channel = _clean(channel) or "email"
    audience = _clean(audience) or "peer"
    situation = _clean(situation) or "standard"
    goal = _clean(goal) or "unknown"
    stakes_clean = _clean(stakes)
    medium = _clean(medium)

    channel = CHANNEL_ALIASES.get(channel, channel)
    audience = AUDIENCE_ALIASES.get(audience, audience)
    situation = SITUATION_ALIASES.get(situation, situation)
    goal = GOAL_ALIASES.get(goal, goal)

    if stakes_clean is not None:
        stakes_final = STAKES_ALIASES.get(stakes_clean, stakes_clean)
    elif situation in STAKES_FROM_SITUATION:
        stakes_final = STAKES_FROM_SITUATION[situation]
        situation = "standard"
    else:
        stakes_final = "routine"

    return {
        "channel": channel,
        "audience": audience,
        "situation": situation,
        "goal": goal,
        "stakes": stakes_final,
        "medium": medium,
    }
