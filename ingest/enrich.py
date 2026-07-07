"""Context enrichment: channel, audience, tone signals, and goal.

Channel and audience values come from the vocabulary in
voice_os.calibration, so exported corpus headers feed the register
calibration matrix without any translation step. All inferences here are
deliberately cheap heuristics tagged inference="heuristic-v1" in the chunk;
a later model-based pass can retag without re-ingesting.
"""

from __future__ import annotations

from voice_os.calibration import AUDIENCES, CHANNELS
from voice_os.tone import tone_signals

from .schema import Context

CHANNEL_BY_SOURCE = {
    "instagram_dm": "chat",
    "messenger": "chat",
    "messenger_archived": "chat",
    "imessage": "text",
    "email_sent": "email",
    "ig_post": "social",
    "ig_story": "social",
    "ig_reel": "social",
    "ig_comment": "social",
    "fb_post": "social",
    "fb_comment": "social",
    "fb_story": "social",
    "document": "doc",
    "video_transcript": "doc",
}

MEDIUM_BY_SOURCE = {
    "instagram_dm": "dm",
    "messenger": "dm",
    "messenger_archived": "dm",
    "imessage": "sms",
    "email_sent": "email",
    "ig_post": "post",
    "ig_story": "story",
    "ig_reel": "post",
    "ig_comment": "comment",
    "fb_post": "post",
    "fb_comment": "comment",
    "fb_story": "story",
    "document": "script",
    "video_transcript": "spoken",
}

PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "aol.com", "icloud.com",
    "me.com", "mac.com", "outlook.com", "live.com", "comcast.net",
    "msn.com", "protonmail.com",
}

def infer_channel(source_type: str) -> str:
    channel = CHANNEL_BY_SOURCE.get(source_type, "doc")
    assert channel in CHANNELS
    return channel


def infer_audience(source_type: str, relationship_hint: str = "") -> str:
    """Cheap audience inference. DMs and texts default to friend-family,
    public social content to external, email by recipient domain."""
    channel = infer_channel(source_type)
    if channel in ("chat", "text"):
        audience = "friend-family"
    elif channel == "social":
        audience = "external"
    elif channel == "email":
        domain = relationship_hint.rsplit("@", 1)[-1].strip().lower()
        if domain in PERSONAL_EMAIL_DOMAINS:
            audience = "friend-family"
        else:
            audience = "peer"
    elif source_type == "video_transcript":
        audience = "external"
    else:
        audience = "peer"
    assert audience in AUDIENCES
    return audience


def infer_goal(text: str, channel: str, signals: dict) -> str:
    if signals["question_ratio"] >= 0.5:
        return "request"
    if channel in ("chat", "text"):
        return "coordinate" if "?" in text else "connect"
    if channel == "social":
        return "connect"
    if signals["word_count"] >= 120:
        return "inform"
    return "inform" if channel in ("email", "doc") else "connect"


def build_context(source_type: str, text: str, relationship_hint: str = "") -> Context:
    channel = infer_channel(source_type)
    signals = tone_signals(text)
    return Context(
        channel=channel,
        audience=infer_audience(source_type, relationship_hint),
        medium=MEDIUM_BY_SOURCE.get(source_type, "script"),
        relationship_hint=relationship_hint,
        goal=infer_goal(text, channel, signals),
        tone_signals=signals,
    )
