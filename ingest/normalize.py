"""Text normalization shared by all adapters.

Consolidates the helpers duplicated across the three legacy extractors
(ingest/legacy/): the Meta latin-1 mojibake fix, self-authorship checks,
and timestamp conversion, plus the hash normalization used for dedup.

Timestamps convert in local time, matching the legacy extractors, so tiers
computed here agree with the existing tiered corpus in Google Drive.
"""

from __future__ import annotations

import re
from datetime import datetime

# Zero-width and placeholder code points that leak out of Meta and iMessage
# exports: ZWSP, ZWNJ, ZWJ, LRM, RLM, BOM, and the object-replacement char
# iMessage uses for attachments.
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF, 0xFFFC]
)
_RUNS_OF_SPACE = re.compile(r"[ \t]+")
_RUNS_OF_BLANK = re.compile(r"\n{3,}")


def decode_meta_text(text: str) -> str:
    """Fix Meta's mojibake: UTF-8 bytes stored as latin-1 code points."""
    if not text:
        return ""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def clean_text(text: str) -> str:
    """Strip invisible characters and collapse whitespace, preserving
    paragraph breaks (they carry rhythm signal for the axes)."""
    if not text:
        return ""
    text = text.translate(_INVISIBLE)
    text = _RUNS_OF_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _RUNS_OF_BLANK.sub("\n\n", text)
    return text.strip()


def normalize_for_hash(text: str) -> str:
    """Whitespace-insensitive form so trivial reformatting between export
    drops does not register as new content."""
    return " ".join(text.split())


def is_self(name: str, identity: dict) -> bool:
    """True when a sender string matches any configured alias (names,
    usernames, emails, phone numbers). Substring match in both directions,
    same behavior as the legacy extractors."""
    if not name:
        return False
    needle = name.strip().lower()
    if not needle:
        return False
    aliases = (
        identity.get("names", [])
        + identity.get("usernames", [])
        + identity.get("emails", [])
        + identity.get("phone_numbers", [])
    )
    for alias in aliases:
        candidate = str(alias).strip().lower()
        if candidate and (candidate in needle or needle in candidate):
            return True
    return False


def ms_to_iso(ts_ms) -> str | None:
    """Millisecond epoch (Meta message timestamps) to ISO-8601."""
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def s_to_iso(ts) -> str | None:
    """Second epoch (Meta post/comment timestamps) to ISO-8601."""
    try:
        return datetime.fromtimestamp(int(ts)).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def word_count(text: str) -> int:
    return len(text.split())
