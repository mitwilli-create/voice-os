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


# A sender string that is plausibly a phone number: an optional +, then
# digits with common formatting (spaces, dashes, dots, parentheses).
_PHONE_SENDER = re.compile(r"^\+?[0-9()\-. ]{7,}$")


def _phone_digits(value: str) -> str:
    return re.sub(r"\D", "", str(value))


def _phones_match(alias_digits: str, sender_digits: str) -> bool:
    """Digit-normalized comparison. Exact, or the same trailing ten
    digits, so "+1 (555) 000-1111" in the config matches the export's
    "+15550001111" and a bare "5550001111"."""
    if not alias_digits or not sender_digits:
        return False
    if alias_digits == sender_digits:
        return True
    return (
        len(alias_digits) >= 10
        and len(sender_digits) >= 10
        and alias_digits[-10:] == sender_digits[-10:]
    )


def is_self(name: str, identity: dict) -> bool:
    """True when a sender string matches any configured alias.

    Names, usernames, and emails use the legacy substring match in both
    directions. Phone numbers are compared on digits only (see
    _phones_match), and only when the sender itself is phone-shaped, so
    digits embedded in an unrelated handle never count as a match."""
    if not name:
        return False
    needle = name.strip().lower()
    if not needle:
        return False
    aliases = (
        identity.get("names", [])
        + identity.get("usernames", [])
        + identity.get("emails", [])
    )
    for alias in aliases:
        candidate = str(alias).strip().lower()
        if candidate and (candidate in needle or needle in candidate):
            return True
    if _PHONE_SENDER.match(name.strip()):
        sender_digits = _phone_digits(needle)
        for alias in identity.get("phone_numbers", []):
            if _phones_match(_phone_digits(alias), sender_digits):
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
