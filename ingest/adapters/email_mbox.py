"""Gmail mbox adapter.

Reads an "All mail" mbox export and keeps only messages Mitchell sent
(From header matches an identity alias). Bodies are reduced to the newly
authored text: quoted replies, forwarded headers, mobile signatures, and
everything below a signature delimiter are stripped, since quoted text is
someone else's voice.
"""

from __future__ import annotations

import mailbox
import re
from email.header import decode_header, make_header
from email.utils import parseaddr, parsedate_to_datetime
from typing import Iterator

from ..normalize import is_self
from .base import RawRecord, SourceAdapter

_REPLY_INTRO = re.compile(r"^On .{4,80} wrote:\s*$")
_FORWARD_HEADER = re.compile(r"^(From|Sent|To|Cc|Subject|Date):\s")
_ORIGINAL_MARKER = re.compile(r"^-{2,}\s*(Original|Forwarded) [Mm]essage\s*-{2,}")
_MOBILE_SIG = re.compile(r"^Sent from my \w[\w ]*$")


def strip_quoted(body: str) -> str:
    """Keep only the self-authored top of an email body."""
    kept: list[str] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">"):
            break
        if _REPLY_INTRO.match(stripped) or _ORIGINAL_MARKER.match(stripped):
            break
        if stripped == "--" or stripped == "-- ":
            break
        if _MOBILE_SIG.match(stripped):
            continue
        if _FORWARD_HEADER.match(stripped) and kept and not kept[-1].strip():
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _decode_str(value) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except (UnicodeDecodeError, LookupError, ValueError):
        return str(value)


def _plain_body(message) -> str:
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain" and not part.get("Content-Disposition", "").startswith("attachment"):
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                try:
                    return payload.decode(charset, errors="replace")
                except LookupError:
                    return payload.decode("utf-8", errors="replace")
        return ""
    payload = message.get_payload(decode=True)
    if payload is None:
        return ""
    charset = message.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


class EmailMboxAdapter(SourceAdapter):
    name = "email"

    def configured_paths(self) -> list[str]:
        return list(self.options.get("mbox_paths", []))

    def iter_records(self) -> Iterator[RawRecord]:
        import os

        for mbox_path in self.configured_paths():
            if not os.path.exists(mbox_path):
                continue
            export_id = os.path.basename(mbox_path)
            box = mailbox.mbox(mbox_path)
            for message in box:
                from_name, from_addr = parseaddr(_decode_str(message.get("From", "")))
                if not (
                    is_self(from_addr, self.identity)
                    or is_self(from_name, self.identity)
                ):
                    continue
                body = strip_quoted(_plain_body(message))
                if not body:
                    continue
                timestamp = None
                try:
                    parsed = parsedate_to_datetime(message.get("Date", ""))
                    if parsed is not None:
                        timestamp = parsed.isoformat()
                except (TypeError, ValueError):
                    pass
                _, to_addr = parseaddr(_decode_str(message.get("To", "")))
                yield RawRecord(
                    text=body,
                    source_type="email_sent",
                    origin_file=export_id,
                    export_id=export_id,
                    timestamp=timestamp,
                    relationship_hint=to_addr,
                    extra={"subject": _decode_str(message.get("Subject", ""))},
                )
