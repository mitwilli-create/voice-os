"""Text-message export adapter.

Handles the two local export formats:

1. Combined sent export: blocks of the form

       ========================================
       FILE: msg_12345.eml
       SUBJECT: Re: something
       DATE: Fri, 12 Jun 2015 09:33:32 -0400
       ----------------------------------------
       body...

   These are already self-authored (sent), but bodies still need the
   quoted-reply stripping shared with the mbox adapter.

2. iMessage line export: "[YYYY-MM-DD HH:MM:SS] sender: text" lines with
   bare-line continuations. Senders are phone numbers or emails, so
   identity.phone_numbers must be configured or every line is skipped
   (a warning says how many, rather than silently ingesting other
   people's words).
"""

from __future__ import annotations

import os
import re
from email.utils import parsedate_to_datetime
from typing import Iterator

from ..normalize import is_self
from .base import RawRecord, SourceAdapter
from .email_mbox import strip_quoted

_BLOCK_RULE = re.compile(r"^={10,}\s*$", re.MULTILINE)
_BODY_RULE = re.compile(r"^-{10,}\s*$", re.MULTILINE)
_IMESSAGE_LINE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] ([^:]+): (.*)$"
)


def parse_combined_sent(raw: str, origin_file: str, export_id: str) -> Iterator[RawRecord]:
    for block in _BLOCK_RULE.split(raw):
        parts = _BODY_RULE.split(block, maxsplit=1)
        if len(parts) != 2:
            continue
        header_text, body = parts
        headers = {}
        for line in header_text.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip().upper()] = value.strip()
        body = strip_quoted(body.strip())
        if not body:
            continue
        timestamp = None
        try:
            parsed = parsedate_to_datetime(headers.get("DATE", ""))
            if parsed is not None:
                timestamp = parsed.isoformat()
        except (TypeError, ValueError):
            pass
        yield RawRecord(
            text=body,
            source_type="email_sent",
            origin_file=origin_file,
            export_id=export_id,
            timestamp=timestamp,
            extra={
                "subject": headers.get("SUBJECT", ""),
                "eml_file": headers.get("FILE", ""),
            },
        )


def parse_imessage(
    raw: str, identity: dict, origin_file: str, export_id: str
) -> tuple[list[RawRecord], int]:
    """Returns (self-authored records, count of lines skipped because the
    sender did not match identity)."""
    records: list[RawRecord] = []
    skipped_other = 0
    current: RawRecord | None = None
    current_is_self = False
    for line in raw.split("\n"):
        match = _IMESSAGE_LINE.match(line)
        if match:
            if current is not None and current_is_self:
                records.append(current)
            ts_raw, sender, text = match.groups()
            current_is_self = is_self(sender, identity)
            if not current_is_self:
                skipped_other += 1
                current = None
                continue
            current = RawRecord(
                text=text,
                source_type="imessage",
                origin_file=origin_file,
                export_id=export_id,
                timestamp=ts_raw.replace(" ", "T"),
            )
        elif current is not None and current_is_self and line.strip():
            current.text += "\n" + line
    if current is not None and current_is_self:
        records.append(current)
    return records, skipped_other


class MessagesAdapter(SourceAdapter):
    name = "messages"

    def configured_paths(self) -> list[str]:
        return list(self.options.get("combined_sent_paths", [])) + list(
            self.options.get("imessage_paths", [])
        )

    def iter_records(self) -> Iterator[RawRecord]:
        for path in self.options.get("combined_sent_paths", []):
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            yield from parse_combined_sent(raw, os.path.basename(path), os.path.basename(path))

        for path in self.options.get("imessage_paths", []):
            if not os.path.exists(path):
                continue
            if not (
                self.identity.get("phone_numbers") or self.identity.get("emails")
            ):
                self.warnings.append(
                    f"{os.path.basename(path)}: identity.phone_numbers is empty, "
                    "cannot attribute iMessage lines; file skipped"
                )
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            records, skipped = parse_imessage(
                raw, self.identity, os.path.basename(path), os.path.basename(path)
            )
            if skipped:
                self.warnings.append(
                    f"{os.path.basename(path)}: skipped {skipped} lines from other senders"
                )
            yield from records
