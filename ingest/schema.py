"""Chunk schema: the contract between ingestion and the scoring layer.

Every extracted piece of text becomes a Chunk with provenance back to the
original export file plus context tags drawn from the calibration vocabulary
in voice_os.calibration, so ingest output flows directly into corpus headers
and the register calibration matrix picks the tags up unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "1.0"


def content_hash(normalized_text: str) -> str:
    """Full sha256 hex digest of the hash-normalized text."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


@dataclass
class Provenance:
    source_type: str
    """Fine-grained origin: instagram_dm, ig_post, messenger, fb_comment,
    email_sent, message_sent, imessage, document, video_transcript, ..."""

    origin_file: str
    """Path of the file inside the export the text came from."""

    export_id: str
    """Which export drop produced it (folder or archive name)."""

    timestamp: str | None
    """ISO-8601 local time, or None when the source carries no date.
    Undated chunks are assigned tier 4 and never replicated."""

    extractor: str
    """Adapter identity, e.g. ingest.adapters.instagram@1.0."""


@dataclass
class Context:
    channel: str
    """One of voice_os.calibration.CHANNELS."""

    audience: str
    """One of voice_os.calibration.AUDIENCES."""

    medium: str
    """Raw medium before vocabulary mapping: dm, post, comment, story,
    email, sms, script, spoken."""

    doc_type: str = ""
    """Document subtype for long-form sources: scripts, segment-intros,
    interview-questions, program-plans, cv, cover-letters, impact-docs,
    writing-samples, on-camera, ... Empty for sources without a subtype
    and for chunks ingested before this field existed."""

    relationship_hint: str = ""
    """Conversation name, recipient domain, or similar counterpart signal."""

    goal: str = "unknown"
    """inform | connect | coordinate | request | persuade | unknown."""

    tone_signals: dict = field(default_factory=dict)
    """Numeric style metrics only (counts, ratios, pacing)."""

    extra: dict = field(default_factory=dict)
    """Adapter-specific non-numeric metadata: email subject, source eml
    file, document chunk index, and similar."""

    inference: str = "heuristic-v1"
    """How the context tags were derived; lets a later model-based pass
    upgrade tags without re-ingesting."""


@dataclass
class Chunk:
    id: str
    text: str
    hash: str
    tier: int
    provenance: Provenance
    context: Context
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            hash=data["hash"],
            tier=data["tier"],
            provenance=Provenance(**data["provenance"]),
            context=Context(**data["context"]),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    @property
    def year(self) -> int | None:
        ts = self.provenance.timestamp
        if ts and len(ts) >= 4 and ts[:4].isdigit():
            return int(ts[:4])
        return None
