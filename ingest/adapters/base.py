"""Adapter contract.

An adapter owns discovery (which configured paths exist) and extraction
(yield RawRecord per authored text). Normalization, dedup, tiering, and
enrichment happen downstream in the CLI so every source shares one path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class RawRecord:
    text: str
    source_type: str
    origin_file: str
    export_id: str
    timestamp: str | None = None
    relationship_hint: str = ""
    extra: dict = field(default_factory=dict)


class SourceAdapter:
    name = ""
    version = "1.0"

    def __init__(self, config: dict):
        self.config = config
        self.identity = config.get("identity", {})
        self.options = config.get("sources", {}).get(self.name, {})
        self.warnings: list[str] = []

    @property
    def extractor_id(self) -> str:
        return f"ingest.adapters.{self.name}@{self.version}"

    def configured_paths(self) -> list[str]:
        """Paths this adapter would read, as configured. Override the key
        name per adapter if it uses something other than 'paths'."""
        return list(self.options.get("paths", []))

    def available(self) -> bool:
        import os

        return any(os.path.exists(p) for p in self.configured_paths())

    def iter_records(self) -> Iterator[RawRecord]:
        raise NotImplementedError
