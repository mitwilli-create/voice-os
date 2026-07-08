"""Loaders for mined JSON artifacts.

The runtime never mines; it loads what mine/ wrote to the gitignored
corpus/mined/ directory. Missing files degrade gracefully (the hand
tables cover everything); a present file with the wrong artifact name or
schema version fails fast, matching the repo's ValueError convention.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

SCHEMA_VERSION = "1.0"

ARTIFACT_FILES = {
    "recipient_deltas": "recipient_deltas.json",
    "context_profiles": "context_profiles.json",
    "ngram_banned": "ngram_banned.json",
    "drift_report": "drift_report.json",
    "evolution_flags": "evolution_flags.json",
}


@dataclass
class MinedArtifacts:
    """In-memory view of the mined artifact set; every field optional."""

    recipient_deltas: dict | None = None
    context_profiles: dict | None = None
    ngram_banned: list[str] = field(default_factory=list)
    drift_report: dict | None = None
    evolution_flags: dict | None = None
    meta: dict = field(default_factory=dict)  # per-artifact generated_at etc.


def validate_artifact(data: dict, expected: str) -> dict:
    """Fail fast on artifact mismatch or version drift."""
    if data.get("artifact") != expected:
        raise ValueError(
            f"artifact mismatch: expected '{expected}', got '{data.get('artifact')}'"
        )
    if data.get("version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported artifact version '{data.get('version')}' for {expected}; "
            f"expected {SCHEMA_VERSION}"
        )
    if not isinstance(data.get("data"), dict):
        raise ValueError(f"artifact {expected} has no data body")
    return data


def load_artifacts(mined_dir: str | None) -> MinedArtifacts:
    """Load whatever artifacts exist under mined_dir; None loads nothing."""
    artifacts = MinedArtifacts()
    if not mined_dir or not os.path.isdir(mined_dir):
        return artifacts

    for name, filename in ARTIFACT_FILES.items():
        path = os.path.join(mined_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        validate_artifact(raw, name)
        artifacts.meta[name] = {
            "generated_at": raw.get("generated_at"),
            "miner": raw.get("miner"),
        }
        if name == "ngram_banned":
            entries = raw["data"].get("banned", [])
            if not isinstance(entries, list) or any(
                not isinstance(e, dict) or not isinstance(e.get("ngram"), str)
                for e in entries
            ):
                raise ValueError(
                    f"artifact {name} has a malformed banned list; expected "
                    "a list of objects each carrying an 'ngram' string"
                )
            artifacts.ngram_banned = [entry["ngram"] for entry in entries]
        elif name == "recipient_deltas":
            artifacts.recipient_deltas = raw["data"]
        elif name == "context_profiles":
            artifacts.context_profiles = raw["data"]
        elif name == "drift_report":
            artifacts.drift_report = raw["data"]
        elif name == "evolution_flags":
            artifacts.evolution_flags = raw["data"]
    return artifacts


def group_profile(context_profiles: dict | None, kind: str, key: str) -> dict | None:
    """Look up one mined group profile (audiences/media/goals/pairs)."""
    if not context_profiles:
        return None
    return context_profiles.get(kind, {}).get(key)


def recipient_profile(recipient_deltas: dict | None, recipient: str) -> dict | None:
    """Look up a recipient (exact hint first, then email domain)."""
    if not recipient_deltas:
        return None
    key = " ".join(recipient.strip().lower().split())
    found = recipient_deltas.get("recipients", {}).get(key)
    if found:
        return found
    if "@" in key:
        domain = key.rsplit("@", 1)[-1]
        return recipient_deltas.get("domains", {}).get(domain)
    return None
