"""KB loading, snapshot versioning, and guidance distillation for the
callable layer.

Loads the legacy claude.ai Projects KB (compact JSON + system prompt)
from the gitignored sources directory, hashes the content, and keeps
content-addressed snapshots under the gitignored var/ directory so any
historical run can be traced to the exact KB it saw.
distill_kb_guidance() turns the loaded compact KB into a bounded list of
prompt-ready voice-pattern statements (docs/kb-fusion.md).

The loader never invents content: a missing directory or file yields a
bundle with status "absent" and the caller surfaces that in trace notes.
The distiller is schema-tolerant the same way: missing sections are
skipped, never fabricated.

Privacy: KB files and snapshots are personal data. Snapshot destinations
live under var/ which is gitignored; nothing here writes inside tracked
paths. Distilled guidance derived from KB content enters live persona
prompts and run checkpoints under var/ (docs/kb-fusion.md).

Stdlib only. Design: docs/callable-layer.md, docs/kb-fusion.md.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone

# All defaults resolve against the repo root this package lives in,
# never the process CWD, so the callable layer works from any working
# directory. Env vars and explicit arguments still override.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_KB_DIR = os.path.join(REPO_ROOT, "sources", "drive-voice-os")
DEFAULT_VAR_DIR = os.path.join(REPO_ROOT, "var")

_VERSION_RE = re.compile(r"System Instructions v(\d+(?:\.\d+)*)", re.IGNORECASE)
_HEADER_LINES = 5


def _kb_dir(kb_dir: str | None) -> str:
    return kb_dir or os.environ.get("VOICE_OS_KB_DIR") or DEFAULT_KB_DIR


def _var_dir(var_dir: str | None) -> str:
    return var_dir or os.environ.get("VOICE_OS_VAR_DIR") or DEFAULT_VAR_DIR


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_version(text: str) -> tuple[int, ...]:
    """Dotted version string to an integer tuple so 5.10 orders above
    5.2 (float comparison would misorder them)."""
    return tuple(int(part) for part in text.split("."))


def _system_prompt_file(kb_dir: str) -> str | None:
    """Newest system prompt by parsed version header; filename tie-break.

    Candidates match *System-Instructions*.md. The version is read from
    the first few lines ("System Instructions v<major.minor>") and
    compared as an integer tuple; files with no parseable version sort
    below any versioned file.
    """
    candidates = sorted(glob.glob(os.path.join(kb_dir, "*System-Instructions*.md")))
    best: tuple[tuple[int, ...], str] | None = None
    for path in candidates:
        version: tuple[int, ...] = (-1,)
        try:
            with open(path, encoding="utf-8") as f:
                head = "".join(f.readline() for _ in range(_HEADER_LINES))
            match = _VERSION_RE.search(head)
            if match:
                version = _parse_version(match.group(1))
        except OSError:
            continue
        if best is None or (version, path) > best:
            best = (version, path)
    return best[1] if best else None


def _compact_file(kb_dir: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(kb_dir, "*voice-os-compact*.json")))
    return candidates[-1] if candidates else None


def load_kb(kb_dir: str | None = None) -> dict:
    """Load the compact KB + system prompt into a hashed bundle.

    Returns a dict with status "ok" or "absent". "ok" requires at least
    one of the two files to exist; per-file problems are recorded in
    "errors" rather than raised, so a partially present KB still loads.
    """
    directory = _kb_dir(kb_dir)
    bundle: dict = {
        "status": "absent",
        "kb_dir": directory,
        "system_prompt": None,
        "system_prompt_file": None,
        "compact": None,
        "compact_file": None,
        "files": [],
        "bundle_hash": None,
        "errors": [],
    }
    if not os.path.isdir(directory):
        return bundle

    prompt_path = _system_prompt_file(directory)
    compact_path = _compact_file(directory)
    if not prompt_path and not compact_path:
        return bundle

    if prompt_path:
        try:
            with open(prompt_path, encoding="utf-8") as f:
                bundle["system_prompt"] = f.read()
            bundle["system_prompt_file"] = os.path.basename(prompt_path)
            bundle["files"].append(_file_record(prompt_path))
        except OSError as exc:
            bundle["errors"].append(f"system prompt unreadable: {exc}")
    if compact_path:
        try:
            with open(compact_path, encoding="utf-8") as f:
                bundle["compact"] = json.load(f)
            bundle["compact_file"] = os.path.basename(compact_path)
            bundle["files"].append(_file_record(compact_path))
        except (OSError, json.JSONDecodeError) as exc:
            bundle["errors"].append(f"compact KB unreadable: {exc}")

    if bundle["files"]:
        bundle["status"] = "ok"
        joined = "".join(
            record["sha256"]
            for record in sorted(bundle["files"], key=lambda r: r["name"])
        )
        bundle["bundle_hash"] = hashlib.sha256(joined.encode("ascii")).hexdigest()
    return bundle


def _file_record(path: str) -> dict:
    return {
        "name": os.path.basename(path),
        "path": path,
        "sha256": _sha256(path),
        "bytes": os.path.getsize(path),
    }


def snapshots_dir(var_dir: str | None = None) -> str:
    return os.path.join(_var_dir(var_dir), "kb_snapshots")


def list_kb_snapshots(var_dir: str | None = None) -> list[dict]:
    """All snapshot manifests, oldest first."""
    root = snapshots_dir(var_dir)
    manifests = []
    if not os.path.isdir(root):
        return manifests
    for entry in sorted(os.listdir(root)):
        manifest_path = os.path.join(root, entry, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifests.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return manifests


def snapshot_kb(
    kb_dir: str | None = None,
    var_dir: str | None = None,
    *,
    bundle: dict | None = None,
) -> dict:
    """Copy the current KB files into a timestamped snapshot + manifest.

    Accepts an already-loaded bundle so ensure_snapshot hashes each file
    exactly once. Raises FileNotFoundError when the KB is absent (there
    is nothing to snapshot; callers wanting soft behavior use
    ensure_snapshot).
    """
    if bundle is None:
        bundle = load_kb(kb_dir)
    if bundle["status"] != "ok":
        raise FileNotFoundError(
            f"no KB found at {bundle['kb_dir']}; nothing to snapshot"
        )
    root = snapshots_dir(var_dir)
    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(root, snapshot_id)
    suffix = 0
    while os.path.exists(dest):
        suffix += 1
        dest = os.path.join(root, f"{snapshot_id}-{suffix}")
    if suffix:
        snapshot_id = f"{snapshot_id}-{suffix}"
    os.makedirs(dest)
    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": bundle["kb_dir"],
        "bundle_hash": bundle["bundle_hash"],
        "files": [
            {key: record[key] for key in ("name", "sha256", "bytes")}
            for record in bundle["files"]
        ],
    }
    for record in bundle["files"]:
        shutil.copy2(record["path"], os.path.join(dest, record["name"]))
    with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def ensure_snapshot(
    bundle: dict | None = None,
    kb_dir: str | None = None,
    var_dir: str | None = None,
) -> dict | None:
    """Snapshot the KB exactly when its content hash is unseen.

    Returns the matching or newly created manifest, or None when the KB
    is absent (soft: the caller records the absence and proceeds).
    """
    if bundle is None:
        bundle = load_kb(kb_dir)
    if bundle["status"] != "ok":
        return None
    for manifest in list_kb_snapshots(var_dir):
        if manifest.get("bundle_hash") == bundle["bundle_hash"]:
            return manifest
    return snapshot_kb(var_dir=var_dir, bundle=bundle)


# --------------------------------------------------------- guidance fusion

# Total word budget for the distilled guidance: the same bounding idea as
# the per-exemplar cap in graph.py (_EXEMPLAR_MAX_WORDS), applied to the
# whole KB section so it informs the live prompt without dominating it.
KB_GUIDANCE_MAX_WORDS = 220
KB_GUIDANCE_MAX_ITEMS = 12


def _dig(mapping: dict, *keys: str):
    """Nested dict lookup that returns None on any missing/mistyped hop."""
    node = mapping
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _top_keys(counter, limit: int = 3) -> list[str]:
    """Highest-count keys of a pattern-frequency dict, deterministic order
    (count descending, then name)."""
    if not isinstance(counter, dict):
        return []
    items = [
        (str(name), value)
        for name, value in counter.items()
        if isinstance(value, (int, float)) and value > 0
    ]
    items.sort(key=lambda pair: (-pair[1], pair[0]))
    return [name for name, _ in items[:limit]]


def _pct_parts(source: dict, fields: tuple[tuple[str, str], ...]) -> list[str]:
    parts = []
    for key, label in fields:
        value = source.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{label} in about {round(value)} percent")
    return parts


def distill_kb_guidance(
    bundle: dict, max_words: int = KB_GUIDANCE_MAX_WORDS
) -> list[str]:
    """Bounded, prompt-ready voice-pattern statements from the compact KB.

    Reads only the highest-signal, current-voice fields (tier 1 pattern
    analysis + LinkedIn voice notes; field rationale in docs/kb-fusion.md)
    and renders each as one plain-English line. Deterministic for a given
    bundle; returns [] when the compact KB is absent or has none of the
    expected sections. The result is capped at max_words total (and
    KB_GUIDANCE_MAX_ITEMS lines) so the persona prompt stays bounded.
    """
    compact = bundle.get("compact") if isinstance(bundle, dict) else None
    if not isinstance(compact, dict):
        return []

    candidates: list[str] = []

    email = _dig(compact, "pattern_analysis_by_tier", "tier_1_current", "email")
    if isinstance(email, dict):
        greetings = _top_keys(email.get("greetings"))
        if greetings:
            candidates.append(
                "Email greetings the author actually uses, most common "
                "first: " + "; ".join(greetings)
            )
        closings = _top_keys(email.get("closings"))
        if closings:
            candidates.append(
                "Email closings, most common first: " + "; ".join(closings)
            )
        structure = email.get("structure")
        if isinstance(structure, dict):
            parts = _pct_parts(structure, (
                ("bullet_usage_pct", "bullet lists"),
                ("bold_usage_pct", "bold emphasis"),
                ("tldr_usage_pct", "a TLDR opener"),
            ))
            if parts:
                candidates.append(
                    "Email structure: " + "; ".join(parts) + " of emails."
                )
        formality = email.get("formality")
        if isinstance(formality, dict):
            contraction = formality.get("contraction_usage_pct")
            if isinstance(contraction, (int, float)):
                candidates.append(
                    f"Contractions appear in about {round(contraction)} "
                    "percent of emails."
                )

    linkedin = _dig(
        compact, "pattern_analysis_by_tier", "tier_1_current", "linkedin"
    )
    if isinstance(linkedin, dict):
        parts = _pct_parts(linkedin, (
            ("exclamation_usage_pct", "exclamation marks"),
            ("question_usage_pct", "questions"),
            ("emoji_usage_pct", "emoji"),
        ))
        if parts:
            candidates.append("LinkedIn posts: " + "; ".join(parts) + ".")

    post_style = _dig(
        compact, "linkedin_voice_notes", "social_media_patterns", "post_style"
    )
    if isinstance(post_style, str) and post_style.strip():
        candidates.append("LinkedIn post style: " + post_style.strip())

    greeting = _dig(
        compact,
        "linkedin_voice_notes",
        "networking_message_patterns",
        "greeting",
    )
    if isinstance(greeting, str) and greeting.strip():
        candidates.append("LinkedIn networking greeting: " + greeting.strip())

    guidance: list[str] = []
    total_words = 0
    for line in candidates[:KB_GUIDANCE_MAX_ITEMS]:
        words = len(line.split())
        if total_words + words > max_words:
            break
        guidance.append(line)
        total_words += words
    return guidance
