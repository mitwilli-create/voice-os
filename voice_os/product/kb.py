"""KB loading and snapshot versioning for the callable layer.

Loads the legacy claude.ai Projects KB (compact JSON + system prompt)
from the gitignored sources directory, hashes the content, and keeps
content-addressed snapshots under the gitignored var/ directory so any
historical run can be traced to the exact KB it saw.

The loader never invents content: a missing directory or file yields a
bundle with status "absent" and the caller surfaces that in trace notes.

Privacy: KB files and snapshots are personal data. Snapshot destinations
live under var/ which is gitignored; nothing here writes inside tracked
paths.

Stdlib only. Design: docs/callable-layer.md.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone

DEFAULT_KB_DIR = os.path.join("sources", "drive-voice-os")
DEFAULT_VAR_DIR = "var"

_VERSION_RE = re.compile(r"System Instructions v(\d+(?:\.\d+)?)", re.IGNORECASE)
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


def _system_prompt_file(kb_dir: str) -> str | None:
    """Newest system prompt by parsed version header; filename tie-break.

    Candidates match *System-Instructions*.md. The version is read from
    the first few lines ("System Instructions v<major.minor>"); files
    with no parseable version sort below any versioned file.
    """
    candidates = sorted(glob.glob(os.path.join(kb_dir, "*System-Instructions*.md")))
    best: tuple[float, str] | None = None
    for path in candidates:
        version = -1.0
        try:
            with open(path, encoding="utf-8") as f:
                head = "".join(f.readline() for _ in range(_HEADER_LINES))
            match = _VERSION_RE.search(head)
            if match:
                version = float(match.group(1))
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
