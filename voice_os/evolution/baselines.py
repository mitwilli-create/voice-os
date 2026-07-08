"""Content-hashed pattern baseline store under the gitignored var/.

Mirrors the KB snapshot scheme (voice_os/product/kb.py): a baseline is
stored once per distinct content hash; the baseline_id is a
timestamped directory name and is run-scoped, while content_hash is
the stable identity to compare across machines and runs.

Var-dir resolution follows the callable-layer convention: explicit
argument, then VOICE_OS_VAR_DIR, then the repo-root-anchored default.
Baseline bodies are personal data (pattern distributions derived from
private text) and never leave var/.

Stdlib only. Design: docs/evolution.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_VAR_DIR = os.path.join(REPO_ROOT, "var")


def _var_dir(var_dir: str | None) -> str:
    return var_dir or os.environ.get("VOICE_OS_VAR_DIR") or DEFAULT_VAR_DIR


def baselines_dir(var_dir: str | None = None) -> str:
    return os.path.join(_var_dir(var_dir), "evolution", "baselines")


def content_hash(body: dict) -> str:
    """Stable identity of a baseline body: sha256 of canonical JSON."""
    canonical = json.dumps(
        body, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _baseline_sort_key(baseline_id: str) -> tuple[str, int]:
    """Chronological ordering key robust to collision suffixes.

    Ids are `<stamp>` or `<stamp>-<n>`; a plain lexicographic sort
    would put `-10` before `-2`, so the suffix is compared as an
    integer (unparseable suffixes sort after their stamp's numbered
    siblings, deterministically by falling back to a large sentinel).
    """
    stamp, _, suffix = baseline_id.partition("-")
    if not suffix:
        return (stamp, 0)
    try:
        return (stamp, int(suffix))
    except ValueError:
        return (stamp, 1 << 30)


def list_baselines(var_dir: str | None = None) -> list[dict]:
    """All baseline manifests, oldest first (collision-suffix aware)."""
    root = baselines_dir(var_dir)
    manifests = []
    if not os.path.isdir(root):
        return manifests
    for entry in sorted(os.listdir(root), key=_baseline_sort_key):
        manifest_path = os.path.join(root, entry, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifests.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return manifests


def load_baseline_body(
    baseline_id: str, var_dir: str | None = None
) -> dict | None:
    path = os.path.join(baselines_dir(var_dir), baseline_id, "profile.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def latest_baseline(
    var_dir: str | None = None,
) -> tuple[dict, dict] | None:
    """(manifest, body) of the newest stored baseline, or None."""
    manifests = list_baselines(var_dir)
    for manifest in reversed(manifests):
        body = load_baseline_body(manifest["baseline_id"], var_dir)
        if body is not None:
            return manifest, body
    return None


def save_baseline(
    body: dict, var_dir: str | None = None, params: dict | None = None
) -> dict:
    """Store a baseline body + manifest under a fresh timestamped id."""
    root = baselines_dir(var_dir)
    baseline_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(root, baseline_id)
    suffix = 0
    while os.path.exists(dest):
        suffix += 1
        dest = os.path.join(root, f"{baseline_id}-{suffix}")
    if suffix:
        baseline_id = f"{baseline_id}-{suffix}"
    os.makedirs(dest)
    manifest = {
        "baseline_id": baseline_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "content_hash": content_hash(body),
        "n_chunks": body.get("profile", {}).get("n_chunks", 0),
        "params": params or {},
    }
    with open(os.path.join(dest, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, sort_keys=True, ensure_ascii=False)
    with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def ensure_baseline(
    body: dict, var_dir: str | None = None, params: dict | None = None
) -> tuple[dict, bool]:
    """Store the body exactly when its content hash is unseen.

    Returns (manifest, created): the matching manifest with
    created=False when identical content already exists, else the new
    manifest with created=True.
    """
    digest = content_hash(body)
    for manifest in list_baselines(var_dir):
        if manifest.get("content_hash") == digest:
            return manifest, False
    return save_baseline(body, var_dir=var_dir, params=params), True
