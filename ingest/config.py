"""Configuration loading.

Machine-specific source paths live in ingest.local.json (gitignored);
ingest.example.json is the committed template. Identity aliases drive the
self-authorship filter, so only Mitchell's own words enter the corpus.
"""

from __future__ import annotations

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CONFIG = os.path.join(REPO_ROOT, "ingest.local.json")
EXAMPLE_CONFIG = os.path.join(REPO_ROOT, "ingest.example.json")


def _expand(value):
    if isinstance(value, str):
        return os.path.expanduser(value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: str | None = None) -> dict:
    candidates = [path] if path else [LOCAL_CONFIG, EXAMPLE_CONFIG]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                config = json.load(f)
            config = _expand(config)
            config.setdefault("corpus_dir", os.path.join(REPO_ROOT, "corpus"))
            config["corpus_dir"] = os.path.expanduser(config["corpus_dir"])
            if not os.path.isabs(config["corpus_dir"]):
                config["corpus_dir"] = os.path.join(REPO_ROOT, config["corpus_dir"])
            config.setdefault("identity", {})
            config.setdefault("sources", {})
            config.setdefault("export", {})
            config["_config_path"] = candidate
            return config
    raise FileNotFoundError(
        "No ingest config found. Copy ingest.example.json to ingest.local.json "
        "and fill in your local export paths."
    )
