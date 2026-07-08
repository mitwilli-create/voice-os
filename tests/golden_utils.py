"""Builders and normalizers shared by the golden envelope locks.

Both tests/test_determinism.py (the assertions) and tests/regen_goldens.py
(the regeneration entry point) import from here, so a golden is always
compared with exactly the inputs and normalization that produced it.

Normalization strips only fields documented as run-scoped or
machine-scoped in docs/determinism.md: run ids, KB snapshot ids, the
absolute corpus path, and the voice_os version string (version
propagation is asserted separately, so a version bump does not churn
the goldens). Everything else is locked byte-for-byte.
"""

from __future__ import annotations

import copy
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORPUS = os.path.join(REPO_ROOT, "data", "sample_corpus.txt")
BANNED = os.path.join(REPO_ROOT, "data", "banned_list.txt")
MINED = os.path.join(REPO_ROOT, "tests", "fixtures", "mined")
SLOP_DRAFT = os.path.join(REPO_ROOT, "data", "sample_draft.txt")

GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "golden")
RUN_PIPELINE_GOLDEN = os.path.join(GOLDEN_DIR, "run_pipeline_default.json")
DRAFT_GOLDEN = os.path.join(GOLDEN_DIR, "draft_offline.json")

VERSION_PLACEHOLDER = "<voice-os-version>"
RUN_ID_PLACEHOLDER = "<run-id>"
SNAPSHOT_PLACEHOLDER = "<snapshot-id>"
CORPUS_PATH_PLACEHOLDER = "<corpus-path>"

# Fixed inputs for the draft() golden. Deliberately duplicated from the
# product tests rather than imported, so an edit there cannot silently
# change what this golden locks.
DRAFT_INPUT_TEXT = (
    "I just wanted to reach out about the launch. Please don't hesitate "
    "to reach out with questions."
)
DRAFT_CONTEXT = {
    "channel": "email",
    "audience": "boss",
    "situation": "high_stakes",
    "goal": "set_expectations",
}
FAKE_KB_FILES = {
    "Test-Person-System-Instructions_2026-01-20.md": (
        "# VOICE OS System Instructions v5.0\n\nTest Person synthetic prompt.\n"
    ),
    "test-person-voice-os-compact_2026-01-19.json": json.dumps(
        {"schema_version": "test", "patterns": ["synthetic"]}
    ),
}


def build_run_pipeline_envelope() -> dict:
    """run_pipeline default-arguments output on the synthetic fixtures."""
    from voice_os import run_pipeline

    with open(SLOP_DRAFT, encoding="utf-8") as f:
        draft_text = f.read()
    return run_pipeline(CORPUS, draft_text, banned_path=BANNED)


def normalize_run_pipeline(envelope: dict) -> dict:
    envelope = copy.deepcopy(envelope)
    envelope["meta"]["voice_os_version"] = VERSION_PLACEHOLDER
    return envelope


def write_fake_kb(kb_dir: str) -> None:
    os.makedirs(kb_dir, exist_ok=True)
    for name, content in FAKE_KB_FILES.items():
        with open(os.path.join(kb_dir, name), "w", encoding="utf-8") as f:
            f.write(content)


def build_draft_envelope(work_dir: str) -> dict:
    """One offline draft() run against the synthetic fixtures.

    work_dir receives the fake KB and the var/ runtime data; callers pass
    a temp directory so nothing lands in the repo tree.
    """
    import voice_os

    kb_dir = os.path.join(work_dir, "kb")
    var_dir = os.path.join(work_dir, "var")
    write_fake_kb(kb_dir)
    return voice_os.draft(
        DRAFT_INPUT_TEXT,
        corpus_path=CORPUS,
        chunks_dir=None,
        mined_dir=MINED,
        banned_path=BANNED,
        kb_dir=kb_dir,
        var_dir=var_dir,
        **DRAFT_CONTEXT,
    )


def normalize_draft_envelope(envelope: dict) -> dict:
    envelope = copy.deepcopy(envelope)
    envelope["run_id"] = RUN_ID_PLACEHOLDER
    if "snapshot_id" in envelope.get("kb", {}):
        envelope["kb"]["snapshot_id"] = SNAPSHOT_PLACEHOLDER
    envelope["trace"] = [
        re.sub(r"kb snapshot \S+", f"kb snapshot {SNAPSHOT_PLACEHOLDER}", note)
        for note in envelope.get("trace", [])
    ]
    provenance = envelope.get("provenance", {})
    if "voice_os_version" in provenance:
        provenance["voice_os_version"] = VERSION_PLACEHOLDER
    if isinstance(provenance.get("corpus"), dict):
        provenance["corpus"]["path"] = CORPUS_PATH_PLACEHOLDER
    return envelope


def load_golden(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_golden(path: str, envelope: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
