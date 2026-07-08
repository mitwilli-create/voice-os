"""Builders and normalizers shared by the golden envelope locks.

Both tests/test_determinism.py (the assertions) and tests/regen_goldens.py
(the regeneration entry point) import from here, so a golden is always
compared with exactly the inputs and normalization that produced it.

Normalization strips only fields documented as run-scoped in
docs/determinism.md: run ids, KB snapshot ids, and the voice_os
version string (version propagation is asserted separately, so a
version bump does not churn the goldens). Everything else, including
the repo-relative provenance corpus path, is locked byte-for-byte.
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
HARNESS_GOLDEN = os.path.join(GOLDEN_DIR, "harness_summary.json")

VERSION_PLACEHOLDER = "<voice-os-version>"
RUN_ID_PLACEHOLDER = "<run-id>"
SNAPSHOT_PLACEHOLDER = "<snapshot-id>"

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
    # provenance.corpus.path is repo-relative by design (machine-neutral),
    # so it stays locked in the golden.
    return envelope


# Fixed inputs for the harness golden: a synthetic chunk store with two
# (channel, audience) cells and real SHA256 hashes, so the holdout split
# behaves exactly as in production. Small per-cell/cap keep the golden
# run to a handful of offline drafts.
HARNESS_CELLS = (("email", "peer"), ("chat", "friend-family"))
HARNESS_PER_CELL = 3
HARNESS_CAP = 6
_HARNESS_STORE_SIZE = 120


def write_harness_chunk_store(chunks_dir: str) -> None:
    """Deterministic synthetic tier-1 store for harness runs.

    Real content hashes mean roughly 20% of chunks land in the holdout
    split; the builder asserts each cell has enough held-out chunks so
    a lexicon or template edit that starves the fixture fails loudly
    here instead of producing a silently smaller golden.
    """
    import hashlib

    from voice_os.holdout import is_holdout

    os.makedirs(chunks_dir, exist_ok=True)
    holdout_per_cell = {cell: 0 for cell in HARNESS_CELLS}
    records = []
    for index in range(_HARNESS_STORE_SIZE):
        text = (
            f"Hey there,\n"
            f"Status note {index} for the demo project. The plan review "
            f"went well and the timeline still holds. I think we can "
            f"confirm the next steps before Friday!\n"
            f"Thanks"
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        channel, audience = HARNESS_CELLS[index % len(HARNESS_CELLS)]
        if is_holdout(digest):
            holdout_per_cell[(channel, audience)] += 1
        records.append(
            {
                "id": digest[:16],
                "hash": digest,
                "text": text,
                "tier": 1,
                "context": {
                    "channel": channel,
                    "audience": audience,
                    "goal": "inform",
                },
                "provenance": {
                    "timestamp": (
                        f"2026-01-{(index % 27) + 1:02d}T10:00:00-08:00"
                    ),
                    "source_type": "synthetic",
                },
            }
        )
    starved = {
        cell: count
        for cell, count in holdout_per_cell.items()
        if count < HARNESS_PER_CELL
    }
    assert not starved, (
        f"harness fixture store starved of held-out chunks: {starved}; "
        "adjust the template or store size"
    )
    with open(
        os.path.join(chunks_dir, "synthetic.jsonl"), "w", encoding="utf-8"
    ) as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def build_harness_result(work_dir: str) -> dict:
    """One offline harness run against the synthetic fixtures.

    Returns the harness_run envelope; the golden locks its summary,
    which contains no run-scoped fields (run id, timestamps, and paths
    live only in the persisted report files and the envelope).
    """
    from voice_os.harness.graph import harness_run

    chunks_dir = os.path.join(work_dir, "chunks")
    kb_dir = os.path.join(work_dir, "kb")
    var_dir = os.path.join(work_dir, "var")
    write_harness_chunk_store(chunks_dir)
    write_fake_kb(kb_dir)
    return harness_run(
        corpus_path=CORPUS,
        chunks_dir=chunks_dir,
        mined_dir=MINED,
        banned_path=BANNED,
        kb_dir=kb_dir,
        var_dir=var_dir,
        per_cell=HARNESS_PER_CELL,
        cap=HARNESS_CAP,
    )


def load_golden(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_golden(path: str, envelope: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
