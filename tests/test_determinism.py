"""Standing determinism invariants + golden envelope locks.

Implements docs/determinism.md hardening items 1 and 4: every offline
pure surface is run twice in-process and must produce byte-identical
JSON, and the two full output envelopes (run_pipeline default output,
offline draft()) are locked against golden fixtures minus documented
run-scoped fields.

Every new module that claims offline determinism (evolution queries,
drift baselines, insights) must add its surfaces to the double-run
section from its first commit.

Regenerate goldens after an intentional behavior change:
    python3 tests/regen_goldens.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["VOICE_OS_OFFLINE"] = "1"

from tests import golden_utils  # noqa: E402
from voice_os import __version__, load_corpus, run_pipeline  # noqa: E402
from voice_os.axes import score_text  # noqa: E402
from voice_os.calibration import calibrate_extended  # noqa: E402
from voice_os.contexts import VoiceContext  # noqa: E402
from voice_os.mined import load_artifacts  # noqa: E402
from voice_os.product import kb as kb_module  # noqa: E402
from voice_os.qa import find_banned, gate_extended, load_banned_list  # noqa: E402

CORPUS = golden_utils.CORPUS
BANNED = golden_utils.BANNED
MINED = golden_utils.MINED
SLOP = Path(golden_utils.SLOP_DRAFT).read_text(encoding="utf-8")

REGEN_HINT = (
    "envelope diverged from the golden fixture; if the change is "
    "intentional, regenerate with: python3 tests/regen_goldens.py "
    "and review the golden diff in the PR"
)


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def assert_double_run(build, strip=None):
    """Run a builder twice and require byte-identical canonical JSON."""
    first, second = build(), build()
    if strip:
        first = strip(copy.deepcopy(first))
        second = strip(copy.deepcopy(second))
    assert canon(first) == canon(second)


def synthetic_chunks() -> list[dict]:
    """A deterministic dated chunk set spanning tiers 1-3.

    Fictional content only; hashes are real SHA256 of the text so the
    holdout split behaves exactly as in production.
    """
    texts = [
        "Quick note on the launch. We ship Friday and the tests pass.",
        "Sounds good, I will send the summary tonight after the review.",
        "Thanks for the heads up. Let me check the numbers and confirm.",
        "The rollout went fine. Two small issues, both fixed this morning.",
        "Can you send the deck before noon? I want one pass before the call.",
        "Yeah that works. See you at the sync.",
    ]
    chunks = []
    for year in range(2018, 2026):
        for half, month in (("1", "03"), ("2", "09")):
            for i, text in enumerate(texts):
                body = f"{text} ({year}H{half})"
                digest = hashlib.sha256(body.encode()).hexdigest()
                chunks.append(
                    {
                        "id": f"c{year}{half}{i}",
                        "hash": digest,
                        "text": body,
                        "tier": 1 if year >= 2024 else 2 if year >= 2021 else 3,
                        "context": {
                            "audience": "peer" if i % 2 else "leadership",
                            "medium": "email" if i % 3 else "chat",
                            "goal": "inform" if i % 2 else "coordinate",
                            "relationship_hint": (
                                "alex@example.com" if i % 2 else "sam"
                            ),
                        },
                        "provenance": {
                            "timestamp": f"{year}-{month}-01T12:00:00"
                        },
                    }
                )
    return chunks


def strip_generated_at(artifact: dict) -> dict:
    artifact.pop("generated_at", None)
    return artifact


# ------------------------------------------------------------- golden locks


def test_run_pipeline_golden_envelope():
    got = golden_utils.normalize_run_pipeline(
        golden_utils.build_run_pipeline_envelope()
    )
    want = golden_utils.load_golden(golden_utils.RUN_PIPELINE_GOLDEN)
    assert got == want, REGEN_HINT


def test_run_pipeline_version_propagates():
    # The golden normalizes the version string so a bump does not churn
    # the fixture; propagation is locked here instead.
    envelope = golden_utils.build_run_pipeline_envelope()
    assert envelope["meta"]["voice_os_version"] == __version__


def test_draft_offline_golden_envelope(tmp_path):
    pytest.importorskip("langgraph")
    got = golden_utils.normalize_draft_envelope(
        golden_utils.build_draft_envelope(str(tmp_path))
    )
    want = golden_utils.load_golden(golden_utils.DRAFT_GOLDEN)
    assert got == want, REGEN_HINT


def test_draft_provenance_is_complete_and_consistent(tmp_path):
    pytest.importorskip("langgraph")
    envelope = golden_utils.build_draft_envelope(str(tmp_path))
    provenance = envelope["provenance"]
    assert provenance["voice_os_version"] == __version__
    assert provenance["kb_bundle_hash"] == envelope["kb"]["bundle_hash"]
    # Repo-relative, never absolute: local paths must not leak into
    # envelopes that leave the machine.
    assert provenance["corpus"]["path"] == os.path.join(
        "data", "sample_corpus.txt"
    )
    assert not os.path.isabs(provenance["corpus"]["path"])
    assert provenance["corpus"]["bytes"] == os.path.getsize(CORPUS)
    assert len(provenance["corpus"]["sha256"]) == 64
    assert "context_profiles" in provenance["artifacts"]
    assert "recipient_deltas" in provenance["artifacts"]
    # Offline run: no live engine to stamp.
    assert provenance["live_model"] is None
    assert envelope["mode"] == "offline"


def test_corpus_identity_memoized_and_machine_neutral(tmp_path):
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    graph_module._IDENTITY_CACHE.clear()
    try:
        first = graph_module._corpus_identity(CORPUS)
        second = graph_module._corpus_identity(CORPUS)
        assert first == second
        # One stat-keyed entry; the second call was served from the memo.
        assert len(graph_module._IDENTITY_CACHE) == 1
        assert first["path"] == os.path.join("data", "sample_corpus.txt")

        # Outside the repo: basename only, still hashed.
        outside = tmp_path / "external_corpus.txt"
        outside.write_text("--- 2025-01-01 ---\nhello\n", encoding="utf-8")
        identity = graph_module._corpus_identity(str(outside))
        assert identity["path"] == "external_corpus.txt"
        assert len(identity["sha256"]) == 64

        # Content change under the same path invalidates the memo.
        outside.write_text("--- 2025-01-01 ---\nchanged\n", encoding="utf-8")
        assert graph_module._corpus_identity(str(outside)) != identity
    finally:
        graph_module._IDENTITY_CACHE.clear()


# ------------------------------------------------- double-run invariants


def test_score_text_double_run():
    assert_double_run(lambda: score_text(SLOP))


def test_calibrate_extended_double_run():
    baseline = load_corpus(CORPUS)
    mined = load_artifacts(MINED)
    ctx = VoiceContext(
        channel="email",
        audience="leadership",
        situation="standard",
        goal="set-expectations",
        stakes="high",
        medium="email",
    )

    def build():
        target, sources = calibrate_extended(baseline, ctx, mined=mined)
        return {"target": target, "sources": sources}

    assert_double_run(build)


def test_gate_extended_double_run():
    baseline = load_corpus(CORPUS)
    banned = load_banned_list(BANNED)
    scores = score_text(SLOP)
    target = dict(baseline.mean)

    def build():
        result = gate_extended(
            scores, baseline, target, find_banned(SLOP, banned)
        )
        return {
            "decision": result.decision,
            "fidelity": result.fidelity,
            "per_axis": result.per_axis,
            "banned_hits": result.banned_hits,
            "revision_signals": result.revision_signals,
        }

    assert_double_run(build)


def test_run_pipeline_double_run():
    assert_double_run(lambda: run_pipeline(CORPUS, SLOP, banned_path=BANNED))


def test_miners_double_run():
    from mine.drift import mine_drift
    from mine.ngrams import mine_ngram_diffs
    from mine.recipients import mine_recipient_deltas
    from mine.tone_norms import mine_context_profiles

    contrast = [
        "I hope this email finds you well. Let us leverage synergy "
        "going forward.",
        "Per my last email, circling back to touch base at your "
        "earliest convenience.",
    ]
    builders = {
        "drift": lambda: mine_drift(synthetic_chunks(), min_chunks=3),
        "recipients": lambda: mine_recipient_deltas(
            synthetic_chunks(), min_chunks=1, min_weighted_words=1.0
        ),
        "tone": lambda: mine_context_profiles(synthetic_chunks(), min_chunks=1),
        "ngrams": lambda: mine_ngram_diffs(
            synthetic_chunks(), contrast, min_contrast_count=1
        ),
    }
    for name, build in builders.items():
        # generated_at is the envelope's documented run-scoped field.
        assert_double_run(build, strip=strip_generated_at)


def test_eval_scorecard_double_run(tmp_path):
    from voice_os.eval import evaluate

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    with open(chunks_dir / "synthetic.jsonl", "w", encoding="utf-8") as f:
        for chunk in synthetic_chunks():
            f.write(json.dumps(chunk) + "\n")
    assert_double_run(lambda: evaluate(CORPUS, str(chunks_dir), MINED))


def test_load_kb_bundle_hash_double_run(tmp_path):
    kb_dir = tmp_path / "kb"
    golden_utils.write_fake_kb(str(kb_dir))
    first = kb_module.load_kb(str(kb_dir))
    second = kb_module.load_kb(str(kb_dir))
    assert first["bundle_hash"] == second["bundle_hash"]
    assert canon(first) == canon(second)


def test_draft_envelope_double_run(tmp_path):
    pytest.importorskip("langgraph")
    first = golden_utils.normalize_draft_envelope(
        golden_utils.build_draft_envelope(str(tmp_path / "a"))
    )
    second = golden_utils.normalize_draft_envelope(
        golden_utils.build_draft_envelope(str(tmp_path / "b"))
    )
    assert canon(first) == canon(second)


# ------------------------------------ evolution module (docs/evolution.md)


def _evolution_store(root) -> str:
    chunks_dir = os.path.join(str(root), "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    with open(
        os.path.join(chunks_dir, "synthetic.jsonl"), "w", encoding="utf-8"
    ) as f:
        for chunk in synthetic_chunks():
            f.write(json.dumps(chunk) + "\n")
    return chunks_dir


def test_evolution_pure_surfaces_double_run(tmp_path):
    from voice_os.evolution import (
        check_drift,
        content_hash,
        diff_profiles,
        evolution_timeline,
        extract_pattern_profile,
        generate_insights,
    )

    texts = [chunk["text"] for chunk in synthetic_chunks()]
    assert_double_run(lambda: extract_pattern_profile(texts))

    profile_a = extract_pattern_profile(texts[: len(texts) // 2])
    profile_b = extract_pattern_profile(texts[len(texts) // 2 :])
    assert_double_run(lambda: diff_profiles(profile_a, profile_b))
    assert content_hash({"profile": profile_a}) == content_hash(
        {"profile": extract_pattern_profile(texts[: len(texts) // 2])}
    )

    chunks_dir = _evolution_store(tmp_path)
    assert_double_run(
        lambda: evolution_timeline(chunks_dir, group_by="window")
    )
    assert_double_run(lambda: evolution_timeline(chunks_dir, group_by="tier"))
    assert_double_run(
        lambda: generate_insights(chunks_dir, min_slice_chunks=3)
    )
    # check_drift with no stored baseline is pure extraction.
    assert_double_run(
        lambda: check_drift(chunks_dir, str(tmp_path / "var-none"))
    )


def _normalize_drift_envelope(envelope: dict) -> dict:
    import re as _re

    envelope = copy.deepcopy(envelope)
    envelope["run_id"] = "<run-id>"
    envelope["baseline"]["baseline_id"] = "<baseline-id>"
    envelope["artifact_path"] = "<artifact-path>"
    envelope["trace"] = [
        _re.sub(r"\d{8}T\d{6}Z(-\d+)?", "<ts>", note)
        for note in envelope.get("trace", [])
    ]
    return envelope


# ------------------------------------ harness module (docs/eval-harness.md)


def test_harness_pure_surfaces_double_run(tmp_path):
    from voice_os.harness import build_brief, select_cases

    text = "Hey there,\nMaybe we just ship the fix today! 🚀\nThanks"
    assert_double_run(lambda: build_brief(text))

    chunks_dir = str(tmp_path / "chunks")
    golden_utils.write_harness_chunk_store(chunks_dir)
    assert_double_run(lambda: select_cases(chunks_dir, per_cell=3, cap=6))


def test_harness_summary_double_run(tmp_path):
    """The full offline eval loop is deterministic: two runs from
    identical inputs produce byte-identical summaries (run-scoped
    fields live only in the envelope and report files, not here)."""
    pytest.importorskip("langgraph")
    first = golden_utils.build_harness_result(str(tmp_path / "a"))["summary"]
    second = golden_utils.build_harness_result(str(tmp_path / "b"))["summary"]
    assert canon(first) == canon(second)


def test_drift_run_envelope_double_run(tmp_path):
    pytest.importorskip("langgraph")
    from voice_os.evolution import drift_run

    chunks_dir = _evolution_store(tmp_path)
    first = _normalize_drift_envelope(
        drift_run(chunks_dir, str(tmp_path / "a"), str(tmp_path / "mined-a"))
    )
    second = _normalize_drift_envelope(
        drift_run(chunks_dir, str(tmp_path / "b"), str(tmp_path / "mined-b"))
    )
    # Both runs establish a baseline from identical inputs: identical
    # envelopes (including the content hash) minus run-scoped fields.
    assert canon(first) == canon(second)
    assert first["baseline"]["content_hash"]
