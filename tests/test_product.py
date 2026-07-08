"""Offline tests for the callable product layer (voice_os/product/).

Alias, state, and KB tests are stdlib-only and always run. Graph tests
skip cleanly when langgraph is not installed. Everything is
offline-deterministic and uses synthetic fixtures only: the sample
corpus, the fictional Test Person mined artifacts, and a fake KB
written into tmp_path. The real personal KB under sources/ is never
touched by tests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["VOICE_OS_OFFLINE"] = "1"

from voice_os.product.aliases import normalize_context  # noqa: E402
from voice_os.product.state import build_result, initial_state  # noqa: E402
from voice_os.product import kb as kb_module  # noqa: E402

CORPUS = str(REPO_ROOT / "data" / "sample_corpus.txt")
BANNED = str(REPO_ROOT / "data" / "banned_list.txt")
MINED = str(REPO_ROOT / "tests" / "fixtures" / "mined")


# ---------------------------------------------------------------- aliases


def test_mission_example_maps_to_canonical_context():
    ctx = normalize_context(
        channel="email",
        audience="boss",
        situation="high_stakes",
        goal="set_expectations",
    )
    assert ctx == {
        "channel": "email",
        "audience": "leadership",
        "situation": "standard",
        "goal": "set-expectations",
        "stakes": "high",
        "medium": None,
    }


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("boss", "leadership"),
        ("manager", "leadership"),
        ("coworker", "peer"),
        ("report", "direct-report"),
        ("direct_report", "direct-report"),
        ("client", "external"),
        ("friend", "friend-family"),
        ("recruiter", "job-seeking"),
        ("connection", "networking"),
    ],
)
def test_audience_aliases(alias, canonical):
    assert normalize_context(audience=alias)["audience"] == canonical


@pytest.mark.parametrize(
    "alias,canonical",
    [("slack", "chat"), ("teams", "chat"), ("sms", "text"), ("imessage", "text")],
)
def test_channel_aliases(alias, canonical):
    assert normalize_context(channel=alias)["channel"] == canonical


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("followup", "follow-up"),
        ("apology", "error-ack"),
        ("badnews", "bad-news"),
        ("follow_up", "follow-up"),
    ],
)
def test_situation_aliases(alias, canonical):
    assert normalize_context(situation=alias)["situation"] == canonical


def test_goal_underscores_and_deescalate():
    assert normalize_context(goal="set_expectations")["goal"] == "set-expectations"
    assert normalize_context(goal="deescalate")["goal"] == "de-escalate"


def test_explicit_stakes_wins_over_situation_reroute():
    ctx = normalize_context(situation="high_stakes", stakes="critical")
    assert ctx["stakes"] == "critical"
    # explicit stakes present, so the situation is NOT rerouted away;
    # the stakes-shaped situation is left for validation to reject.
    assert ctx["situation"] == "high-stakes"


def test_stakes_defaults_to_routine():
    assert normalize_context()["stakes"] == "routine"


def test_stakes_aliases():
    assert normalize_context(stakes="high_stakes")["stakes"] == "high"
    assert normalize_context(stakes="normal")["stakes"] == "routine"


def test_unknown_values_pass_through_for_validation():
    from voice_os.contexts import VoiceContext

    ctx = normalize_context(audience="alien-overlord")
    with pytest.raises(ValueError, match="audience"):
        VoiceContext(**ctx).validate()


# ------------------------------------------------------------------ state


def test_initial_state_is_json_serializable_and_complete():
    state = initial_state(
        input_text="hello",
        channel="email",
        audience="peer",
        situation="standard",
        goal="unknown",
        stakes="routine",
        medium=None,
        max_revisions=2,
    )
    json.dumps(state)  # raises on anything non-serializable
    assert state["qa_decision"] == "revise"
    assert state["revision_count"] == 0
    assert state["current_draft"] == ""


def test_build_result_envelope_shape():
    state = initial_state(
        input_text="hello",
        channel="email",
        audience="peer",
        situation="standard",
        goal="unknown",
        stakes="routine",
        medium=None,
        max_revisions=2,
    )
    state.update(
        qa_decision="pass",
        current_draft="done",
        fidelity_scores={"overall": 0.9, "per_axis": {}},
        persona_modes=["offline"],
    )
    result = build_result(state, "run-x")
    json.dumps(result)
    assert result["run_id"] == "run-x"
    assert result["decision"] == "pass"
    assert result["output_text"] == "done"
    assert result["mode"] == "offline"
    assert result["context"]["audience"] == "peer"


# --------------------------------------------------------------------- kb


def _write_fake_kb(root: Path) -> Path:
    kb_dir = root / "kb"
    kb_dir.mkdir()
    (kb_dir / "Test-Person-System-Instructions_2026-01-20.md").write_text(
        "# VOICE OS System Instructions v5.0\n\nTest Person synthetic prompt.\n",
        encoding="utf-8",
    )
    (kb_dir / "Test-Person-System-Instructions_2_2026-01-20.md").write_text(
        "# VOICE OS System Instructions v4.0\n\nOlder synthetic prompt.\n",
        encoding="utf-8",
    )
    (kb_dir / "test-person-voice-os-compact_2026-01-19.json").write_text(
        json.dumps({"schema_version": "test", "patterns": ["synthetic"]}),
        encoding="utf-8",
    )
    return kb_dir


def test_load_kb_picks_highest_version_and_hashes(tmp_path):
    kb_dir = _write_fake_kb(tmp_path)
    bundle = kb_module.load_kb(str(kb_dir))
    assert bundle["status"] == "ok"
    assert bundle["system_prompt_file"] == (
        "Test-Person-System-Instructions_2026-01-20.md"
    )
    assert "v5.0" in bundle["system_prompt"]
    assert bundle["compact"]["schema_version"] == "test"
    assert len(bundle["files"]) == 2
    assert all(len(f["sha256"]) == 64 for f in bundle["files"])
    assert bundle["bundle_hash"]
    assert bundle["errors"] == []


def test_load_kb_absent_dir_reports_status_not_content(tmp_path):
    bundle = kb_module.load_kb(str(tmp_path / "nope"))
    assert bundle["status"] == "absent"
    assert bundle["system_prompt"] is None
    assert bundle["compact"] is None
    assert bundle["bundle_hash"] is None


def test_snapshot_versioning_is_content_addressed(tmp_path):
    kb_dir = _write_fake_kb(tmp_path)
    var_dir = str(tmp_path / "var")

    first = kb_module.ensure_snapshot(kb_dir=str(kb_dir), var_dir=var_dir)
    assert first is not None
    assert len(kb_module.list_kb_snapshots(var_dir)) == 1

    # Unchanged content: no new snapshot, same manifest returned.
    again = kb_module.ensure_snapshot(kb_dir=str(kb_dir), var_dir=var_dir)
    assert again["snapshot_id"] == first["snapshot_id"]
    assert len(kb_module.list_kb_snapshots(var_dir)) == 1

    # Changed content: a second snapshot appears.
    compact = kb_dir / "test-person-voice-os-compact_2026-01-19.json"
    compact.write_text(json.dumps({"schema_version": "test-2"}), encoding="utf-8")
    third = kb_module.ensure_snapshot(kb_dir=str(kb_dir), var_dir=var_dir)
    assert third["snapshot_id"] != first["snapshot_id"]
    assert len(kb_module.list_kb_snapshots(var_dir)) == 2

    # Snapshot copies + manifest exist on disk under var/.
    snap_dir = Path(kb_module.snapshots_dir(var_dir)) / third["snapshot_id"]
    assert (snap_dir / "manifest.json").is_file()
    assert (snap_dir / compact.name).is_file()


def test_ensure_snapshot_absent_kb_returns_none(tmp_path):
    assert (
        kb_module.ensure_snapshot(
            kb_dir=str(tmp_path / "nope"), var_dir=str(tmp_path / "var")
        )
        is None
    )


# ------------------------------------------------------------------ graph


def _draft_kwargs(tmp_path, **overrides):
    kwargs = {
        "corpus_path": CORPUS,
        "chunks_dir": None,
        "mined_dir": MINED,
        "banned_path": BANNED,
        "kb_dir": str(tmp_path / "kb"),
        "var_dir": str(tmp_path / "var"),
    }
    kwargs.update(overrides)
    return kwargs


def test_import_voice_os_never_requires_langgraph():
    """Core import must not touch langgraph even when it is installed."""
    import subprocess

    code = (
        "import sys; sys.modules['langgraph'] = None; "
        "import voice_os; "
        "assert 'voice_os.product.graph' not in sys.modules; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_draft_runs_full_graph_offline(tmp_path):
    pytest.importorskip("langgraph")
    import voice_os

    _write_fake_kb(tmp_path)
    result = voice_os.draft(
        "I just wanted to reach out about the launch. Please don't hesitate "
        "to reach out with questions.",
        channel="email",
        audience="boss",
        situation="high_stakes",
        goal="set_expectations",
        **_draft_kwargs(tmp_path),
    )
    json.dumps(result)
    assert result["decision"] in ("pass", "reject")
    assert result["mode"] == "offline"
    assert result["output_text"].strip()
    assert 0 <= result["revisions"] <= 2
    assert len(result["revision_history"]) == result["revisions"]
    assert result["context"]["audience"] == "leadership"
    assert result["context"]["stakes"] == "high"
    assert result["fidelity"]["overall"] is not None
    assert result["kb"]["status"] == "ok"
    assert result["trace"]

    # Checkpoints landed under the tmp var dir, not the repo tree.
    db = Path(tmp_path / "var" / "runs.sqlite")
    assert db.is_file()

    history = voice_os.run_history(result["run_id"], var_dir=str(tmp_path / "var"))
    assert history
    assert history[-1]["qa_decision"] == result["decision"]
    nodes = {step["node"] for step in history}
    assert {"prepare", "generate", "critique", "qa_gate"} <= nodes


def test_draft_reject_path_bounded(tmp_path):
    pytest.importorskip("langgraph")
    import voice_os

    # Zero revisions allowed: a below-threshold first pass must reject
    # rather than loop.
    result = voice_os.draft(
        "Per my last email, let's leverage synergy and circle back to touch "
        "base at your earliest convenience. Just checking in. I hope this "
        "email finds you well.",
        audience="friend",
        channel="sms",
        max_revisions=0,
        **_draft_kwargs(tmp_path),
    )
    assert result["decision"] in ("pass", "reject")
    assert result["revisions"] == 0
    assert result["revision_history"] == []


def test_draft_validates_before_graph(tmp_path):
    pytest.importorskip("langgraph")
    import voice_os

    with pytest.raises(ValueError, match="audience"):
        voice_os.draft("hello", audience="alien-overlord", **_draft_kwargs(tmp_path))
    with pytest.raises(ValueError, match="text"):
        voice_os.draft("   ", **_draft_kwargs(tmp_path))
    with pytest.raises(ValueError, match="max_revisions"):
        voice_os.draft("hello", max_revisions=-1, **_draft_kwargs(tmp_path))


def test_describe_graph_names_all_nodes():
    pytest.importorskip("langgraph")
    import voice_os

    mermaid = voice_os.describe_graph()
    for node in ("prepare", "generate", "critique", "qa_gate", "revise"):
        assert node in mermaid


def test_repo_tree_stays_clean_after_draft(tmp_path):
    pytest.importorskip("langgraph")
    import subprocess
    import voice_os

    _write_fake_kb(tmp_path)
    voice_os.draft("Quick note about the plan.", **_draft_kwargs(tmp_path))
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    dirty = [
        line
        for line in proc.stdout.splitlines()
        # Untracked handoff docs and in-flight source edits are fine;
        # what must never appear is run/snapshot data outside var/.
        if "runs.sqlite" in line or "kb_snapshots" in line
    ]
    assert dirty == [], f"draft() leaked run data into the repo tree: {dirty}"
