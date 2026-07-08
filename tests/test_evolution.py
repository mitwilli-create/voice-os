"""Offline tests for the evolution module (voice_os/evolution/).

Pattern, timeline, baseline, insight, and drift-check tests are
stdlib-only and always run. Graph tests skip cleanly when langgraph is
not installed. Synthetic fixtures only (fictional Test Person text);
the real chunk store and var/ are never touched.
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

from voice_os.evolution import baselines as baseline_store  # noqa: E402
from voice_os.evolution import check_drift, evolution_timeline  # noqa: E402
from voice_os.evolution.insights import (  # noqa: E402
    era_insights,
    generate_insights,
)
from voice_os.evolution.patterns import (  # noqa: E402
    diff_profiles,
    extract_pattern_profile,
)
from voice_os.mined import load_artifacts  # noqa: E402

import hashlib  # noqa: E402


def make_chunk(text: str, *, year=2025, month="03", tier=1, audience="peer",
               medium="email", goal="inform") -> dict:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "id": f"c-{digest[:8]}",
        "hash": digest,
        "text": text,
        "tier": tier,
        "context": {"audience": audience, "medium": medium, "goal": goal},
        "provenance": {"timestamp": f"{year}-{month}-01T12:00:00"},
    }


def write_store(tmp_path: Path, chunks: list[dict]) -> str:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    with open(chunks_dir / "synthetic.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    return str(chunks_dir)


# ------------------------------------------------------------- patterns


def test_extract_pattern_profile_shape_and_classification():
    texts = [
        "hey, quick note on the launch! We ship Friday.",
        "Hi Alex\nThe rollout went fine this morning.\nThanks",
        "Good morning team\nTwo small issues, both fixed.\nBest",
        "Totally unrelated opener line.\nSee you at the sync.",
    ]
    profile = extract_pattern_profile(texts)
    json.dumps(profile)  # JSON-safe
    assert profile["n_chunks"] == 4
    assert profile["greeting_counts"]["hey"] == 1
    assert profile["greeting_counts"]["hi"] == 1
    assert profile["greeting_counts"]["good morning"] == 1
    assert profile["greeting_counts"]["other"] == 1
    assert profile["signoff_counts"]["thanks"] == 1
    assert profile["signoff_counts"]["best"] == 1
    assert profile["greetings"]["hey"] == 0.25
    assert profile["exclamations_per_100w"] > 0
    assert profile["sentence_length"]["mean"] > 0
    assert profile["sentence_length"]["p90"] >= profile["sentence_length"]["p50"]


def test_greeting_lexicon_is_the_privacy_boundary():
    # A name-led opener never enters the profile as its own key.
    profile = extract_pattern_profile(["Morning Alexandra, quick question."])
    assert set(profile["greeting_counts"]) == {"other"}


def test_marker_frequencies_per_100_words():
    text = "yeah " * 10 + "word " * 90
    profile = extract_pattern_profile([text])
    assert profile["marker_counts"]["yeah"] == 10
    assert profile["markers_per_100w"]["yeah"] == 10.0


def _profile_from(greeting: str, n: int, filler: str = "") -> dict:
    texts = [
        f"{greeting}, note number {i}.\n{filler}The plan holds. Thanks"
        for i in range(n)
    ]
    return extract_pattern_profile(texts)


def test_diff_profiles_emerging_and_fading():
    baseline = _profile_from("hey", 10)
    current = _profile_from("yo", 10)
    diff = diff_profiles(baseline, current)
    emerging = {e["key"] for e in diff["emerging"]}
    fading = {e["key"] for e in diff["fading"]}
    assert "yo" in emerging
    assert "hey" in fading
    assert any("emerging greeting 'yo'" in flag for flag in diff["flags"])
    assert any("fading greeting 'hey'" in flag for flag in diff["flags"])


def test_diff_profiles_support_floor_suppresses_noise():
    baseline = _profile_from("hey", 10)
    # Only 2 occurrences of the new form: below MIN_SUPPORT, no flag.
    current_texts = ["yo, note.\nThanks"] * 2 + ["hey, note.\nThanks"] * 8
    diff = diff_profiles(baseline, extract_pattern_profile(current_texts))
    assert all(e["key"] != "yo" for e in diff["emerging"])


def test_diff_profiles_other_bucket_never_flags():
    baseline = _profile_from("hey", 10)
    current = _profile_from("Zanzibar", 10)  # unlisted -> "other"
    diff = diff_profiles(baseline, current)
    assert all(e["key"] != "other" for e in diff["emerging"])
    assert all(e["key"] != "other" for e in diff["shifted"])


def test_diff_profiles_sentence_shift():
    short = extract_pattern_profile(["We ship. It works. Done."] * 5)
    long = extract_pattern_profile(
        ["This sentence keeps going with many words in a row before it "
         "finally stops somewhere far away from the start."] * 5
    )
    diff = diff_profiles(short, long)
    assert diff["sentence_shift"] is not None
    assert diff["sentence_shift"]["delta"] > 0
    assert any("sentence length shifted" in flag for flag in diff["flags"])


def test_identical_profiles_produce_no_flags():
    profile = _profile_from("hey", 10)
    diff = diff_profiles(profile, profile)
    assert diff["emerging"] == []
    assert diff["fading"] == []
    assert diff["shifted"] == []
    assert diff["sentence_shift"] is None
    assert diff["flags"] == []


# ------------------------------------------------------------- baselines


def test_baseline_store_is_content_addressed(tmp_path):
    var_dir = str(tmp_path / "var")
    body = {"profile": _profile_from("hey", 5), "windows": []}

    first, created = baseline_store.ensure_baseline(body, var_dir=var_dir)
    assert created
    assert len(baseline_store.list_baselines(var_dir)) == 1

    again, created_again = baseline_store.ensure_baseline(body, var_dir=var_dir)
    assert not created_again
    assert again["baseline_id"] == first["baseline_id"]
    assert len(baseline_store.list_baselines(var_dir)) == 1

    changed = {"profile": _profile_from("yo", 5), "windows": []}
    third, created_third = baseline_store.ensure_baseline(
        changed, var_dir=var_dir
    )
    assert created_third
    assert third["content_hash"] != first["content_hash"]
    assert len(baseline_store.list_baselines(var_dir)) == 2

    manifest, stored_body = baseline_store.latest_baseline(var_dir)
    assert manifest["baseline_id"] == third["baseline_id"]
    assert stored_body == changed

    snap_dir = Path(baseline_store.baselines_dir(var_dir)) / third["baseline_id"]
    assert (snap_dir / "manifest.json").is_file()
    assert (snap_dir / "profile.json").is_file()


def test_baseline_ordering_is_numeric_on_collision_suffixes(tmp_path):
    """...Z-10 must sort after ...Z-2 (lexicographic order would not)."""
    var_dir = str(tmp_path / "var")
    root = Path(baseline_store.baselines_dir(var_dir))
    stamp = "20260101T000000Z"
    for suffix, hash_tag in (("", "a"), ("-2", "b"), ("-10", "c")):
        baseline_id = f"{stamp}{suffix}"
        dest = root / baseline_id
        dest.mkdir(parents=True)
        (dest / "profile.json").write_text(
            json.dumps({"profile": {"n_chunks": 1}, "tag": hash_tag}),
            encoding="utf-8",
        )
        (dest / "manifest.json").write_text(
            json.dumps({
                "baseline_id": baseline_id,
                "content_hash": hash_tag * 4,
                "created_at": "2026-01-01T00:00:00+00:00",
                "n_chunks": 1,
                "params": {},
            }),
            encoding="utf-8",
        )
    order = [m["baseline_id"] for m in baseline_store.list_baselines(var_dir)]
    assert order == [stamp, f"{stamp}-2", f"{stamp}-10"]
    manifest, body = baseline_store.latest_baseline(var_dir)
    assert manifest["baseline_id"] == f"{stamp}-10"
    assert body["tag"] == "c"


def test_baseline_var_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_OS_VAR_DIR", str(tmp_path / "env-var"))
    assert baseline_store.baselines_dir(None).startswith(
        str(tmp_path / "env-var")
    )
    # Explicit argument still wins over the env var.
    assert baseline_store.baselines_dir(str(tmp_path / "arg")).startswith(
        str(tmp_path / "arg")
    )


# -------------------------------------------------------------- timeline


def _era_chunks() -> list[dict]:
    chunks = []
    for i in range(6):
        chunks.append(make_chunk(
            f"hey, old note {i}.\nLonger sentences dominated back then, "
            "with plenty of words in every line. Thanks",
            year=2021, tier=2, audience="leadership",
        ))
    for i in range(6):
        chunks.append(make_chunk(
            f"yo, note {i}.\nShort now. Thanks",
            year=2025, tier=1, audience="peer", medium="chat",
        ))
    return chunks


def test_timeline_groups_by_window_year_tier(tmp_path):
    chunks_dir = write_store(tmp_path, _era_chunks())
    windows = evolution_timeline(chunks_dir, group_by="window")
    assert [g["group"] for g in windows] == ["2021H1", "2025H1"]
    assert all(g["n_chunks"] == 6 for g in windows)
    assert set(windows[0]["axis_mean"]) == {
        "editorial_register", "sentence_rhythm", "rhetorical_pace",
        "hedging_behavior", "escalation_pattern", "risk_tolerance",
    }
    assert windows[1]["patterns"]["greeting_counts"]["yo"] == 6

    years = evolution_timeline(chunks_dir, group_by="year")
    assert [g["group"] for g in years] == ["2021", "2025"]
    tiers = evolution_timeline(chunks_dir, group_by="tier")
    assert [g["group"] for g in tiers] == ["tier-1", "tier-2"]


def test_timeline_slice_by_context(tmp_path):
    chunks_dir = write_store(tmp_path, _era_chunks())
    sliced = evolution_timeline(
        chunks_dir, group_by="window", slice_by={"audience": "leadership"}
    )
    assert [g["group"] for g in sliced] == ["2021H1"]
    empty = evolution_timeline(
        chunks_dir, group_by="window", slice_by={"audience": "nobody"}
    )
    assert empty == []


def test_tier_accepts_numeric_strings(tmp_path):
    """String-typed tiers must not silently drop chunks (Qodo PR #14)."""
    from voice_os.evolution import tier1_texts

    chunks = _era_chunks()
    for chunk in chunks:
        chunk["tier"] = str(chunk["tier"])
    chunks.append(make_chunk("malformed tier note.", tier=1))
    chunks[-1]["tier"] = "not-a-tier"
    chunks_dir = write_store(tmp_path, chunks)

    tiers = evolution_timeline(chunks_dir, group_by="tier")
    assert [g["group"] for g in tiers] == ["tier-1", "tier-2"]
    assert len(tier1_texts(chunks_dir)) == 6  # malformed tier skipped


def test_timeline_rejects_unknown_group_by(tmp_path):
    chunks_dir = write_store(tmp_path, _era_chunks())
    with pytest.raises(ValueError, match="group_by"):
        evolution_timeline(chunks_dir, group_by="constellation")


# -------------------------------------------------------------- insights


def test_era_insights_surface_greeting_displacement(tmp_path):
    chunks_dir = write_store(tmp_path, _era_chunks())
    windows = evolution_timeline(chunks_dir, group_by="window")
    findings = era_insights(windows)
    texts = " | ".join(f["text"] for f in findings)
    assert "greeting 'yo' rose" in texts
    assert "greeting 'hey' fell" in texts


def test_generate_insights_ranked_and_deterministic(tmp_path):
    chunks_dir = write_store(tmp_path, _era_chunks())
    first = generate_insights(chunks_dir, top_k=5, min_slice_chunks=3)
    second = generate_insights(chunks_dir, top_k=5, min_slice_chunks=3)
    assert first == second
    assert len(first) <= 5
    effects = [f["effect"] for f in first]
    assert effects == sorted(effects, reverse=True)
    for finding in first:
        assert set(finding) == {"kind", "subject", "effect", "text"}


def test_generate_insights_empty_store(tmp_path):
    chunks_dir = write_store(tmp_path, [])
    assert generate_insights(chunks_dir) == []


# ----------------------------------------------------------- check_drift


def test_check_drift_no_baseline_then_ok(tmp_path):
    chunks_dir = write_store(tmp_path, _era_chunks())
    var_dir = str(tmp_path / "var")

    result = check_drift(chunks_dir, var_dir)
    assert result["status"] == "no-baseline"
    assert result["profile"]["n_chunks"] == 6  # tier 1 only

    baseline_store.ensure_baseline(
        {"profile": result["profile"], "windows": []}, var_dir=var_dir
    )
    result = check_drift(chunks_dir, var_dir)
    assert result["status"] == "ok"
    assert result["diff"]["flags"] == []  # unchanged store: no drift


# ------------------------------------------------- mined integration


def _write_flags_artifact(mined_dir: Path, flags: list[str]) -> None:
    mined_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact": "evolution_flags",
        "version": "1.0",
        "generated_at": "2026-07-07T12:00:00",
        "miner": "evolution.graph@1.0",
        "params": {},
        "data": {"flags": flags, "emerging": [], "fading": [],
                 "shifted": [], "sentence_shift": None},
    }
    with open(mined_dir / "evolution_flags.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f)


def test_evolution_flags_load_and_surface_in_query_meta(tmp_path):
    from voice_os.model import VoiceModel

    mined_dir = tmp_path / "mined"
    _write_flags_artifact(mined_dir, ["emerging greeting 'yo' (0.0 -> 0.5)"])
    artifacts = load_artifacts(str(mined_dir))
    assert artifacts.evolution_flags["flags"]

    model = VoiceModel.load(
        str(REPO_ROOT / "data" / "sample_corpus.txt"),
        chunks_dir=None,
        mined_dir=str(mined_dir),
        banned_path=None,
    )
    q = model.query(channel="email", audience="peer", situation="standard")
    assert q.meta["evolution_flags"] == [
        "emerging greeting 'yo' (0.0 -> 0.5)"
    ]


def test_import_voice_os_evolution_never_requires_langgraph():
    import subprocess

    code = (
        "import sys; sys.modules['langgraph'] = None; "
        "import voice_os; import voice_os.evolution; "
        "assert callable(voice_os.evolution_timeline); "
        "assert callable(voice_os.check_drift); "
        "assert 'voice_os.evolution.graph' not in sys.modules; "
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


# ------------------------------------------------------------------ graph


def test_drift_run_establishes_then_detects(tmp_path):
    pytest.importorskip("langgraph")
    import voice_os

    chunks_dir = write_store(tmp_path, _era_chunks())
    var_dir = str(tmp_path / "var")
    mined_dir = str(tmp_path / "mined")

    # Run 1: no stored baseline -> establishes one, empty diff.
    first = voice_os.drift_run(chunks_dir, var_dir, mined_dir)
    json.dumps(first)
    assert first["baseline"]["established_this_run"]
    assert first["flags"] == []
    assert first["profile_summary"]["n_chunks"] == 6

    # Run 2 on the unchanged store: diff against the baseline, no drift.
    second = voice_os.drift_run(chunks_dir, var_dir, mined_dir)
    assert not second["baseline"]["established_this_run"]
    assert second["flags"] == []
    assert second["diff"]["emerging"] == []

    # The voice changes: tier-1 greetings flip to "hiya".
    changed = [c for c in _era_chunks() if c["tier"] == 2]
    for i in range(6):
        changed.append(make_chunk(
            f"hiya, note {i}.\nShort now. Thanks", year=2025, tier=1,
        ))
    chunks_dir_2 = write_store(tmp_path / "second", changed)
    third = voice_os.drift_run(chunks_dir_2, var_dir, mined_dir)
    emerging = {e["key"] for e in third["diff"]["emerging"]}
    fading = {e["key"] for e in third["diff"]["fading"]}
    assert "hiya" in emerging
    assert "yo" in fading
    assert third["flags"]

    # The mined artifact landed and loads through the standard loader,
    # carrying the current Tier 1 profile for generation-side fusion
    # (docs/pattern-fusion.md).
    artifacts = load_artifacts(mined_dir)
    assert artifacts.evolution_flags["flags"] == third["flags"]
    profile = artifacts.evolution_flags["pattern_profile"]
    assert profile["n_chunks"] == 6
    assert profile["greetings"].get("hiya") == 1.0

    # Baseline anchor did NOT move (update_baseline not passed).
    assert len(baseline_store.list_baselines(var_dir)) == 1

    # Explicit update moves the anchor; the next run is clean again.
    fourth = voice_os.drift_run(
        chunks_dir_2, var_dir, mined_dir, update_baseline=True
    )
    assert len(baseline_store.list_baselines(var_dir)) == 2
    fifth = voice_os.drift_run(chunks_dir_2, var_dir, mined_dir)
    assert fifth["flags"] == []
    assert fourth["run_id"] != fifth["run_id"]


def test_drift_run_history_lists_runs_and_steps(tmp_path):
    pytest.importorskip("langgraph")
    import voice_os

    chunks_dir = write_store(tmp_path, _era_chunks())
    var_dir = str(tmp_path / "var")
    mined_dir = str(tmp_path / "mined")

    first = voice_os.drift_run(chunks_dir, var_dir, mined_dir)
    second = voice_os.drift_run(chunks_dir, var_dir, mined_dir)

    runs = voice_os.drift_run_history(var_dir=var_dir)
    run_ids = {r["run_id"] for r in runs}
    assert {first["run_id"], second["run_id"]} <= run_ids

    steps = voice_os.drift_run_history(first["run_id"], var_dir=var_dir)
    assert steps
    nodes = {step["node"] for step in steps}
    assert {"extract", "compare", "flag", "record"} <= nodes

    db = Path(var_dir) / "evolution.sqlite"
    assert db.is_file()
    # Product runs DB untouched: the drift graph has its own file.
    assert not (Path(var_dir) / "runs.sqlite").exists()


def test_describe_drift_graph_names_all_nodes():
    pytest.importorskip("langgraph")
    import voice_os

    mermaid = voice_os.describe_drift_graph()
    for node in ("extract", "compare", "flag", "record"):
        assert node in mermaid


def test_repo_tree_stays_clean_after_drift_run(tmp_path):
    pytest.importorskip("langgraph")
    import subprocess
    import voice_os

    chunks_dir = write_store(tmp_path, _era_chunks())
    voice_os.drift_run(
        chunks_dir, str(tmp_path / "var"), str(tmp_path / "mined")
    )
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
        # Runtime-data names only; "evolution/baselines/" keeps the
        # trailing slash so the SOURCE file baselines.py (which may be
        # legitimately modified in a working tree) never matches.
        if "evolution.sqlite" in line
        or "evolution/baselines/" in line
        or "evolution_flags.json" in line
    ]
    assert dirty == [], f"drift run leaked data into the repo tree: {dirty}"


# --------------------------------------------------------------------- cli


def test_cli_timeline_and_check(tmp_path, capsys):
    from voice_os.evolution.__main__ import main

    chunks_dir = write_store(tmp_path, _era_chunks())
    assert main([
        "--chunks-dir", chunks_dir, "--json", "timeline",
        "--group-by", "tier",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [g["group"] for g in payload] == ["tier-1", "tier-2"]

    var_dir = str(tmp_path / "var")
    assert main([
        "--chunks-dir", chunks_dir, "--var-dir", var_dir, "--json", "check",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no-baseline"


def test_cli_accepts_common_flags_after_subcommand(tmp_path, capsys):
    from voice_os.evolution.__main__ import main

    chunks_dir = write_store(tmp_path, _era_chunks())
    # Post-command placement (the natural typing order) must also work,
    # and pre-command values must survive an omitted post-command flag.
    assert main([
        "check", "--chunks-dir", chunks_dir,
        "--var-dir", str(tmp_path / "var"), "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no-baseline"

    assert main([
        "--chunks-dir", chunks_dir, "timeline", "--group-by", "tier",
        "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [g["group"] for g in payload] == ["tier-1", "tier-2"]
