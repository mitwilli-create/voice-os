"""Offline tests for the evaluation harness (voice_os/harness/).

Case selection, briefs, scoring, and gate tests are stdlib-only and
always run. Graph tests skip cleanly when langgraph is not installed.
Everything is offline-deterministic and uses synthetic fixtures only;
the real personal corpus is never touched.

The fixture-corpus golden lock in this file is enforcement layer 1 of
the regression gate (docs/eval-harness.md): it runs in the standard
suite on every PR.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["VOICE_OS_OFFLINE"] = "1"

from tests import golden_utils  # noqa: E402
from voice_os.axes import AXES  # noqa: E402
from voice_os.harness import cases as cases_module  # noqa: E402
from voice_os.harness import gate as gate_module  # noqa: E402
from voice_os.harness import scoring  # noqa: E402
from voice_os.holdout import is_holdout  # noqa: E402

REGEN_HINT = (
    "harness summary diverged from the golden fixture; if the change is "
    "intentional, regenerate with: python3 tests/regen_goldens.py "
    "and review the golden diff in the PR"
)


def _store(tmp_path) -> str:
    chunks_dir = str(tmp_path / "chunks")
    golden_utils.write_harness_chunk_store(chunks_dir)
    return chunks_dir


# ------------------------------------------------------------------ briefs


def test_brief_strips_greeting_signoff_exclaims_and_fillers():
    text = "Hey there,\nI think we just need to ship the fix today!\nThanks"
    brief = cases_module.build_brief(text)
    assert "Hey there" not in brief
    assert "Thanks" not in brief
    assert "!" not in brief
    assert "i think" not in brief.lower()
    assert "just" not in brief.lower()
    assert "ship the fix today." in brief


def test_brief_keeps_content_lines_that_start_like_greetings():
    # A long first line is content, not a greeting, even if it starts
    # with a lexicon word.
    text = "Hey I wanted to walk through the full rollout plan for next week."
    assert "rollout plan" in cases_module.build_brief(text)


def test_brief_falls_back_when_everything_is_style():
    # The greeting line is stripped, leaving nothing, so the brief
    # falls back to the whitespace-collapsed original text.
    assert cases_module.build_brief("Hey there!") == "Hey there!"


def test_brief_is_deterministic():
    text = "Hello,\nMaybe we could possibly review the numbers soon! 🚀\nBest"
    assert cases_module.build_brief(text) == cases_module.build_brief(text)


# --------------------------------------------------------------- selection


def test_select_cases_only_tier1_holdout_in_band(tmp_path):
    chunks_dir = _store(tmp_path)
    selected = cases_module.select_cases(chunks_dir, per_cell=50, cap=1000)
    assert selected
    for case in selected:
        assert is_holdout(case["hash"])
    # Everything selected exists in the store as tier 1.
    with open(
        os.path.join(chunks_dir, "synthetic.jsonl"), encoding="utf-8"
    ) as f:
        by_hash = {
            record["hash"]: record
            for record in (json.loads(line) for line in f)
        }
    for case in selected:
        assert by_hash[case["hash"]]["tier"] == 1


def test_select_cases_deterministic_and_stratified(tmp_path):
    chunks_dir = _store(tmp_path)
    first = cases_module.select_cases(chunks_dir, per_cell=3, cap=6)
    second = cases_module.select_cases(chunks_dir, per_cell=3, cap=6)
    assert first == second
    assert len(first) == 6
    cells = [(case["channel"], case["audience"]) for case in first]
    assert cells.count(("chat", "friend-family")) == 3
    assert cells.count(("email", "peer")) == 3
    # Within each cell, hashes ascend (hash-sorted, not file-ordered).
    for cell in set(cells):
        hashes = [c["hash"] for c in first if (c["channel"], c["audience"]) == cell]
        assert hashes == sorted(hashes)


def test_select_cases_cap_trims_tail(tmp_path):
    chunks_dir = _store(tmp_path)
    assert len(cases_module.select_cases(chunks_dir, per_cell=3, cap=4)) == 4


def test_select_cases_rejects_bad_params(tmp_path):
    with pytest.raises(ValueError, match="per_cell"):
        cases_module.select_cases(str(tmp_path), per_cell=0)
    with pytest.raises(ValueError, match="cap"):
        cases_module.select_cases(str(tmp_path), cap=0)


def test_select_cases_empty_store(tmp_path):
    assert cases_module.select_cases(str(tmp_path)) == []


# ----------------------------------------------------------------- scoring


def test_cosine_identity_and_empty():
    assert scoring.cosine({"a": 2, "b": 1}, {"a": 2, "b": 1}) == pytest.approx(1.0)
    assert scoring.cosine({}, {"a": 1}) == 0.0
    assert scoring.cosine({"a": 1}, {"b": 1}) == 0.0


def test_paired_style_identical_text_is_perfect():
    text = "Quick note on the launch. We ship Friday and the tests pass."
    result = scoring.paired_style(text, text, {axis: 0.1 for axis in AXES})
    assert result["overall"] == 1.0
    assert all(value == 1.0 for value in result["per_axis"].values())


def test_judge_offline_is_derived_and_labeled():
    style = {"overall": 0.75, "per_axis": {axis: 0.5 for axis in AXES}}
    judge = scoring.judge_case("real", "generated", style, live=False)
    assert judge["mode"] == "offline"
    assert judge["same_author"] == 4  # 1 + round(4 * 0.75)
    assert all(rating == 3 for rating in judge["axes"].values())


def test_judge_parser_accepts_strict_json_and_rejects_garbage():
    good = json.dumps({**{axis: 4 for axis in AXES}, "same_author": 5})
    parsed = scoring._parse_judge(f"noise {good} trailing")
    assert parsed and parsed["same_author"] == 5
    assert scoring._parse_judge("not json at all") is None
    missing = json.dumps({"same_author": 5})
    assert scoring._parse_judge(missing) is None
    clamped = scoring._parse_judge(
        json.dumps({**{axis: 9 for axis in AXES}, "same_author": -3})
    )
    assert clamped["axes"][AXES[0]] == 5
    assert clamped["same_author"] == 1


def _fake_case_and_envelope():
    case = {
        "id": "c1",
        "hash": "ab" * 32,
        "channel": "email",
        "audience": "peer",
        "medium": None,
        "goal": "inform",
        "real_text": "Quick note on the launch. We ship Friday and tests pass.",
        "brief": "note on the launch. ship Friday and tests pass.",
    }
    envelope = {
        "run_id": "run-x",
        "decision": "pass",
        "output_text": "Quick note on the launch. We ship Friday and tests pass.",
        "fidelity": {"overall": 0.9},
        "revisions": 1,
        "mode": "offline",
        "banned_hits": [],
        "provenance": {"live_model": None},
    }
    return case, envelope


def test_score_case_composites_and_shape():
    case, envelope = _fake_case_and_envelope()
    std = {axis: 0.1 for axis in AXES}
    record = scoring.score_case(case, envelope, std, live=False)
    json.dumps(record)
    # Identical texts: perfect style and similarity, so the offline
    # composite is exactly the weight sum.
    assert record["style"]["overall"] == 1.0
    assert record["similarity"]["content"] == 1.0
    assert record["alignment_offline"] == 1.0
    assert record["alignment_judged"] is None  # offline judge never blends
    assert record["em_dash_hits"] == 0
    assert record["judge"]["mode"] == "offline"


def test_summarize_groups_and_modes():
    case, envelope = _fake_case_and_envelope()
    std = {axis: 0.1 for axis in AXES}
    record_a = scoring.score_case(case, envelope, std, live=False)
    case_b = {**case, "id": "c2", "channel": "chat", "audience": "friend-family"}
    envelope_b = {**envelope, "decision": "reject", "output_text": "ok."}
    record_b = scoring.score_case(case_b, envelope_b, std, live=False)
    summary = scoring.summarize([record_a, record_b])
    json.dumps(summary)
    assert summary["cases"] == 2
    assert summary["mode"]["personas"] == "offline"
    assert set(summary["by_channel"]) == {"email", "chat"}
    assert set(summary["by_audience"]) == {"peer", "friend-family"}
    assert summary["by_cell"]["email|peer"]["n"] == 1
    assert summary["overall"]["pass_rate"] == 0.5
    # No message text anywhere in the summary.
    assert "real_text" not in json.dumps(summary)


# -------------------------------------------------------------------- gate


def _summary_fixture() -> dict:
    metrics = {
        "n": 6,
        "alignment_offline": 0.7,
        "style_overall": 0.6,
        "banned_hit_rate": 0.0,
        "em_dash_rate": 0.0,
    }
    return {
        "overall": dict(metrics),
        "by_channel": {"email": dict(metrics)},
        "by_audience": {"peer": dict(metrics)},
    }


def test_gate_passes_against_itself_and_within_tolerance():
    baseline = _summary_fixture()
    assert gate_module.gate(baseline, baseline)["status"] == "pass"
    near = copy.deepcopy(baseline)
    near["overall"]["alignment_offline"] = 0.6996  # inside 0.005
    assert gate_module.gate(near, baseline)["status"] == "pass"


def test_gate_flags_overall_and_group_regressions():
    baseline = _summary_fixture()
    worse = copy.deepcopy(baseline)
    worse["overall"]["alignment_offline"] = 0.65
    worse["by_channel"]["email"]["alignment_offline"] = 0.6
    result = gate_module.gate(worse, baseline)
    assert result["status"] == "regression"
    failing = {record["metric"] for record in result["regressions"]}
    assert "overall.alignment_offline" in failing
    assert "by_channel.email.alignment_offline" in failing


def test_gate_rates_fail_on_increase_only():
    baseline = _summary_fixture()
    worse = copy.deepcopy(baseline)
    worse["overall"]["em_dash_rate"] = 0.2
    result = gate_module.gate(worse, baseline)
    assert {r["metric"] for r in result["regressions"]} == {
        "overall.em_dash_rate"
    }
    better = copy.deepcopy(baseline)
    better["overall"]["banned_hit_rate"] = 0.0
    assert gate_module.gate(better, baseline)["status"] == "pass"


def test_gate_rejects_degenerate_summaries():
    """A starved eval (zero cases, None aggregates) must block, not
    slide through as all-skipped; and it can never anchor the gate."""
    baseline = _summary_fixture()
    empty = {
        "overall": {
            "n": 0,
            "alignment_offline": None,
            "style_overall": None,
            "banned_hit_rate": None,
            "em_dash_rate": None,
        },
        "by_channel": {},
        "by_audience": {},
    }
    result = gate_module.gate(empty, baseline)
    assert result["status"] == "invalid"
    assert result["regressions"]
    assert gate_module.gate(baseline, empty)["status"] == "invalid"
    with pytest.raises(ValueError, match="refusing to store baseline"):
        gate_module.write_baseline(empty, "/tmp/never-written.json")
    assert gate_module.degenerate_reason(baseline) is None


def test_gate_cli_refuses_degenerate_baseline(tmp_path):
    empty = {
        "overall": {"n": 0, "alignment_offline": None},
        "by_channel": {},
        "by_audience": {},
    }
    path = tmp_path / "empty.summary.json"
    path.write_text(json.dumps(empty), encoding="utf-8")
    from voice_os.harness.__main__ import main

    baseline = str(tmp_path / "baseline.json")
    assert (
        main(
            [
                "gate",
                "--summary",
                str(path),
                "--baseline",
                baseline,
                "--update-baseline",
                "--force",
            ]
        )
        == 2
    )
    assert not os.path.exists(baseline)


def test_embed_similarity_honors_offline_override(monkeypatch):
    """VOICE_OS_OFFLINE is the privacy override: no text may leave the
    process, so the voyage branch must be skipped entirely."""
    monkeypatch.setenv(scoring.EMBED_BACKEND_ENV, "voyage")
    monkeypatch.setenv("VOICE_OS_OFFLINE", "1")
    block = scoring.embed_similarity("real text here", "generated text here")
    assert block["backend"] == "lexical"
    assert "semantic" not in block


def test_gate_skips_small_n_cells_and_missing_metrics():
    baseline = _summary_fixture()
    current = copy.deepcopy(baseline)
    current["by_channel"]["email"]["n"] = 2
    current["by_channel"]["email"]["alignment_offline"] = 0.1  # would fail
    result = gate_module.gate(current, baseline)
    assert result["status"] == "pass"
    assert any(
        record["status"] == "skipped-small-n" for record in result["checks"]
    )
    # A channel present on one side only is never gated.
    current = copy.deepcopy(baseline)
    current["by_channel"]["linkedin"] = dict(baseline["overall"])
    assert gate_module.gate(current, baseline)["status"] == "pass"


# ------------------------------------------------------------------- graph


def test_import_voice_os_never_touches_harness_graph():
    import subprocess

    code = (
        "import voice_os; "
        "voice_os.harness_gate; "  # resolves the lazy attribute
        "import sys; "
        "assert 'voice_os.harness.graph' not in sys.modules; "
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


def test_harness_run_matches_golden(tmp_path):
    """Enforcement layer 1 of the regression gate: the fixture-corpus
    summary is locked byte-for-byte."""
    pytest.importorskip("langgraph")
    result = golden_utils.build_harness_result(str(tmp_path))
    want = golden_utils.load_golden(golden_utils.HARNESS_GOLDEN)
    assert result["summary"] == want, REGEN_HINT


def test_harness_run_persists_reports_and_history(tmp_path):
    pytest.importorskip("langgraph")
    result = golden_utils.build_harness_result(str(tmp_path))
    assert result["cases"] == golden_utils.HARNESS_CAP

    # Full report has text; summary file has none.
    report = json.loads(
        Path(result["report_path"]).read_text(encoding="utf-8")
    )
    assert len(report["cases"]) == result["cases"]
    assert report["cases"][0]["real_text"]
    summary_text = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert "real_text" not in summary_text
    assert "output_text" not in summary_text

    # Checkpoints: one step per case visible in the run history, and
    # inner draft() checkpoints isolated under var/eval/.
    from voice_os.harness.graph import harness_history

    var_dir = str(tmp_path / "var")
    steps = harness_history(result["run_id"], var_dir=var_dir)
    assert steps
    assert steps[-1]["scored"] == result["cases"]
    runs = harness_history(var_dir=var_dir)
    assert [record["run_id"] for record in runs] == [result["run_id"]]
    assert (tmp_path / "var" / "harness.sqlite").is_file()
    assert (tmp_path / "var" / "eval" / "runs.sqlite").is_file()


def test_describe_harness_graph_names_all_nodes():
    pytest.importorskip("langgraph")
    import voice_os

    mermaid = voice_os.describe_harness_graph()
    for node in ("select", "run_case", "aggregate", "persist"):
        assert node in mermaid


def test_gate_cli_lifecycle(tmp_path):
    pytest.importorskip("langgraph")
    from voice_os.harness.__main__ import main

    result = golden_utils.build_harness_result(str(tmp_path))
    summary_path = result["summary_path"]
    baseline = str(tmp_path / "var" / "eval" / "baseline.json")

    # No baseline yet: exit 2; --update-baseline establishes it (exit 0).
    assert main(["gate", "--summary", summary_path, "--baseline", baseline]) == 2
    assert (
        main(
            [
                "gate",
                "--summary",
                summary_path,
                "--baseline",
                baseline,
                "--update-baseline",
            ]
        )
        == 0
    )
    # Same summary against the fresh baseline: pass.
    assert main(["gate", "--summary", summary_path, "--baseline", baseline]) == 0

    # Degrade the gated composite: exit 1.
    degraded = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    degraded["overall"]["alignment_offline"] -= 0.1
    degraded_path = str(tmp_path / "degraded.summary.json")
    Path(degraded_path).write_text(json.dumps(degraded), encoding="utf-8")
    assert main(["gate", "--summary", degraded_path, "--baseline", baseline]) == 1

    # A regressed summary cannot move the baseline without --force;
    # a forced move is explicit acceptance and succeeds.
    assert (
        main(
            [
                "gate",
                "--summary",
                degraded_path,
                "--baseline",
                baseline,
                "--update-baseline",
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "gate",
                "--summary",
                degraded_path,
                "--baseline",
                baseline,
                "--update-baseline",
                "--force",
            ]
        )
        == 0
    )
    updated = json.loads(Path(baseline).read_text(encoding="utf-8"))
    assert updated["overall"]["alignment_offline"] == pytest.approx(
        degraded["overall"]["alignment_offline"]
    )


def test_repo_tree_stays_clean_after_harness_run(tmp_path):
    pytest.importorskip("langgraph")
    import subprocess

    golden_utils.build_harness_result(str(tmp_path))
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
        if "harness.sqlite" in line
        or "eval/reports" in line
        or "runs.sqlite" in line
    ]
    assert dirty == [], f"harness run leaked data into the repo tree: {dirty}"
