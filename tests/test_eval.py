"""Tests for the evaluation scorecard. Synthetic data only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["VOICE_OS_OFFLINE"] = "1"

from voice_os.eval import evaluate, main, render, sample_for_labeling  # noqa: E402
from voice_os.holdout import is_holdout  # noqa: E402

from test_mine import corpus_of, make_chunk  # noqa: E402

CORPUS = str(REPO_ROOT / "data" / "sample_corpus.txt")
FIXTURE_MINED = str(REPO_ROOT / "tests" / "fixtures" / "mined")


def build_store(tmp_path, chunks) -> str:
    chunks_dir = tmp_path / "corpus" / "chunks"
    chunks_dir.mkdir(parents=True)
    with open(chunks_dir / "synthetic.jsonl", "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    return str(chunks_dir)


def mixed_chunks() -> list[dict]:
    return (
        corpus_of(40, hint="eval friend", train=True)
        + corpus_of(40, hint="eval holdout friend", train=False)
    )


def test_evaluate_uses_only_holdout_chunks(tmp_path):
    chunks_dir = build_store(tmp_path, mixed_chunks())
    scorecard = evaluate(CORPUS, chunks_dir, mined_dir=None)
    assert scorecard["holdout"]["chunks"] == 40


def test_fidelity_variants_present_and_bounded(tmp_path):
    chunks_dir = build_store(tmp_path, mixed_chunks())
    scorecard = evaluate(CORPUS, chunks_dir, mined_dir=FIXTURE_MINED)
    for variant in ("baseline_only", "hand_calibrated", "extended_mined"):
        value = scorecard["fidelity"][variant]
        assert value is not None
        assert 0.0 <= value <= 1.0
    assert "friend-family" in scorecard["fidelity_breakouts"]["audience"]


def test_tone_mae_reported_with_mined_norms(tmp_path):
    chunks_dir = build_store(tmp_path, mixed_chunks())
    scorecard = evaluate(CORPUS, chunks_dir, mined_dir=FIXTURE_MINED)
    assert scorecard["tone_mae"]["mined"] is not None
    assert scorecard["tone_mae"]["mined"]["chunks"] == 40
    assert scorecard["tone_mae"]["global"] is not None


def test_tier4_holdout_chunks_are_excluded(tmp_path):
    chunks = mixed_chunks() + corpus_of(10, hint="old friend", train=False, tier=4)
    chunks_dir = build_store(tmp_path, chunks)
    scorecard = evaluate(CORPUS, chunks_dir, mined_dir=None)
    assert scorecard["holdout"]["chunks"] == 40


def test_goal_labels_scored_when_present(tmp_path):
    chunks = mixed_chunks()
    chunks_dir = build_store(tmp_path, chunks)
    labels_dir = tmp_path / "corpus" / "labels"
    labels_dir.mkdir(parents=True)
    holdout = [c for c in chunks if is_holdout(c["hash"])]
    with open(labels_dir / "goals.jsonl", "w") as f:
        f.write(json.dumps({"id": holdout[0]["id"], "goal": "connect"}) + "\n")
        f.write(json.dumps({"id": holdout[1]["id"], "goal": "inform"}) + "\n")
    scorecard = evaluate(CORPUS, chunks_dir, mined_dir=None)
    assert scorecard["goal_labels"]["labeled"] == 2
    assert scorecard["goal_labels"]["accuracy"] == 0.5  # chunks carry goal=connect


def test_render_produces_readable_scorecard(tmp_path):
    chunks_dir = build_store(tmp_path, mixed_chunks())
    text = render(evaluate(CORPUS, chunks_dir, mined_dir=FIXTURE_MINED))
    assert "context fidelity" in text
    assert "extended_mined" in text
    assert "tone mean absolute error" in text


def test_sample_for_labeling_is_deterministic(tmp_path):
    chunks_dir = build_store(tmp_path, mixed_chunks())
    first = [c["id"] for c in sample_for_labeling(chunks_dir, 5)]
    second = [c["id"] for c in sample_for_labeling(chunks_dir, 5)]
    assert first == second
    assert len(first) == 5
    assert all(
        is_holdout(c["hash"]) for c in sample_for_labeling(chunks_dir, 5)
    )


def test_cli_run_and_label(tmp_path, capsys, monkeypatch):
    build_store(tmp_path, mixed_chunks())
    corpus_txt = tmp_path / "corpus" / "voice_corpus.txt"
    corpus_txt.write_text(Path(CORPUS).read_text())
    monkeypatch.chdir(tmp_path)

    assert main(["--corpus-dir", "corpus"]) == 0
    assert "scorecard" in capsys.readouterr().out

    assert main(["label", "--sample", "3", "--corpus-dir", "corpus"]) == 0
    out = capsys.readouterr().out
    assert len([l for l in out.splitlines() if l.startswith("{")]) == 3

    save_path = tmp_path / "corpus" / "runs" / "eval-test.json"
    assert main(["--corpus-dir", "corpus", "--save", str(save_path)]) == 0
    assert save_path.exists()
