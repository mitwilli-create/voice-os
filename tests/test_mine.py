"""Tests for the mining layer. Synthetic chunks only, built in-code and
written to tmp_path, so no real corpus content is used or committed."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["VOICE_OS_OFFLINE"] = "1"

from mine.cli import main as mine_main  # noqa: E402
from mine.recipients import mine_recipient_deltas  # noqa: E402
from mine.tone_norms import mine_context_profiles  # noqa: E402
from voice_os.mined import group_profile, load_artifacts, validate_artifact  # noqa: E402
from voice_os.tone import tone_signals  # noqa: E402

# Hash prefixes with known split membership: 0x00000000 % 100 = 0 (holdout
# at 20 pct), 0xffffffff % 100 = 95 (train).
TRAIN_HASH = "ffffffff" + "0" * 56
HOLDOUT_HASH = "00000000" + "0" * 56


def make_chunk(
    text: str,
    hint: str = "test friend",
    tier: int = 1,
    audience: str = "friend-family",
    medium: str = "dm",
    goal: str = "connect",
    train: bool = True,
    year: int = 2025,
    doc_type: str | None = None,
) -> dict:
    # doc_type None omits the key entirely, modeling chunks ingested
    # before context.doc_type existed.
    prefix = TRAIN_HASH if train else HOLDOUT_HASH
    body_hash = prefix[:8] + hashlib.sha256(text.encode()).hexdigest()[:56]
    chunk = {
        "id": body_hash[:16],
        "text": text,
        "hash": body_hash,
        "tier": tier,
        "provenance": {
            "source_type": "instagram_dm",
            "origin_file": "messages.json",
            "export_id": "synthetic",
            "timestamp": f"{year}-06-15T12:00:00",
            "extractor": "tests.synthetic@1.0",
        },
        "context": {
            "channel": "chat",
            "audience": audience,
            "medium": medium,
            "relationship_hint": hint,
            "goal": goal,
            "tone_signals": tone_signals(text),
            "extra": {},
            "inference": "heuristic-v1",
        },
        "schema_version": "1.0",
    }
    if doc_type is not None:
        chunk["context"]["doc_type"] = doc_type
    return chunk


def corpus_of(n: int, **kwargs) -> list[dict]:
    return [
        make_chunk(
            f"hey this is synthetic test message number {i}, pretty casual and "
            f"short. see you soon!",
            **kwargs,
        )
        for i in range(n)
    ]


def test_recipient_deltas_respect_support_gate():
    chunks = corpus_of(50, hint="big friend") + corpus_of(3, hint="small friend")
    artifact = mine_recipient_deltas(chunks, min_chunks=40, min_weighted_words=100)
    recipients = artifact["data"]["recipients"]
    assert "big friend" in recipients
    assert "small friend" not in recipients
    entry = recipients["big friend"]
    assert entry["n_chunks"] == 50
    assert set(entry["axis_delta"]) == {
        "rhetorical_pace", "risk_tolerance", "sentence_rhythm",
        "escalation_pattern", "hedging_behavior", "editorial_register",
    }
    for delta in entry["axis_delta"].values():
        assert -0.35 <= delta <= 0.35


def test_tier4_chunks_carry_zero_weight():
    chunks = corpus_of(60, hint="undated friend", tier=4)
    try:
        mine_recipient_deltas(chunks, min_chunks=1, min_weighted_words=1)
        raised = False
    except ValueError:
        raised = True
    assert raised, "tier-4-only input has no weighted data and must fail fast"


def test_holdout_chunks_excluded_from_mining():
    chunks = corpus_of(50, hint="train friend", train=True) + corpus_of(
        50, hint="holdout friend", train=False
    )
    artifact = mine_recipient_deltas(chunks, min_chunks=10, min_weighted_words=10)
    assert "train friend" in artifact["data"]["recipients"]
    assert "holdout friend" not in artifact["data"]["recipients"]
    assert artifact["data"]["global"]["n_chunks"] == 50


def test_email_recipients_aggregate_by_domain():
    chunks = corpus_of(30, hint="pal.one@synthetic-example.com") + corpus_of(
        30, hint="pal.two@synthetic-example.com"
    )
    artifact = mine_recipient_deltas(chunks, min_chunks=40, min_weighted_words=10)
    assert artifact["data"]["recipients"] == {}
    assert "synthetic-example.com" in artifact["data"]["domains"]
    assert artifact["data"]["domains"]["synthetic-example.com"]["n_chunks"] == 60


def test_context_profiles_group_by_audience_medium_goal_pair():
    chunks = corpus_of(45, audience="friend-family", medium="dm", goal="connect")
    chunks += corpus_of(45, audience="peer", medium="email", goal="inform")
    artifact = mine_context_profiles(chunks, min_chunks=40)
    data = artifact["data"]
    assert set(data["audiences"]) == {"friend-family", "peer"}
    assert set(data["media"]) == {"dm", "email"}
    assert set(data["goals"]) == {"connect", "inform"}
    assert set(data["pairs"]) == {"friend-family|dm", "peer|email"}
    group = data["audiences"]["friend-family"]
    assert group["n_chunks"] == 45
    assert "exclaim_per_100w" in group["tone_mean"]


def test_context_profiles_group_by_doc_type():
    chunks = corpus_of(
        45, audience="external", medium="script", doc_type="scripts"
    )
    chunks += corpus_of(45)  # old-style chunks: no doc_type key at all
    artifact = mine_context_profiles(chunks, min_chunks=40)
    data = artifact["data"]
    assert set(data["doc_types"]) == {"scripts"}
    assert data["doc_types"]["scripts"]["n_chunks"] == 45
    assert data["global"]["n_chunks"] == 90, "old chunks still count globally"


def test_empty_doc_type_not_grouped():
    chunks = corpus_of(45, doc_type="")
    artifact = mine_context_profiles(chunks, min_chunks=10)
    assert artifact["data"]["doc_types"] == {}


def test_group_profile_tolerates_artifact_without_doc_types():
    # Artifacts mined before doc_type existed have no doc_types key.
    old_artifact_data = {"audiences": {"peer": {"n_chunks": 50}}}
    assert group_profile(old_artifact_data, "doc_types", "scripts") is None
    assert group_profile(old_artifact_data, "audiences", "peer") is not None


def test_unknown_goal_not_grouped():
    chunks = corpus_of(45, goal="unknown")
    artifact = mine_context_profiles(chunks, min_chunks=10)
    assert artifact["data"]["goals"] == {}


def test_artifacts_round_trip_through_loader(tmp_path):
    chunks = corpus_of(50, hint="loop friend")
    mined_dir = tmp_path / "mined"
    mined_dir.mkdir()
    for artifact, name in (
        (mine_recipient_deltas(chunks, min_chunks=10, min_weighted_words=10),
         "recipient_deltas.json"),
        (mine_context_profiles(chunks, min_chunks=10), "context_profiles.json"),
    ):
        (mined_dir / name).write_text(json.dumps(artifact))

    loaded = load_artifacts(str(mined_dir))
    assert loaded.recipient_deltas is not None
    assert "loop friend" in loaded.recipient_deltas["recipients"]
    assert loaded.context_profiles is not None
    assert loaded.ngram_banned == []
    assert "recipient_deltas" in loaded.meta


def test_loader_rejects_wrong_artifact_and_version():
    good = {"artifact": "recipient_deltas", "version": "1.0", "data": {}}
    validate_artifact(good, "recipient_deltas")
    bad_name = dict(good, artifact="something_else")
    bad_version = dict(good, version="9.9")
    for bad, expected in ((bad_name, "recipient_deltas"), (bad_version, "recipient_deltas")):
        try:
            validate_artifact(bad, expected)
            raised = False
        except ValueError:
            raised = True
        assert raised


def test_loader_handles_missing_dir():
    loaded = load_artifacts(None)
    assert loaded.recipient_deltas is None
    loaded = load_artifacts("/nonexistent/mined/dir")
    assert loaded.context_profiles is None


def test_cli_run_writes_artifacts(tmp_path):
    chunks_dir = tmp_path / "corpus" / "chunks"
    chunks_dir.mkdir(parents=True)
    with open(chunks_dir / "synthetic.jsonl", "w") as f:
        for chunk in corpus_of(50, hint="cli friend"):
            f.write(json.dumps(chunk) + "\n")

    out = tmp_path / "corpus" / "mined"
    code = mine_main([
        "run", "--job", "all",
        "--corpus-dir", str(tmp_path / "corpus"),
        "--out", str(out),
    ])
    assert code == 0
    assert (out / "recipient_deltas.json").exists()
    assert (out / "context_profiles.json").exists()

    assert mine_main(["status", "--out", str(out)]) == 0
    assert mine_main(["run", "--job", "bogus", "--corpus-dir",
                      str(tmp_path / "corpus"), "--out", str(out)]) == 2


# --- n-gram anti-pattern mining -------------------------------------------

from mine.contrast import generate_contrast, load_contrast  # noqa: E402
from mine.ngrams import load_never_ban, mine_ngram_diffs, tokenize  # noqa: E402

SLOP = (
    "I hope this finds you well. Let's delve into the exciting updates and "
    "circle back on the synergies we discussed."
)


def contrast_of(n: int, text: str = SLOP) -> list[str]:
    return [text for _ in range(n)]


def test_cli_gate_job_writes_artifact_when_corpus_present(tmp_path):
    corpus_dir = tmp_path / "corpus"
    chunks_dir = corpus_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    with open(chunks_dir / "synthetic.jsonl", "w") as f:
        for chunk in corpus_of(60, hint="cli friend"):
            f.write(json.dumps(chunk) + "\n")
    sample = (REPO_ROOT / "data" / "sample_corpus.txt").read_text()
    (corpus_dir / "voice_corpus.txt").write_text(sample)

    out = corpus_dir / "mined"
    assert mine_main([
        "run", "--job", "gate", "--corpus-dir", str(corpus_dir),
        "--out", str(out),
    ]) == 0
    assert (out / "gate_calibration.json").exists()
    loaded = load_artifacts(str(out))
    assert loaded.gate_calibration["cells"]["chat|friend-family"]["n"] == 60


def test_gate_calibration_percentiles_train_split_only(tmp_path):
    from mine.gate_calibration import mine_gate_calibration

    corpus = str(REPO_ROOT / "data" / "sample_corpus.txt")
    chunks = corpus_of(60) + corpus_of(10, train=False)
    # A second, thin cell stays below min_chunks and must not emit.
    chunks += corpus_of(5, audience="peer", medium="email")
    artifact = mine_gate_calibration(
        iter(chunks), corpus_path=corpus, mined_dir=None
    )
    validate_artifact(artifact, "gate_calibration")
    cells = artifact["data"]["cells"]
    assert list(cells) == ["chat|friend-family"]
    cell = cells["chat|friend-family"]
    assert cell["n"] == 60  # holdout chunks never counted
    assert 0.0 <= cell["p25"] <= cell["p40"] <= cell["p50"] <= 1.0
    assert artifact["params"]["min_chunks"] == 50


def test_gate_calibration_round_trips_through_loader(tmp_path):
    from mine.gate_calibration import mine_gate_calibration

    corpus = str(REPO_ROOT / "data" / "sample_corpus.txt")
    artifact = mine_gate_calibration(
        iter(corpus_of(60)), corpus_path=corpus, mined_dir=None
    )
    (tmp_path / "gate_calibration.json").write_text(json.dumps(artifact))
    loaded = load_artifacts(str(tmp_path))
    assert loaded.gate_calibration is not None
    assert loaded.gate_calibration["cells"]["chat|friend-family"]["n"] == 60
    assert loaded.meta["gate_calibration"]["miner"] == "mine.gate_calibration@1.0"


def test_gate_calibration_loader_rejects_malformed_cells(tmp_path):
    bad = {
        "artifact": "gate_calibration", "version": "1.0",
        "generated_at": "2026-07-07", "miner": "t",
        "data": {"cells": {"chat|friend-family": {"n": "sixty"}}},
    }
    (tmp_path / "gate_calibration.json").write_text(json.dumps(bad))
    try:
        load_artifacts(str(tmp_path))
    except ValueError as err:
        assert "cells" in str(err)
    else:
        raise AssertionError("malformed cells map must fail fast")


def test_tokenize_keeps_internal_apostrophes():
    assert tokenize("Let's ship it, don't wait!") == [
        "let's", "ship", "it", "don't", "wait",
    ]


def test_planted_slop_phrase_gets_banned():
    self_chunks = corpus_of(50, hint="ngram friend")
    artifact = mine_ngram_diffs(self_chunks, contrast_of(10))
    banned_phrases = [e["ngram"] for e in artifact["data"]["banned"]]
    assert any("delve" in p for p in banned_phrases)
    for entry in artifact["data"]["banned"]:
        assert entry["log_odds"] >= 2.0
        assert entry["contrast_count"] >= 5


def test_never_ban_guard_protects_unigrams():
    self_chunks = corpus_of(50, hint="guard friend")
    contrast = contrast_of(10, "well well well well well well")
    with_guard = mine_ngram_diffs(self_chunks, contrast, never_ban={"well"})
    unigrams = [e["ngram"] for e in with_guard["data"]["banned"] if e["n"] == 1]
    assert "well" not in unigrams


def test_self_corpus_phrases_not_banned():
    # A phrase common in BOTH corpora has low log odds and stays allowed.
    phrase = "see you soon"
    self_chunks = corpus_of(50, hint="both friend")  # corpus_of text ends "see you soon!"
    artifact = mine_ngram_diffs(self_chunks, contrast_of(10, phrase))
    banned_phrases = [e["ngram"] for e in artifact["data"]["banned"]]
    assert phrase not in banned_phrases


def test_subgrams_of_banned_phrases_are_suppressed():
    self_chunks = corpus_of(50, hint="dedup friend")
    artifact = mine_ngram_diffs(self_chunks, contrast_of(10))
    phrases = [e["ngram"] for e in artifact["data"]["banned"]]
    padded = [f" {p} " for p in phrases]
    for p in phrases:
        containers = [q for q in padded if f" {p} " in q and q.strip() != p]
        assert not containers, f"'{p}' is a sub-gram of a kept phrase"


def test_empty_contrast_fails_fast():
    try:
        mine_ngram_diffs(corpus_of(10), [])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_load_contrast_parses_txt_and_jsonl(tmp_path):
    txt = tmp_path / "seed.txt"
    txt.write_text("# comment line\nfirst passage line one\nline two\n\nsecond passage\n")
    jl = tmp_path / "generated.jsonl"
    jl.write_text('{"text": "third passage"}\n\n{"text": ""}\n')
    passages = load_contrast([str(txt), str(jl), str(tmp_path / "missing.txt")])
    assert passages == [
        "first passage line one line two", "second passage", "third passage",
    ]


def test_load_never_ban(tmp_path):
    guard = tmp_path / "never_ban.txt"
    guard.write_text("# guard\nthe\nAnd\n\nwell\n")
    assert load_never_ban(str(guard)) == {"the", "and", "well"}


def test_contrast_gen_refuses_offline(tmp_path):
    # The whole test module sets VOICE_OS_OFFLINE=1.
    try:
        generate_contrast(1, str(tmp_path / "gen.jsonl"))
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_ngram_artifact_round_trips_through_loader(tmp_path):
    artifact = mine_ngram_diffs(corpus_of(50, hint="rt friend"), contrast_of(10))
    mined_dir = tmp_path / "mined"
    mined_dir.mkdir()
    (mined_dir / "ngram_banned.json").write_text(json.dumps(artifact))
    loaded = load_artifacts(str(mined_dir))
    assert loaded.ngram_banned
    assert all(isinstance(p, str) for p in loaded.ngram_banned)


def test_cli_ngrams_job(tmp_path):
    chunks_dir = tmp_path / "corpus" / "chunks"
    chunks_dir.mkdir(parents=True)
    with open(chunks_dir / "synthetic.jsonl", "w") as f:
        for chunk in corpus_of(50, hint="cli ngram friend"):
            f.write(json.dumps(chunk) + "\n")
    seed = tmp_path / "seed.txt"
    seed.write_text("\n\n".join(contrast_of(10)))

    out = tmp_path / "corpus" / "mined"
    code = mine_main([
        "run", "--job", "ngrams",
        "--corpus-dir", str(tmp_path / "corpus"),
        "--out", str(out),
        "--contrast", str(seed),
    ])
    assert code == 0
    assert (out / "ngram_banned.json").exists()


def test_tokenize_normalizes_curly_apostrophes():
    assert tokenize("Don’t let’s") == ["don't", "let's"]


def test_curly_apostrophe_contractions_hit_never_ban_guard():
    self_chunks = corpus_of(50, hint="curly friend")
    contrast = contrast_of(10, "don’t don’t don’t don’t don’t don’t")
    artifact = mine_ngram_diffs(self_chunks, contrast, never_ban={"don't"})
    assert "don't" not in [e["ngram"] for e in artifact["data"]["banned"]]


def test_load_contrast_skips_malformed_jsonl_lines(tmp_path):
    jl = tmp_path / "generated.jsonl"
    jl.write_text('{"text": "good passage"}\n{truncated garbage\n"bare string"\n')
    assert load_contrast([str(jl)]) == ["good passage"]


def test_self_counting_restricted_to_contrast_candidates():
    from mine.ngrams import _count_ngrams

    contrast_counts, _ = _count_ngrams([(tokenize(SLOP), 1.0)] * 10, 4)
    candidates = {g for g, c in contrast_counts.items() if c >= 5}
    self_counts, total = _count_ngrams(
        [(tokenize("a completely different self text with delve inside"), 1.0)],
        4,
        only=candidates,
    )
    assert total > 0
    assert set(self_counts) <= candidates
