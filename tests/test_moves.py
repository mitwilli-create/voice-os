"""Tests for signature-move detection and the avoid-list contract
(voice_os/moves.py + product graph wiring).

Fixtures are the class-6 receipts from the 2026-07-08 field report:
the fragment date openers, staccato "No X." runs, and punch-tag
closers that converged across the site pass's 17 independent runs.

Module tests are stdlib-only; graph tests skip without langgraph.
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

from voice_os import moves  # noqa: E402
from voice_os.product.state import build_result, initial_state  # noqa: E402

CORPUS = str(REPO_ROOT / "data" / "sample_corpus.txt")
BANNED = str(REPO_ROOT / "data" / "banned_list.txt")
MINED = str(REPO_ROOT / "tests" / "fixtures" / "mined")

# Receipt: the mong-kok voiced body opened "October 2014." and stacked
# "No satellite truck. No control room. No studio."
MONG_KOK_STYLE = (
    "October 2014. The Umbrella Revolution filled the streets of Hong "
    "Kong. I field-produced the coverage with no script. No satellite "
    "truck. No control room. No studio. The whole operation ran from a "
    "backpack on my back."
)


# ---------------------------------------------------------- detectors


class TestDetect:
    def test_fragment_date_opener_receipts(self):
        for opener in ("December 2007.", "October 2014.", "May 2, 2011."):
            found = moves.detect(opener + " The rest of the story follows "
                                 "in complete sentences here.")
            assert found.get("fragment-date-opener") == [opener], opener

    def test_full_sentence_date_opener_is_not_flagged(self):
        found = moves.detect(
            "In December 2007 I was the overnight producer at the station."
        )
        assert "fragment-date-opener" not in found

    def test_x_not_y_construction(self):
        found = moves.detect(
            "The output you measure is the content. Not the people who "
            "compound behind it."
        )
        assert found.get("x-not-y")

    def test_no_x_run_receipt(self):
        found = moves.detect(MONG_KOK_STYLE)
        assert "no-x-run" in found
        assert "fragment-date-opener" in found

    def test_lowercase_no_does_not_anchor_a_run(self):
        found = moves.detect(
            "There was no script and zero infrastructure to speak of. "
            "Nothing else about the day was unusual in any way."
        )
        assert "no-x-run" not in found

    def test_punch_tag_closer_receipts(self):
        for tag in ("That's the whole point.", "Every time.", "On purpose."):
            found = moves.detect(
                "We built the system to expect chaos from the start. " + tag
            )
            assert found.get("punch-tag-closer") == [tag], tag

    def test_fragment_closer(self):
        found = moves.detect(
            "The inputs changed under us all night. The workflow did not."
        )
        assert found.get("fragment-closer") == ["The workflow did not."]

    def test_full_sentence_closer_is_clean(self):
        found = moves.detect(
            "The inputs changed under us all night. The workflow held "
            "steady because we had designed it to absorb exactly that."
        )
        assert found == {}

    def test_quoted_closer_is_not_a_fragment_closer(self):
        found = moves.detect('She said it plainly. "Hold the line."')
        assert "fragment-closer" not in found


# ---------------------------------------------------------- avoid list


class TestValidateAvoid:
    def test_canonicalizes_case_and_underscores(self):
        assert moves.validate_avoid(
            ["PUNCH_TAG_CLOSER", "fragment-date-opener"]
        ) == ["punch-tag-closer", "fragment-date-opener"]

    def test_deduplicates(self):
        assert moves.validate_avoid(["x-not-y", "x_not_y"]) == ["x-not-y"]

    def test_unknown_move_raises_with_vocabulary(self):
        with pytest.raises(ValueError) as excinfo:
            moves.validate_avoid(["dramatic-pause"])
        assert "known moves" in str(excinfo.value)

    def test_guidance_covers_every_catalog_key(self):
        lines = moves.avoid_guidance(sorted(moves.CATALOG))
        assert len(lines) == len(moves.CATALOG)
        assert all(line.startswith("avoid this signature move:")
                   for line in lines)


# ------------------------------------------------------------- graph


def _gate_state(input_text, draft, *, avoid, revision_count=0):
    from voice_os.model import VoiceModel

    model = VoiceModel.load(
        CORPUS, chunks_dir=None, mined_dir=MINED, banned_path=BANNED
    )
    q = model.query()
    state = initial_state(
        input_text=input_text,
        channel="doc",
        audience="external",
        situation="standard",
        goal="connect",
        stakes="high",
        medium=None,
        max_revisions=2,
        avoid=avoid,
    )
    state.update(
        target_profile=dict(q.target_profile),
        baseline_mean=dict(model.baseline.mean),
        baseline_std=dict(model.baseline.std),
        banned=[],
        current_draft=draft,
        revision_count=revision_count,
        gate_threshold=0.0001,  # voice gate always passes; moves decide
    )
    return state


LONG_INPUT = (
    "In October 2014 the Umbrella Revolution filled the streets of Hong "
    "Kong and I field-produced the network's breaking coverage from "
    "inside the confrontations, coordinating three competing broadcast "
    "organizations along with multiple live feeds and real-time social "
    "data while the story kept changing on its way to air."
)


def test_avoided_move_blocks_pass_and_signals():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    update = graph_module.qa_gate(_gate_state(
        LONG_INPUT, MONG_KOK_STYLE, avoid=["fragment-date-opener"]
    ))
    assert update["qa_decision"] == "revise"
    assert update["signature_moves"]["violations"] == ["fragment-date-opener"]
    assert any("fragment-date-opener" in s for s in update["revision_signals"])


def test_unavoided_moves_are_reported_not_blocking():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    update = graph_module.qa_gate(_gate_state(LONG_INPUT, MONG_KOK_STYLE,
                                              avoid=[]))
    assert update["qa_decision"] == "pass"
    assert "fragment-date-opener" in update["signature_moves"]["detected"]
    assert update["signature_moves"]["violations"] == []


def test_avoided_move_rejects_when_revisions_exhausted():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    update = graph_module.qa_gate(_gate_state(
        LONG_INPUT, MONG_KOK_STYLE, avoid=["no-x-run"], revision_count=2
    ))
    assert update["qa_decision"] == "reject"


def test_avoid_guidance_reaches_the_generate_prompt(tmp_path):
    pytest.importorskip("langgraph")
    from unittest import mock

    from voice_os.product import graph as graph_module

    state = _gate_state(LONG_INPUT, LONG_INPUT, avoid=["punch-tag-closer"])
    captured = {}

    def fake_complete(system, prompt, max_tokens=2000):
        captured["prompt"] = prompt
        return None  # fall back offline; the prompt is what matters

    with mock.patch("voice_os.llm.complete", side_effect=fake_complete):
        graph_module.generate(state)
    assert "avoid this signature move" in captured["prompt"]
    assert "punch tag" in captured["prompt"]


def test_draft_validates_avoid_before_graph():
    import voice_os

    with pytest.raises(ValueError) as excinfo:
        voice_os.draft("some text to voice", avoid=["dramatic-pause"])
    assert "known moves" in str(excinfo.value)


def test_envelope_signature_moves_field_is_additive():
    envelope = build_result(
        initial_state(
            input_text="hello there team",
            channel="email",
            audience="peer",
            situation="standard",
            goal="unknown",
            stakes="routine",
            medium=None,
            max_revisions=2,
            avoid=["punch-tag-closer"],
        ),
        "run-test",
    )
    for key in ("mode", "decision", "fidelity", "banned_hits", "output_text",
                "conservation"):
        assert key in envelope
    assert envelope["signature_moves"]["avoid"] == ["punch-tag-closer"]
    assert envelope["signature_moves"]["violations"] == []
    json.dumps(envelope)


def test_cli_avoid_flag_parses_repeats_and_commas(tmp_path, capsys):
    pytest.importorskip("langgraph")
    import io

    from voice_os.product.cli import main

    argv = [
        "draft", "--corpus", CORPUS, "--mined-dir", MINED,
        "--banned-path", BANNED, "--kb-dir", str(tmp_path / "kb"),
        "--var-dir", str(tmp_path / "var"),
        "--avoid", "punch-tag-closer,fragment-date-opener",
        "--avoid", "x_not_y",
    ]
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(
        "A steady update on the plan with enough words to carry the "
        "message through the gate and back out the other side today."
    )
    try:
        code = main(argv)
    finally:
        sys.stdin = old_stdin
    out, _ = capsys.readouterr()
    envelope = json.loads(out)
    assert envelope["signature_moves"]["avoid"] == [
        "punch-tag-closer", "fragment-date-opener", "x-not-y"
    ]
    assert code in (0, 1)
