"""Tests for content conservation (voice_os/conservation.py) and its
wiring into the product graph.

Fixtures are the verbatim receipts from the 2026-07-08 field report
(feedback/2026-07-08-storytellermitch-site-pass.md): the builder-turn
invented opinions, the stream-launch-night fabricated closer, the
mong-kok quoted pull, and the dropped-qualifier cases. Each failure the
report documents must be caught here so the defect class stays closed.

Module tests are stdlib-only; graph tests skip without langgraph,
matching tests/test_product.py conventions.
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

from voice_os import conservation  # noqa: E402
from voice_os.product.state import build_result, initial_state  # noqa: E402

CORPUS = str(REPO_ROOT / "data" / "sample_corpus.txt")
BANNED = str(REPO_ROOT / "data" / "banned_list.txt")
MINED = str(REPO_ROOT / "tests" / "fixtures" / "mined")

# builder-turn input (field report class 1 receipt): the passage whose
# redraft invented four opinions and still passed at fidelity 0.718.
BUILDER_TURN = (
    "After sixteen years in newsrooms and communications, I turned the "
    "production instinct on my own job search. I forked an open-source "
    "project called career-ops, which was a scaffold, and extended it into "
    "a production pipeline. It grew into a fleet of roughly fifty scheduled "
    "agents running unattended on launchd: scanning portals, triaging "
    "postings, batch-evaluating roles, rebuilding a live dashboard, and "
    "reporting dead-man heartbeats so silence never masquerades as health. "
    "Running it taught me that the hard problem isn't generation quality. "
    "It's coherence at scale, catching outputs that silently drift from the "
    "source. The clearest lesson: tooling doesn't protect you from "
    "architectural confusion. It removes friction from execution, which "
    "means unclear thinking ships faster and breaks harder. So I draw the "
    "stage boundaries before I build, and I decide what should be automated "
    "out of human attention entirely versus what requires judgment and must "
    "be protected from automation."
)

# The four sentences the pipeline invented (report receipt, verbatim).
BUILDER_TURN_INVENTED = (
    "Most job seekers are losing because they're doing it by hand.",
    "Generation quality is a solved problem, and everyone obsessing over "
    "it is fighting the last war.",
    "Good tools make bad architecture lethal.",
    "Blur that line and you're done.",
)

STREAM_LAUNCH = (
    "The night before our first broadcast, Osama bin Laden was killed in "
    "Abbottabad, Pakistan. Every piece of content we had prepared was "
    "suddenly worthless. We had roughly twelve hours to tear the launch "
    "show apart and rebuild it around a story no one had planned for, on a "
    "program that had never aired. We had treated the rundown as a "
    "hypothesis, permanently open to revision by incoming signal, so the "
    "worst-case scenario for a launch looked structurally like any other "
    "show day. The inputs changed. The workflow did not."
)

STREAM_LAUNCH_CLOSER = (
    "That's the whole point. Build the machine to expect chaos, and chaos "
    "becomes just another Tuesday."
)

# mong-kok pull (field report class 5 receipt): a verbatim broadcast
# quote, 13 words, quotation marks included.
MONG_KOK_PULL = (
    '"We\'re coming to you live from a backpack on the back of my '
    'producer."'
)


# ---------------------------------------------------------- claims diff


class TestUnsupportedSentences:
    def test_builder_turn_invented_opinions_are_caught(self):
        for sentence in BUILDER_TURN_INVENTED:
            flagged = conservation.unsupported_sentences(
                BUILDER_TURN, sentence
            )
            assert flagged, f"invented sentence not caught: {sentence}"
            assert flagged[0]["support"] < 0.5

    def test_stream_launch_appended_closer_is_caught(self):
        output = STREAM_LAUNCH + "\n\n" + STREAM_LAUNCH_CLOSER
        flagged = conservation.unsupported_sentences(STREAM_LAUNCH, output)
        assert any("Tuesday" in f["sentence"] for f in flagged)

    def test_input_own_sentences_are_never_flagged(self):
        for text in (BUILDER_TURN, STREAM_LAUNCH):
            assert conservation.unsupported_sentences(text, text) == []

    def test_faithful_paraphrase_is_not_flagged(self):
        # The human-approved rewrite style from the same session:
        # reordering and contraction without new claims.
        output = (
            "We had roughly twelve hours to tear the launch show apart and "
            "rebuild it around a story nobody had planned for. The inputs "
            "changed. The workflow did not."
        )
        assert conservation.unsupported_sentences(STREAM_LAUNCH, output) == []

    def test_short_fragments_are_cadence_not_claims(self):
        # Punch tags below the content-word floor are a style concern
        # (class 5), not a conservation finding.
        assert conservation.unsupported_sentences(
            STREAM_LAUNCH, "On purpose."
        ) == []


# ---------------------------------------------------------- quote spans


class TestQuoteSpans:
    def test_mong_kok_pull_stripped_quotes_are_a_violation(self):
        stripped = MONG_KOK_PULL.strip('"')
        assert conservation.quote_violations(MONG_KOK_PULL, stripped) == [
            MONG_KOK_PULL
        ]

    def test_verbatim_quote_is_clean(self):
        assert conservation.quote_violations(MONG_KOK_PULL, MONG_KOK_PULL) == []

    def test_reworded_quote_interior_is_a_violation(self):
        original = 'She said "we hit the design targets" on the call.'
        reworded = 'She said "we hit the targets" on the call.'
        assert conservation.quote_violations(original, reworded)

    def test_curly_and_straight_glyphs_are_equivalent(self):
        curly = "“We're coming to you live” she said."
        straight = '"We\'re coming to you live" she said.'
        assert conservation.quote_violations(curly, straight) == []

    def test_dropping_one_of_two_identical_quotes_is_a_violation(self):
        original = 'First "hold the line" then again "hold the line" close.'
        one_kept = 'First "hold the line" then again close.'
        assert conservation.quote_violations(original, one_kept) == [
            '"hold the line"'
        ]
        assert conservation.quote_violations(original, original) == []


# ------------------------------------------------------ dropped modifiers


class TestDroppedModifiers:
    def test_hurricane_maria_electrical_is_caught(self):
        flagged = conservation.dropped_modifiers(
            "a four-month electrical blackout hit the island",
            "a four-month blackout hit the island",
        )
        assert [f["modifier"] for f in flagged] == ["electrical"]

    def test_builder_turn_roughly_is_caught(self):
        flagged = conservation.dropped_modifiers(
            "a fleet of roughly fifty agents", "a fleet of fifty agents"
        )
        assert [f["modifier"] for f in flagged] == ["roughly"]

    def test_about_before_a_digit_is_caught(self):
        flagged = conservation.dropped_modifiers(
            "about 30 people came", "30 people came"
        )
        assert [f["modifier"] for f in flagged] == ["about"]

    def test_jazz_jennings_on_air_framing_label_is_caught(self):
        flagged = conservation.dropped_modifiers(
            "On air, the panel discussed her story.",
            "The panel discussed her story.",
        )
        assert flagged and flagged[0]["kind"] == "framing-label"

    def test_surviving_qualifiers_are_not_flagged(self):
        assert conservation.dropped_modifiers(BUILDER_TURN, BUILDER_TURN) == []

    def test_rephrased_verbs_before_numerals_are_out_of_scope(self):
        # "reached 50" -> "hit 50" is legitimate revision; verbs were the
        # only false positives in the 17-story calibration sweep.
        assert conservation.dropped_modifiers(
            "the channel reached 50 million views",
            "the channel hit 50 million views",
        ) == []

    def test_windows_do_not_cross_sentence_boundaries(self):
        assert conservation.dropped_modifiers(
            "It was about engineering. Three levers decided everything.",
            "It was engineering. Three levers decided everything.",
        ) == []

    def test_lost_numeral_is_the_number_checkers_finding(self):
        # If the numeral itself vanished there is no anchor to modify.
        assert conservation.dropped_modifiers(
            "roughly fifty agents", "a large fleet of agents"
        ) == []

    def test_word_to_digit_rewrite_still_anchors_the_hedge(self):
        # "fifty" rewritten to "50" is the same surviving numeral; the
        # dropped hedge must still be caught (and vice versa).
        flagged = conservation.dropped_modifiers(
            "a fleet of roughly fifty agents", "a fleet of 50 agents"
        )
        assert [f["modifier"] for f in flagged] == ["roughly"]
        flagged = conservation.dropped_modifiers(
            "about 12 hours of work", "twelve hours of work"
        )
        assert [f["modifier"] for f in flagged] == ["about"]
        # Hedge kept across the same rewrite: clean.
        assert conservation.dropped_modifiers(
            "a fleet of roughly fifty agents", "a fleet of roughly 50 agents"
        ) == []


# ------------------------------------------------------ format + diction


class TestFormatAndDiction:
    def test_markdown_bullets_into_prose_are_flagged(self):
        flagged = conservation.format_flags(
            "A trans Navy pilot. A veteran.",
            "- A trans Navy pilot.\n- A veteran.",
        )
        assert flagged

    def test_matching_prose_is_clean(self):
        assert conservation.format_flags(BUILDER_TURN, BUILDER_TURN) == []

    def test_scientology_hunting_escalation_is_flagged(self):
        # Reported as the exact output term, never a lossy stem.
        assert conservation.escalated_diction(
            "the organization pursuing its critics with legal tools",
            "the organization hunting its critics",
        ) == ["hunting"]

    def test_charged_term_already_in_input_is_not_flagged(self):
        assert conservation.escalated_diction(
            "they declared war on error", "they declared war on error"
        ) == []

    def test_input_stem_family_suppresses_output_variants(self):
        # "hunted" in the input covers "hunting" in the output; the
        # register was already there, not escalated.
        assert conservation.escalated_diction(
            "they hunted for answers", "they kept hunting for answers"
        ) == []

    def test_inflected_charged_terms_are_caught(self):
        cases = {
            "annihilating": "the review annihilating the proposal",
            "slaughtered": "the numbers slaughtered the forecast",
            "gossiping": "reporters gossiping about the filing",
            "weaponizing": "weaponizing the complaint process",
        }
        for word, output in cases.items():
            assert conservation.escalated_diction(
                "a neutral description of the events", output
            ) == [word], word

    def test_plain_dead_is_not_charged(self):
        # "deadly" is excluded from the lexicon precisely because its
        # stem collides with ordinary "dead".
        assert conservation.escalated_diction(
            "the line was quiet", "the line went dead"
        ) == []


# ---------------------------------------------------------- aggregate


def test_check_is_json_safe_and_stable_keyed():
    result = conservation.check(BUILDER_TURN, BUILDER_TURN_INVENTED[0])
    assert set(result) == {
        "unsupported_sentences",
        "quote_violations",
        "dropped_modifiers",
        "format_flags",
        "diction_flags",
    }
    json.dumps(result)


# ------------------------------------------------------------- graph


def _gate_state(input_text, draft, *, redraft, revision_count=0,
                max_revisions=2):
    """A qa_gate-ready state over the synthetic fixtures with the gate
    threshold floored, isolating the conservation decision path."""
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
        max_revisions=max_revisions,
        redraft=redraft,
    )
    state.update(
        target_profile=dict(q.target_profile),
        baseline_mean=dict(model.baseline.mean),
        baseline_std=dict(model.baseline.std),
        banned=[],
        current_draft=draft,
        revision_count=revision_count,
        gate_threshold=0.0001,  # voice gate always passes; conservation decides
    )
    return state


def test_redraft_blocks_pass_on_invented_content():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    draft = BUILDER_TURN + " " + BUILDER_TURN_INVENTED[2]
    update = graph_module.qa_gate(
        _gate_state(BUILDER_TURN, draft, redraft=True)
    )
    assert update["qa_decision"] == "revise"
    assert update["conservation"]["unsupported_sentences"]
    assert any(
        "does not support" in signal for signal in update["revision_signals"]
    )


def test_redraft_rejects_when_revisions_exhausted():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    draft = BUILDER_TURN + " " + BUILDER_TURN_INVENTED[0]
    update = graph_module.qa_gate(
        _gate_state(
            BUILDER_TURN, draft, redraft=True,
            revision_count=2, max_revisions=2,
        )
    )
    assert update["qa_decision"] == "reject"


def test_compose_mode_reports_without_blocking():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    draft = BUILDER_TURN + " " + BUILDER_TURN_INVENTED[0]
    update = graph_module.qa_gate(
        _gate_state(BUILDER_TURN, draft, redraft=False)
    )
    assert update["qa_decision"] == "pass"
    assert update["conservation"]["unsupported_sentences"]


def test_quote_violation_blocks_pass_in_both_modes():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    original = (
        "Anchor Mariana Atencio confirmed it for viewers on the broadcast "
        "itself: \"we're coming to you live from a backpack on the back of "
        "my producer.\" That assignment compressed everything years of "
        "live news taught me about holding a through-line while conditions "
        "change around the broadcast."
    )
    modified = original.replace('"', "")
    for redraft in (True, False):
        update = graph_module.qa_gate(
            _gate_state(original, modified, redraft=redraft)
        )
        assert update["qa_decision"] == "revise"
        assert update["conservation"]["quote_violations"]


def test_short_unchanged_input_passes_on_the_conservative_floor():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    state = _gate_state(MONG_KOK_PULL, MONG_KOK_PULL, redraft=True)
    state["gate_threshold"] = 0.9999  # voice gate alone would cycle
    update = graph_module.qa_gate(state)
    assert update["qa_decision"] == "pass"
    assert update["conservation"]["input_retained"] is True


def test_short_input_guard_retains_input_against_weak_rewrites():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    state = _gate_state(MONG_KOK_PULL, MONG_KOK_PULL, redraft=True)
    # Quote stripped: retained.
    text, note = graph_module._short_input_guard(
        state, MONG_KOK_PULL.strip('"')
    )
    assert text == MONG_KOK_PULL and "quoted span" in note
    # Invented cynicism (report receipt): "I treat voice as
    # infrastructure, not magic." came back with a fabricated second
    # beat. Retained.
    infra = "I treat voice as infrastructure, not magic."
    text, note = graph_module._short_input_guard(
        _gate_state(infra, infra, redraft=True),
        "Voice is the plumbing. Nobody bothers to check it.",
    )
    assert text == infra and "unentailed" in note
    # Long inputs are out of the guard's scope.
    text, note = graph_module._short_input_guard(
        _gate_state(BUILDER_TURN, BUILDER_TURN, redraft=True),
        BUILDER_TURN + " Extra.",
    )
    assert note is None


def test_short_input_guard_lets_compose_briefs_expand():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    brief = "quick note to say the launch plan looks right, one timing question"
    expansion = (
        "The launch plan looks right to me. One question on timing: does "
        "the Thursday window still hold if the review slips a day?"
    )
    state = _gate_state(brief, brief, redraft=False)
    # Compose semantics: the brief is meant to be expanded, so the
    # entailment and fidelity-margin retentions must not fire...
    text, note = graph_module._short_input_guard(state, expansion)
    assert text == expansion and note is None
    # ...but quote spans are protected in every mode.
    quoted = 'say the launch plan "looks right, full stop" to the team'
    text, note = graph_module._short_input_guard(
        _gate_state(quoted, quoted, redraft=False),
        "say the launch plan looks right to the team",
    )
    assert text == quoted and "quoted span" in note


def test_dropped_modifier_is_advisory_and_signaled():
    pytest.importorskip("langgraph")
    from voice_os.product import graph as graph_module

    original = (
        "It grew into a fleet of roughly fifty scheduled agents running "
        "unattended on launchd, scanning portals and triaging postings "
        "while the dashboard rebuilt itself and heartbeats reported so "
        "silence never masqueraded as health."
    )
    dropped = original.replace("roughly ", "")
    update = graph_module.qa_gate(
        _gate_state(original, dropped, redraft=True)
    )
    assert update["qa_decision"] == "pass"  # advisory, never blocking
    assert [
        f["modifier"] for f in update["conservation"]["dropped_modifiers"]
    ] == ["roughly"]
    assert any("roughly" in s for s in update["revision_signals"])


def test_envelope_conservation_field_is_additive():
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
            redraft=True,
        ),
        "run-test",
    )
    # Existing caller surface is untouched...
    for key in ("mode", "decision", "fidelity", "banned_hits", "output_text"):
        assert key in envelope
    # ...and the new field is additive with the redraft contract echoed.
    assert envelope["conservation"]["redraft"] is True
    json.dumps(envelope)


def test_draft_offline_envelope_carries_conservation(tmp_path):
    pytest.importorskip("langgraph")
    import voice_os

    result = voice_os.draft(
        "Quick note to say the launch plan looks right, one timing "
        "question for the team about the review window on Thursday.",
        channel="email",
        audience="peer",
        redraft=True,
        corpus_path=CORPUS,
        chunks_dir=None,
        mined_dir=MINED,
        banned_path=BANNED,
        kb_dir=str(tmp_path / "kb"),
        var_dir=str(tmp_path / "var"),
    )
    conserve = result["conservation"]
    assert conserve["redraft"] is True
    assert "unsupported_sentences" in conserve
    assert "quote_violations" in conserve
    json.dumps(result)
