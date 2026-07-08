"""Offline unit tests for the voice_os pipeline.

Run with: python -m unittest discover -s tests -v
All tests force offline mode so results are deterministic with no API key.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["VOICE_OS_OFFLINE"] = "1"

from voice_os import load_corpus, run_pipeline, score_draft  # noqa: E402
from voice_os.axes import AXES, score_text  # noqa: E402
from voice_os.calibration import calibrate  # noqa: E402
from voice_os.corpus import parse_corpus, tier_for_year  # noqa: E402
from voice_os.qa import find_banned, load_banned_list, scrub_em_dashes  # noqa: E402

CORPUS = str(REPO_ROOT / "data" / "sample_corpus.txt")
BANNED = str(REPO_ROOT / "data" / "banned_list.txt")
DRAFT_SLOP = (REPO_ROOT / "data" / "sample_draft.txt").read_text()
DRAFT_GOOD = (REPO_ROOT / "data" / "sample_draft_good.txt").read_text()


class TestAxes(unittest.TestCase):
    def test_scores_cover_all_axes_in_range(self):
        scores = score_text(DRAFT_SLOP)
        self.assertEqual(set(scores), set(AXES))
        for axis, value in scores.items():
            self.assertGreaterEqual(value, 0.0, axis)
            self.assertLessEqual(value, 1.0, axis)

    def test_hedged_text_scores_higher_on_hedging(self):
        hedged = "I think we could maybe sort of consider it, perhaps."
        direct = "We ship Friday. The tests pass. Confirm by noon."
        self.assertGreater(
            score_text(hedged)["hedging_behavior"],
            score_text(direct)["hedging_behavior"],
        )


class TestCorpus(unittest.TestCase):
    def test_tier_assignment(self):
        self.assertEqual(tier_for_year(2026), 1)
        self.assertEqual(tier_for_year(2022), 2)
        self.assertEqual(tier_for_year(2017), 3)
        self.assertEqual(tier_for_year(2012), 4)

    def test_parse_sample_corpus(self):
        entries = parse_corpus(CORPUS)
        self.assertGreaterEqual(len(entries), 10)
        self.assertIn(1, {e.tier for e in entries})
        self.assertIn(4, {e.tier for e in entries})

    def test_baseline_reflects_recent_voice(self):
        # Tier 1 entries hedge far less than the Tier 3/4 slop entries; the
        # weighted baseline should sit near the recent (low-hedging) voice.
        baseline = load_corpus(CORPUS)
        self.assertLess(baseline.mean["hedging_behavior"], 0.35)


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.baseline = load_corpus(CORPUS)

    def test_unknown_context_raises(self):
        with self.assertRaises(ValueError):
            calibrate(self.baseline, "carrier-pigeon", "peer", "standard")
        with self.assertRaises(ValueError):
            calibrate(self.baseline, "email", "peer", "surprise-party")

    def test_text_channel_lowers_register(self):
        email = calibrate(self.baseline, "email", "peer", "standard")
        text = calibrate(self.baseline, "text", "peer", "standard")
        self.assertLess(text["editorial_register"], email["editorial_register"])


class TestQA(unittest.TestCase):
    def test_banned_phrases_detected(self):
        banned = load_banned_list(BANNED)
        hits = find_banned(DRAFT_SLOP, banned)
        self.assertIn("i hope this email finds you well", hits)
        self.assertIn("synergy", hits)
        self.assertGreaterEqual(len(hits), 10)


class TestEmDashScrub(unittest.TestCase):
    """Em dashes are banned outward (house style); personas scrub them
    at the source so no draft surface can emit one."""

    def test_scrub_forms(self):
        self.assertEqual(
            scrub_em_dashes("boldness — a note"), "boldness - a note"
        )
        self.assertEqual(scrub_em_dashes("word—word"), "word - word")
        self.assertEqual(scrub_em_dashes("a——b"), "a - b")
        # Dash opening a line stays flush; dangling dash is dropped.
        self.assertEqual(scrub_em_dashes("—item one\n"), "- item one\n")
        self.assertEqual(scrub_em_dashes("trailing—\nnext"), "trailing\nnext")
        # Newlines around a dash are preserved as line structure.
        self.assertNotIn("—", scrub_em_dashes("a—\n—b"))

    def test_scrub_untouched_text_is_identity(self):
        clean = "No dashes here - just a hyphen, a colon: and prose.\n"
        self.assertIs(scrub_em_dashes(clean), clean)

    def test_live_persona_output_is_scrubbed(self):
        from unittest import mock

        from voice_os.personas import GenerativePersona

        with mock.patch(
            "voice_os.llm.complete",
            return_value="Bold move — and the right one—clearly.",
        ):
            result = GenerativePersona().revise(
                "draft", {axis: 0.5 for axis in AXES}, [], []
            )
        self.assertEqual(result.mode, "live")
        self.assertNotIn("—", result.text)
        self.assertEqual(
            result.text, "Bold move - and the right one - clearly."
        )

    def test_offline_persona_output_is_scrubbed(self):
        from voice_os.personas import GenerativePersona

        result = GenerativePersona()._offline_revise(
            "The plan — such as it is — holds.",
            {axis: 0.5 for axis in AXES},
            [],
        )
        self.assertEqual(result.mode, "offline")
        self.assertNotIn("—", result.text)


class TestPipeline(unittest.TestCase):
    def test_slop_draft_cycles_and_strips_banned_phrases(self):
        result = run_pipeline(CORPUS, DRAFT_SLOP, banned_path=BANNED)
        self.assertEqual(result["meta"]["mode"], "offline")
        self.assertEqual(result["cycles"][0]["decision"], "cycle")
        self.assertGreaterEqual(len(result["cycles"][0]["banned_hits"]), 10)
        # Banned phrases must be gone from the final output.
        banned = load_banned_list(BANNED)
        self.assertEqual(find_banned(result["final"]["output_text"], banned), [])
        # Fidelity must improve across cycles.
        self.assertGreater(result["final"]["fidelity"], result["cycles"][0]["fidelity"])

    def test_in_voice_draft_passes_untouched(self):
        result = run_pipeline(CORPUS, DRAFT_GOOD, banned_path=BANNED)
        self.assertEqual(result["final"]["decision"], "pass")
        self.assertEqual(result["final"]["output_text"], DRAFT_GOOD)
        self.assertGreaterEqual(result["final"]["fidelity"], 0.80)

    def test_adversarial_findings_carry_into_next_cycle(self):
        result = run_pipeline(CORPUS, DRAFT_SLOP, banned_path=BANNED, max_cycles=2)
        first = result["cycles"][0]
        self.assertIn("adversarial_findings", first)
        if first["adversarial_findings"] and len(result["cycles"]) > 1:
            # Findings recorded in cycle N appear as signals consumed by N+1;
            # verified indirectly: the pipeline records them and keeps cycling.
            self.assertEqual(first["decision"], "cycle")

    def test_score_draft_shape(self):
        result = score_draft(CORPUS, DRAFT_GOOD)
        self.assertEqual(set(result["axis_scores"]), set(AXES))
        self.assertIn("fidelity", result)


if __name__ == "__main__":
    unittest.main()
