"""Offline unit tests for the VoiceModel facade.

Uses the synthetic sample corpus plus fictional Test Person mined
artifacts committed under tests/fixtures/mined/. Run with:
python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["VOICE_OS_OFFLINE"] = "1"

from voice_os import VoiceModel, load_corpus  # noqa: E402
from voice_os.calibration import calibrate  # noqa: E402
from voice_os.tone import ToneProfile  # noqa: E402

CORPUS = str(REPO_ROOT / "data" / "sample_corpus.txt")
BANNED = str(REPO_ROOT / "data" / "banned_list.txt")
MINED = str(REPO_ROOT / "tests" / "fixtures" / "mined")
DRAFT_GOOD = (REPO_ROOT / "data" / "sample_draft_good.txt").read_text()


def load_model(mined_dir: str | None = MINED) -> VoiceModel:
    return VoiceModel.load(
        CORPUS, chunks_dir=None, mined_dir=mined_dir, banned_path=BANNED,
    )


class TestVoiceModelLoad(unittest.TestCase):
    def test_loads_with_mined_artifacts(self):
        model = load_model()
        self.assertIsNotNone(model.mined.recipient_deltas)
        self.assertIsNotNone(model.mined.context_profiles)

    def test_degrades_without_mined_dir(self):
        model = load_model(mined_dir=None)
        self.assertIsNone(model.mined.context_profiles)
        q = model.query(audience="friend-family")
        self.assertEqual(q.sources["audience"], "heuristic")
        self.assertIsNone(q.tone)
        self.assertEqual(q.sources["tone"], "absent")


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.model = load_model()

    def test_query_returns_every_field(self):
        q = self.model.query(audience="friend-family", medium="dm", goal="connect")
        self.assertEqual(q.context["audience"], "friend-family")
        self.assertEqual(set(q.target_profile), {
            "rhetorical_pace", "risk_tolerance", "sentence_rhythm",
            "escalation_pattern", "hedging_behavior", "editorial_register",
        })
        self.assertIsInstance(q.tone, ToneProfile)
        self.assertTrue(q.banned)
        self.assertEqual(q.exemplars, [])  # chunks_dir=None
        self.assertIn("communication goal: connect", q.guidance)
        self.assertIn("voice_os_version", q.meta)

    def test_mined_audience_reports_mined_source(self):
        q = self.model.query(audience="friend-family")
        self.assertEqual(q.sources["audience"], "mined")
        base = self.model.query()
        self.assertEqual(base.sources["audience"], "heuristic")  # peer not mined

    def test_tone_prefers_most_specific_group(self):
        q = self.model.query(audience="friend-family", medium="dm")
        # pair profile has exclaim mean 2.9; audience-only has 2.8
        self.assertAlmostEqual(q.tone.mean["exclaim_per_100w"], 2.9)
        q_audience = self.model.query(audience="friend-family")
        self.assertAlmostEqual(q_audience.tone.mean["exclaim_per_100w"], 2.8)

    def test_recipient_delta_applies(self):
        # Default (peer/email) context so the target is far from the clamp
        # floor and the recipient's -0.2 editorial_register delta is visible.
        without = self.model.query()
        with_recipient = self.model.query(recipient="Test Person")
        self.assertEqual(with_recipient.sources["recipient"], "mined")
        self.assertLess(
            with_recipient.target_profile["editorial_register"],
            without.target_profile["editorial_register"],
        )

    def test_recipient_falls_back_to_email_domain(self):
        q = self.model.query(recipient="unknown.name@synthetic-example.com")
        self.assertEqual(q.sources["recipient"], "mined")

    def test_unknown_recipient_is_absent(self):
        q = self.model.query(recipient="nobody i know")
        self.assertEqual(q.sources["recipient"], "absent")

    def test_default_query_matches_legacy_calibrate(self):
        model = load_model(mined_dir=None)
        q = model.query()
        self.assertEqual(
            q.target_profile,
            calibrate(load_corpus(CORPUS), "email", "peer", "standard"),
        )

    def test_invalid_context_raises(self):
        with self.assertRaises(ValueError):
            self.model.query(goal="world-domination")

    def test_merged_banned_includes_mined_ngrams(self):
        model = load_model()
        model.mined.ngram_banned = ["delve into", "leverage"]  # leverage dupes
        merged = model.banned
        self.assertIn("delve into", merged)
        self.assertEqual(merged.count("leverage"), 1)


class TestGateAndRun(unittest.TestCase):
    def setUp(self):
        self.model = load_model()

    def test_gate_draft_returns_gate_result(self):
        q = self.model.query()
        result = self.model.gate_draft(DRAFT_GOOD, q)
        self.assertIn(result.decision, ("pass", "cycle"))
        self.assertGreaterEqual(result.fidelity, 0.0)

    def test_run_produces_pipeline_report(self):
        report = self.model.run(DRAFT_GOOD, audience="friend-family", goal="connect")
        self.assertEqual(report["classification"]["goal"], "connect")
        self.assertEqual(report["sources"]["audience"], "mined")
        self.assertIn(report["final"]["decision"], ("pass", "cycle"))
        self.assertEqual(report["meta"]["mode"], "offline")


if __name__ == "__main__":
    unittest.main()
