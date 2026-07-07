"""Offline unit tests for the extended-model foundations.

Covers contexts, tone metrics, holdout split, extended calibration, the
extended QA gate, and pipeline back-compat. Run with:
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

from voice_os import load_corpus, run_pipeline  # noqa: E402
from voice_os.calibration import calibrate, calibrate_extended  # noqa: E402
from voice_os.contexts import (  # noqa: E402
    GOALS,
    MEDIA,
    STAKES,
    VoiceContext,
    infer_stakes,
)
from voice_os.holdout import is_holdout  # noqa: E402
from voice_os.qa import gate, gate_extended  # noqa: E402
from voice_os.tone import (  # noqa: E402
    TONE_METRICS,
    ToneProfile,
    derive_metrics,
    tone_guidance,
    tone_signals,
)

CORPUS = str(REPO_ROOT / "data" / "sample_corpus.txt")
BANNED = str(REPO_ROOT / "data" / "banned_list.txt")
DRAFT_GOOD = (REPO_ROOT / "data" / "sample_draft_good.txt").read_text()


class TestVoiceContext(unittest.TestCase):
    def test_default_context_validates(self):
        VoiceContext().validate()

    def test_unknown_goal_raises(self):
        with self.assertRaises(ValueError):
            VoiceContext(goal="world-domination").validate()

    def test_unknown_stakes_raises(self):
        with self.assertRaises(ValueError):
            VoiceContext(stakes="extreme").validate()

    def test_unknown_medium_raises(self):
        with self.assertRaises(ValueError):
            VoiceContext(medium="telegraph").validate()

    def test_vocabularies_are_supersets_of_ingest_values(self):
        # heuristic-v1 goal set must map into GOALS with no translation
        for goal in ("inform", "connect", "coordinate", "request", "unknown"):
            self.assertIn(goal, GOALS)
        # MEDIA must cover every MEDIUM_BY_SOURCE value in ingest.enrich
        from ingest.enrich import MEDIUM_BY_SOURCE

        for medium in MEDIUM_BY_SOURCE.values():
            self.assertIn(medium, MEDIA)


class TestInferStakes(unittest.TestCase):
    def test_bad_news_situation_is_high(self):
        self.assertEqual(infer_stakes("Here is the update.", "bad-news"), "high")

    def test_critical_markers_win(self):
        self.assertEqual(
            infer_stakes("We need to involve legal before replying.", "standard"),
            "critical",
        )

    def test_casual_markers_lower(self):
        self.assertEqual(infer_stakes("lol no rush at all", "standard"), "low")

    def test_plain_text_is_routine(self):
        self.assertEqual(infer_stakes("Here are the meeting notes.", "standard"), "routine")

    def test_returns_valid_vocabulary(self):
        for text in ("legal notice", "deadline tomorrow", "lol", "notes"):
            self.assertIn(infer_stakes(text), STAKES)


class TestCalibrateExtended(unittest.TestCase):
    def setUp(self):
        self.baseline = load_corpus(CORPUS)

    def test_default_context_matches_legacy_calibrate(self):
        target, sources = calibrate_extended(self.baseline, VoiceContext())
        self.assertEqual(target, calibrate(self.baseline, "email", "peer", "standard"))
        self.assertEqual(sources["medium"], "absent")
        self.assertEqual(sources["goal"], "heuristic")

    def test_persuade_raises_risk_tolerance(self):
        base, _ = calibrate_extended(self.baseline, VoiceContext())
        persuade, _ = calibrate_extended(self.baseline, VoiceContext(goal="persuade"))
        self.assertGreater(persuade["risk_tolerance"], base["risk_tolerance"])

    def test_critical_stakes_lower_escalation(self):
        base, _ = calibrate_extended(self.baseline, VoiceContext())
        critical, _ = calibrate_extended(self.baseline, VoiceContext(stakes="critical"))
        self.assertLess(critical["escalation_pattern"], base["escalation_pattern"])

    def test_targets_stay_clamped(self):
        ctx = VoiceContext(
            channel="text", audience="friend-family", goal="connect",
            stakes="low", medium="story",
        )
        target, _ = calibrate_extended(self.baseline, ctx)
        for axis, value in target.items():
            self.assertGreaterEqual(value, 0.0, axis)
            self.assertLessEqual(value, 1.0, axis)

    def test_invalid_context_raises(self):
        with self.assertRaises(ValueError):
            calibrate_extended(self.baseline, VoiceContext(goal="nope"))


class TestTone(unittest.TestCase):
    def test_signals_keep_chunk_schema_keys(self):
        signals = tone_signals("Wait, really?! That is huge news. 🎉")
        self.assertEqual(
            set(signals),
            {"exclaim_per_100w", "question_ratio", "emoji_count",
             "avg_sentence_words", "caps_ratio", "word_count"},
        )

    def test_derive_metrics_normalizes_emoji(self):
        metrics = derive_metrics({"emoji_count": 2, "word_count": 50})
        self.assertEqual(metrics["emoji_per_100w"], 4.0)
        self.assertEqual(set(metrics), set(TONE_METRICS))

    def test_deviation_flagged_outside_tolerance(self):
        profile = ToneProfile(
            mean={m: 0.0 for m in TONE_METRICS},
            std={m: 0.0 for m in TONE_METRICS},
        )
        loud = {m: 0.0 for m in TONE_METRICS}
        loud["exclaim_per_100w"] = 12.0
        signals = profile.deviations(loud)
        self.assertTrue(any("exclaim_per_100w" in s for s in signals))
        self.assertTrue(all(isinstance(s, str) for s in signals))

    def test_no_deviation_within_tolerance(self):
        profile = ToneProfile(
            mean={m: 1.0 for m in TONE_METRICS},
            std={m: 5.0 for m in TONE_METRICS},
        )
        self.assertEqual(profile.deviations({m: 1.5 for m in TONE_METRICS}), [])

    def test_partial_profile_degrades_gracefully(self):
        profile = ToneProfile(mean={"caps_ratio": 0.05}, std={"caps_ratio": 0.0})
        signals = profile.deviations({"caps_ratio": 0.9, "exclaim_per_100w": 50.0})
        self.assertEqual(len(signals), 1)
        self.assertIn("caps_ratio", signals[0])

    def test_tone_guidance_runs_on_text(self):
        profile = ToneProfile(
            mean={m: 0.0 for m in TONE_METRICS},
            std={m: 100.0 for m in TONE_METRICS},
        )
        self.assertEqual(tone_guidance(profile, DRAFT_GOOD), [])


class TestHoldout(unittest.TestCase):
    def test_deterministic(self):
        h = "ab34f2c19e77d012"
        self.assertEqual(is_holdout(h), is_holdout(h))

    def test_split_rate_approximates_pct(self):
        import hashlib

        hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(2000)]
        rate = sum(1 for h in hashes if is_holdout(h, 20)) / len(hashes)
        self.assertGreater(rate, 0.15)
        self.assertLess(rate, 0.25)

    def test_boundary_percentages(self):
        h = "ffffffff00000000"
        self.assertFalse(is_holdout(h, 0))
        self.assertTrue(is_holdout(h, 100))

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            is_holdout("ab34f2c19e77d012", 101)
        with self.assertRaises(ValueError):
            is_holdout("ab", 20)


class TestGateExtended(unittest.TestCase):
    def setUp(self):
        self.baseline = load_corpus(CORPUS)
        self.target = calibrate(self.baseline, "email", "peer", "standard")
        # Scores exactly at target: the axis gate passes by construction.
        self.at_target = dict(self.target)

    def test_matches_gate_without_tone(self):
        plain = gate(self.at_target, self.baseline, self.target, [])
        extended = gate_extended(self.at_target, self.baseline, self.target, [])
        self.assertEqual(extended.decision, plain.decision)
        self.assertEqual(extended.revision_signals, plain.revision_signals)

    def test_tone_deviation_is_advisory_only(self):
        profile = ToneProfile(
            mean={m: 0.0 for m in TONE_METRICS},
            std={m: 0.0 for m in TONE_METRICS},
        )
        loud = {m: 99.0 for m in TONE_METRICS}
        result = gate_extended(
            self.at_target, self.baseline, self.target, [],
            tone_observed=loud, tone_profile=profile,
        )
        self.assertEqual(result.decision, "pass")
        self.assertTrue(any("reduce" in s for s in result.revision_signals))


class TestPipelineBackCompat(unittest.TestCase):
    def test_default_args_keep_legacy_output_shape(self):
        report = run_pipeline(CORPUS, DRAFT_GOOD, BANNED)
        self.assertEqual(
            set(report["classification"]),
            {"channel", "audience", "situation"},
        )
        self.assertEqual(
            report["target_profile"],
            calibrate(load_corpus(CORPUS), "email", "peer", "standard"),
        )

    def test_extended_args_engage_extended_stack(self):
        report = run_pipeline(
            CORPUS, DRAFT_GOOD, BANNED, goal="persuade", stakes="high",
        )
        self.assertEqual(report["classification"]["goal"], "persuade")
        self.assertEqual(report["classification"]["stakes"], "high")
        baseline = load_corpus(CORPUS)
        expected, _ = calibrate_extended(
            baseline, VoiceContext(goal="persuade", stakes="high"),
        )
        self.assertEqual(report["target_profile"], expected)

    def test_extended_args_validate(self):
        with self.assertRaises(ValueError):
            run_pipeline(CORPUS, DRAFT_GOOD, BANNED, goal="nope")


if __name__ == "__main__":
    unittest.main()
