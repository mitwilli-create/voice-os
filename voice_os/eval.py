"""Evaluation scorecard: measures the model against held-out real text.

Usage:
    python -m voice_os.eval [--corpus-dir corpus] [--json] [--save PATH]
    python -m voice_os.eval label --sample 30

Miners train on the held-in split; everything here measures on the
held-out split, so improvement claims are grounded in text the model
never saw. Metrics:

1. Context fidelity: how close each held-out chunk's real text scores to
   (a) the baseline-only profile, (b) the hand-calibrated target, and
   (c) the extended mined target for the chunk's own context. The
   extended model wins when (c) > (b) > (a).
2. Tone calibration error: mean absolute error of tone norms (mined per
   context vs the global norm) against observed tone metrics.
3. Goal label accuracy: stored heuristic goal tags against hand labels
   in corpus/labels/goals.jsonl (gitignored), when labels exist.
4. Banned-list efficacy: false-positive rate of the mined n-gram list on
   held-out self text (target near zero) and recall on the contrast
   corpus.
5. Drift: flags and suggestions from the drift report, printed verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .axes import AXES, AxisProfile, score_text
from .calibration import calibrate, calibrate_extended
from .contexts import GOALS, MEDIA, VoiceContext
from .corpus import TIER_WEIGHTS
from .holdout import is_holdout
from .mined import MinedArtifacts, group_profile, load_artifacts
from .qa import find_banned
from .store import iter_chunks
from .tone import TONE_METRICS, ToneProfile, derive_metrics

DEFAULT_CORPUS_DIR = "corpus"
LABELS_FILE = os.path.join("labels", "goals.jsonl")
# The committed seed resolves relative to this file (not the working
# directory); the generated set lives under the corpus dir given at run
# time, so a non-default corpus root reads its own contrast data.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_CONTRAST_PATH = os.path.join(_REPO_ROOT, "data", "contrast", "synthetic_llm.txt")


def _holdout_generation_chunk(chunk: dict, holdout_pct: int) -> bool:
    """True for a well-formed held-out chunk with generation weight.

    Malformed records (missing text, short or non-hex hash, non-numeric
    tier) are skipped, matching the store's graceful-degradation contract.
    """
    if not isinstance(chunk.get("text"), str) or not chunk["text"]:
        return False
    try:
        if not is_holdout(chunk.get("hash", ""), holdout_pct):
            return False
        tier = int(chunk.get("tier", 4))
    except (TypeError, ValueError):
        return False
    return TIER_WEIGHTS.get(tier, 0.0) > 0


def _profile_tone(profile: dict | None) -> ToneProfile | None:
    """ToneProfile from a mined group profile; None when the shape is off."""
    if not isinstance(profile, dict):
        return None
    mean, std = profile.get("tone_mean"), profile.get("tone_std")
    if isinstance(mean, dict) and isinstance(std, dict) and mean:
        return ToneProfile(mean=mean, std=std)
    return None


def _chunk_context(chunk: dict) -> VoiceContext | None:
    context = chunk.get("context", {})
    goal = context.get("goal", "unknown")
    medium = context.get("medium")
    try:
        ctx = VoiceContext(
            channel=context.get("channel", "email"),
            audience=context.get("audience", "peer"),
            goal=goal if goal in GOALS else "unknown",
            medium=medium if medium in MEDIA else None,
        )
        ctx.validate()
    except ValueError:
        return None
    return ctx


def _tone_lookup(mined: MinedArtifacts, ctx: VoiceContext) -> ToneProfile | None:
    profiles = mined.context_profiles
    if not profiles:
        return None
    lookups = []
    if ctx.medium:
        lookups.append(("pairs", f"{ctx.audience}|{ctx.medium}"))
    lookups.append(("audiences", ctx.audience))
    if ctx.medium:
        lookups.append(("media", ctx.medium))
    if ctx.goal != "unknown":
        lookups.append(("goals", ctx.goal))
    for kind, key in lookups:
        tone = _profile_tone(group_profile(profiles, kind, key))
        if tone:
            return tone
    return None


def _load_goal_labels(corpus_dir: str) -> dict[str, str]:
    path = os.path.join(corpus_dir, LABELS_FILE)
    labels: dict[str, str] = {}
    if not os.path.exists(path):
        return labels
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and entry.get("id") and entry.get("goal"):
                labels[entry["id"]] = entry["goal"]
    return labels


def _load_contrast_passages(corpus_dir: str) -> list[str]:
    # Local, minimal reader: eval must not depend on the mine package.
    passages: list[str] = []
    paths = (
        SEED_CONTRAST_PATH,
        os.path.join(corpus_dir, "contrast", "generated.jsonl"),
    )
    for path in paths:
        if not os.path.exists(path):
            continue
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            text = json.loads(line).get("text", "")
                        except json.JSONDecodeError:
                            continue
                        if text.strip():
                            passages.append(text.strip())
        else:
            with open(path, encoding="utf-8") as f:
                block: list[str] = []
                for line in f:
                    if line.strip():
                        if not line.lstrip().startswith("#"):
                            block.append(line.strip())
                    elif block:
                        passages.append(" ".join(block))
                        block = []
                if block:
                    passages.append(" ".join(block))
    return passages


def evaluate(
    corpus_path: str,
    chunks_dir: str,
    mined_dir: str | None,
    holdout_pct: int = 20,
) -> dict:
    """Compute the scorecard. JSON-safe dict."""
    from . import load_corpus

    baseline = load_corpus(corpus_path)
    mined = load_artifacts(mined_dir)
    global_tone = _profile_tone((mined.context_profiles or {}).get("global"))

    fidelity_sums = {"baseline_only": 0.0, "hand_calibrated": 0.0, "extended_mined": 0.0}
    breakouts: dict[str, dict[str, list[float]]] = {
        "audience": {}, "medium": {}, "goal": {},
    }
    tone_abs_err = {"mined": {m: 0.0 for m in TONE_METRICS},
                    "global": {m: 0.0 for m in TONE_METRICS}}
    tone_counts = {"mined": 0, "global": 0}
    goal_labels = _load_goal_labels(os.path.dirname(chunks_dir) or DEFAULT_CORPUS_DIR)
    goal_hits = 0
    goal_seen = 0
    banned_fp = 0
    n = 0

    target_cache: dict[tuple, tuple[dict, dict]] = {}

    for chunk in iter_chunks(chunks_dir):
        if not _holdout_generation_chunk(chunk, holdout_pct):
            continue
        ctx = _chunk_context(chunk)
        if ctx is None:
            continue
        n += 1
        scores = score_text(chunk["text"])

        key = (ctx.channel, ctx.audience, ctx.goal, ctx.medium)
        if key not in target_cache:
            hand = calibrate(baseline, ctx.channel, ctx.audience, "standard")
            extended, _ = calibrate_extended(baseline, ctx, mined=mined)
            target_cache[key] = (hand, extended)
        hand_target, extended_target = target_cache[key]

        fidelity_sums["baseline_only"] += baseline.fidelity(scores)[0]
        hand_profile = AxisProfile(mean=hand_target, std=baseline.std)
        fidelity_sums["hand_calibrated"] += hand_profile.fidelity(scores)[0]
        extended_profile = AxisProfile(mean=extended_target, std=baseline.std)
        extended_fidelity = extended_profile.fidelity(scores)[0]
        fidelity_sums["extended_mined"] += extended_fidelity

        for label, value in (
            ("audience", ctx.audience),
            ("medium", ctx.medium or "none"),
            ("goal", ctx.goal),
        ):
            breakouts[label].setdefault(value, []).append(extended_fidelity)

        observed = derive_metrics(chunk.get("context", {}).get("tone_signals") or {})
        for name, profile in (("mined", _tone_lookup(mined, ctx)), ("global", global_tone)):
            if profile is None:
                continue
            tone_counts[name] += 1
            for metric in TONE_METRICS:
                predicted = profile.mean.get(metric, 0.0)
                tone_abs_err[name][metric] += abs(observed.get(metric, 0.0) - predicted)

        if mined.ngram_banned and find_banned(chunk["text"], mined.ngram_banned):
            banned_fp += 1

        label = goal_labels.get(chunk.get("id", ""))
        if label:
            goal_seen += 1
            if label == chunk.get("context", {}).get("goal"):
                goal_hits += 1

    scorecard: dict = {
        "holdout": {"chunks": n, "pct": holdout_pct},
        "fidelity": {
            variant: round(total / n, 4) if n else None
            for variant, total in fidelity_sums.items()
        },
        "fidelity_breakouts": {
            label: {
                value: {"n": len(vals), "mean": round(sum(vals) / len(vals), 4)}
                for value, vals in sorted(groups.items())
            }
            for label, groups in breakouts.items()
        },
        "tone_mae": {
            name: (
                {
                    **{m: round(err / tone_counts[name], 4) for m, err in errs.items()},
                    "chunks": tone_counts[name],
                }
                if tone_counts[name]
                else None
            )
            for name, errs in tone_abs_err.items()
        },
        "goal_labels": {"labeled": goal_seen,
                        "accuracy": round(goal_hits / goal_seen, 4) if goal_seen else None},
        "banned": {
            "mined_phrases": len(mined.ngram_banned),
            "false_positive_rate": round(banned_fp / n, 4) if n else None,
        },
        "drift": {
            "flags": (mined.drift_report or {}).get("flags", []),
            "suggestions": (mined.drift_report or {}).get("suggestions", []),
        },
    }

    if mined.ngram_banned:
        contrast = _load_contrast_passages(os.path.dirname(chunks_dir) or DEFAULT_CORPUS_DIR)
        if contrast:
            hits = sum(
                1 for text in contrast if find_banned(text, mined.ngram_banned)
            )
            scorecard["banned"]["contrast_recall"] = round(hits / len(contrast), 4)
            scorecard["banned"]["contrast_passages"] = len(contrast)
    return scorecard


def render(scorecard: dict) -> str:
    lines = ["Voice OS evaluation scorecard", "=" * 34]
    holdout = scorecard["holdout"]
    lines.append(f"held-out chunks: {holdout['chunks']} ({holdout['pct']}%)")
    lines.append("")
    lines.append("context fidelity of real held-out text (higher is better):")
    for variant in ("baseline_only", "hand_calibrated", "extended_mined"):
        value = scorecard["fidelity"][variant]
        lines.append(f"  {variant:16} {value if value is not None else 'n/a'}")
    lines.append("")
    for label, groups in scorecard["fidelity_breakouts"].items():
        if not groups:
            continue
        lines.append(f"extended fidelity by {label}:")
        for value, stats in groups.items():
            lines.append(f"  {value:20} {stats['mean']}  (n={stats['n']})")
    lines.append("")
    lines.append("tone mean absolute error (lower is better):")
    for name in ("mined", "global"):
        stats = scorecard["tone_mae"][name]
        if stats is None:
            lines.append(f"  {name}: n/a")
            continue
        metrics = ", ".join(
            f"{m}={stats[m]}" for m in TONE_METRICS if m in stats
        )
        lines.append(f"  {name} ({stats['chunks']} chunks): {metrics}")
    lines.append("")
    goal = scorecard["goal_labels"]
    if goal["labeled"]:
        lines.append(f"goal label accuracy: {goal['accuracy']} over {goal['labeled']} labels")
    else:
        lines.append("goal label accuracy: no hand labels yet "
                     "(python -m voice_os.eval label --sample 30)")
    banned = scorecard["banned"]
    lines.append(
        f"mined banned list: {banned['mined_phrases']} phrases, "
        f"holdout false-positive rate {banned['false_positive_rate']}"
        + (
            f", contrast recall {banned['contrast_recall']} "
            f"over {banned['contrast_passages']} passages"
            if "contrast_recall" in banned
            else ""
        )
    )
    drift = scorecard["drift"]
    if drift["flags"] or drift["suggestions"]:
        lines.append("")
        lines.append("drift report:")
        for flag in drift["flags"]:
            lines.append(
                f"  {flag['window']} {flag['axis']}: {flag['from']} -> {flag['to']}"
            )
        for suggestion in drift["suggestions"]:
            lines.append(f"  suggestion: {suggestion}")
    return "\n".join(lines)


def sample_for_labeling(chunks_dir: str, n: int = 30, holdout_pct: int = 20) -> list[dict]:
    """Deterministic held-out sample for hand-labeling goals.

    Sorted by content hash so the same store always yields the same
    sample; append labels to corpus/labels/goals.jsonl as
    {"id": ..., "goal": ...} lines.
    """
    held_out = [
        chunk
        for chunk in iter_chunks(chunks_dir)
        if _holdout_generation_chunk(chunk, holdout_pct)
    ]
    held_out.sort(key=lambda c: c.get("hash", ""))
    return held_out[:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice_os.eval", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="compute and print the scorecard (default)")
    label_p = sub.add_parser("label", help="print a held-out sample for hand-labeling")
    label_p.add_argument("--sample", type=int, default=30)

    for p in (parser, run_p):
        p.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
        p.add_argument("--json", action="store_true", dest="as_json")
        p.add_argument("--save", default=None, help="also write the JSON scorecard here")
    label_p.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)

    args = parser.parse_args(argv)
    chunks_dir = os.path.join(args.corpus_dir, "chunks")

    if args.command == "label":
        for chunk in sample_for_labeling(chunks_dir, args.sample):
            context = chunk.get("context", {})
            print(json.dumps({
                "id": chunk.get("id"),
                "heuristic_goal": context.get("goal"),
                "channel": context.get("channel"),
                "text": chunk.get("text", "")[:280],
            }))
        print(
            f"\nlabel these into {os.path.join(args.corpus_dir, LABELS_FILE)} as "
            '{"id": ..., "goal": ...} lines',
            file=sys.stderr,
        )
        return 0

    scorecard = evaluate(
        corpus_path=os.path.join(args.corpus_dir, "voice_corpus.txt"),
        chunks_dir=chunks_dir,
        mined_dir=os.path.join(args.corpus_dir, "mined"),
    )
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=1)
    print(json.dumps(scorecard, indent=1) if args.as_json else render(scorecard))
    return 0


if __name__ == "__main__":
    sys.exit(main())
