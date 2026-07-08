"""Mining CLI: python -m mine run|status.

Artifacts are written to the gitignored corpus/mined/ directory. Each job
reads the chunk store, trains on the held-in split only, and writes one
validated JSON artifact the voice_os runtime can load offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from ingest.export import iter_chunk_dicts

from .contrast import DEFAULT_GENERATED, DEFAULT_SEED, generate_contrast, load_contrast
from .drift import mine_drift
from .gate_calibration import mine_gate_calibration
from .ngrams import load_never_ban, mine_ngram_diffs
from .recipients import mine_recipient_deltas
from .tone_norms import mine_context_profiles

NEVER_BAN_PATH = os.path.join("data", "never_ban.txt")

# job name -> (miner callable over a chunk iterator, output filename);
# the ngrams and gate jobs need extra inputs and are dispatched specially.
JOBS = {
    "recipients": (mine_recipient_deltas, "recipient_deltas.json"),
    "tone": (mine_context_profiles, "context_profiles.json"),
    "ngrams": (None, "ngram_banned.json"),
    "drift": (mine_drift, "drift_report.json"),
    "gate": (None, "gate_calibration.json"),
}


def _mine_ngrams(args: argparse.Namespace) -> dict:
    contrast_paths = args.contrast or [DEFAULT_SEED, DEFAULT_GENERATED]
    contrast = load_contrast(contrast_paths)
    if not contrast:
        raise ValueError(f"no contrast passages found in {', '.join(contrast_paths)}")
    never_ban = (
        load_never_ban(NEVER_BAN_PATH) if os.path.exists(NEVER_BAN_PATH) else set()
    )
    return mine_ngram_diffs(
        iter_chunk_dicts(args.corpus_dir), contrast, never_ban=never_ban
    )


def _run(args: argparse.Namespace) -> int:
    requested = [j.strip() for j in args.job.split(",") if j.strip()]
    if "all" in requested:
        requested = list(JOBS)
    unknown = [j for j in requested if j not in JOBS]
    if unknown:
        print(f"unknown job(s): {', '.join(unknown)}; valid: {', '.join(JOBS)}, all",
              file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    for job in requested:
        miner, filename = JOBS[job]
        if job == "ngrams":
            artifact = _mine_ngrams(args)
        elif job == "gate":
            # Needs the corpus baseline for its calibrated targets; a
            # chunks-only corpus dir skips the job instead of failing the
            # whole run.
            corpus_path = os.path.join(args.corpus_dir, "voice_corpus.txt")
            if not os.path.exists(corpus_path):
                print(f"gate: skipped (no corpus file at {corpus_path})")
                continue
            # Reads the other mined artifacts for its targets, so it runs
            # meaningfully after them (dict order in JOBS).
            artifact = mine_gate_calibration(
                iter_chunk_dicts(args.corpus_dir),
                corpus_path=corpus_path,
                mined_dir=args.out,
            )
        else:
            artifact = miner(iter_chunk_dicts(args.corpus_dir))
        out_path = os.path.join(args.out, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=1)
        groups = {
            k: len(v)
            for k, v in artifact["data"].items()
            if isinstance(v, (dict, list)) and k != "global"
        }
        print(f"{job}: wrote {out_path} ({groups})")
    return 0


def _contrast_gen(args: argparse.Namespace) -> int:
    written = generate_contrast(args.n, args.out)
    return 0 if written else 1


def _status(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.out):
        print(f"no mined artifacts at {args.out}")
        return 0
    for _, filename in JOBS.values():
        path = os.path.join(args.out, filename)
        if not os.path.exists(path):
            print(f"{filename}: absent")
            continue
        with open(path, encoding="utf-8") as f:
            artifact = json.load(f)
        groups = {
            k: len(v)
            for k, v in artifact.get("data", {}).items()
            if isinstance(v, dict) and k != "global"
        }
        print(f"{filename}: {artifact.get('artifact')} v{artifact.get('version')} "
              f"generated {artifact.get('generated_at')} {groups}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run mining jobs")
    run_p.add_argument("--job", default="all", help="comma-separated: "
                       f"{', '.join(JOBS)}, or all (default)")
    run_p.add_argument("--corpus-dir", default="corpus")
    run_p.add_argument("--out", default=os.path.join("corpus", "mined"))
    run_p.add_argument("--contrast", nargs="*", default=None,
                       help="contrast corpus paths for the ngrams job "
                       f"(default: {DEFAULT_SEED} plus {DEFAULT_GENERATED} if present)")
    run_p.set_defaults(func=_run)

    gen_p = sub.add_parser(
        "contrast-gen",
        help="generate a contrast corpus via Claude (live API calls)",
    )
    gen_p.add_argument("--n", type=int, default=300)
    gen_p.add_argument("--out", default=DEFAULT_GENERATED)
    gen_p.set_defaults(func=_contrast_gen)

    status_p = sub.add_parser("status", help="show artifact ages and sizes")
    status_p.add_argument("--out", default=os.path.join("corpus", "mined"))
    status_p.set_defaults(func=_status)

    args = parser.parse_args(argv)
    return args.func(args)
