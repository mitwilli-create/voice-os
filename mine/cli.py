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

from .recipients import mine_recipient_deltas
from .tone_norms import mine_context_profiles

# job name -> (miner callable over a chunk iterator, output filename)
JOBS = {
    "recipients": (mine_recipient_deltas, "recipient_deltas.json"),
    "tone": (mine_context_profiles, "context_profiles.json"),
}


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
        artifact = miner(iter_chunk_dicts(args.corpus_dir))
        out_path = os.path.join(args.out, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=1)
        groups = {
            k: len(v)
            for k, v in artifact["data"].items()
            if isinstance(v, dict) and k != "global"
        }
        print(f"{job}: wrote {out_path} ({groups})")
    return 0


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
    run_p.set_defaults(func=_run)

    status_p = sub.add_parser("status", help="show artifact ages and sizes")
    status_p.add_argument("--out", default=os.path.join("corpus", "mined"))
    status_p.set_defaults(func=_status)

    args = parser.parse_args(argv)
    return args.func(args)
