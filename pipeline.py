#!/usr/bin/env python3
"""Full dual-persona pipeline with QA gate.

Usage:
    python pipeline.py \
        --corpus data/sample_corpus.txt \
        --banned-list data/banned_list.txt \
        --draft data/sample_draft.txt \
        --output output/scored_draft.json

Optional context flags: --channel, --audience, --situation (see
voice_os/calibration.py for accepted values). Runs deterministically
offline; uses Claude for the personas when API credentials resolve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from voice_os import run_pipeline
from voice_os.calibration import AUDIENCES, CHANNELS, SITUATIONS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--banned-list", default=None)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", default=None, help="write result JSON here")
    parser.add_argument("--channel", default="email", choices=CHANNELS)
    parser.add_argument("--audience", default="peer", choices=AUDIENCES)
    parser.add_argument("--situation", default="standard", choices=SITUATIONS)
    parser.add_argument("--max-cycles", type=int, default=2)
    args = parser.parse_args()

    with open(args.draft, encoding="utf-8") as f:
        draft = f.read()

    result = run_pipeline(
        corpus_path=args.corpus,
        draft_text=draft,
        banned_path=args.banned_list,
        channel=args.channel,
        audience=args.audience,
        situation=args.situation,
        max_cycles=args.max_cycles,
    )

    payload = json.dumps(result, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"wrote {args.output}")

    final = result["final"]
    print(f"mode:      {result['meta']['mode']}")
    print(f"decision:  {final['decision']}")
    print(f"fidelity:  {final['fidelity']:.2f}")
    print(f"cycles:    {len(result['cycles'])}")
    print("--- output ---")
    print(final["output_text"])
    return 0 if final["decision"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
