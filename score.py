#!/usr/bin/env python3
"""Score a draft against a voice corpus on the six canonical axes.

Usage:
    python score.py --corpus data/sample_corpus.txt --draft data/sample_draft.txt
"""

from __future__ import annotations

import argparse
import json
import sys

from voice_os import score_draft
from voice_os.axes import AXES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="voice corpus file")
    parser.add_argument("--draft", required=True, help="draft to score")
    parser.add_argument("--json", action="store_true", help="emit raw JSON only")
    args = parser.parse_args()

    with open(args.draft, encoding="utf-8") as f:
        draft = f.read()

    result = score_draft(args.corpus, draft)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"{'axis':<22}{'draft':>8}{'baseline':>10}{'closeness':>11}")
    for axis in AXES:
        print(
            f"{axis:<22}"
            f"{result['axis_scores'][axis]:>8.2f}"
            f"{result['baseline']['mean'][axis]:>10.2f}"
            f"{result['per_axis_fidelity'][axis]:>11.2f}"
        )
    print(f"\noverall fidelity: {result['fidelity']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
