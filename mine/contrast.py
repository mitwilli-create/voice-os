"""Contrast corpus loading and generation for n-gram mining.

Two pluggable sources, composable:
1. The committed synthetic seed at data/contrast/synthetic_llm.txt
   (blank-line-separated passages), which tests run against.
2. An optionally generated set at corpus/contrast/generated.jsonl,
   produced by `python -m mine contrast-gen` through voice_os.llm. That
   command spends API money, so it refuses to run under VOICE_OS_OFFLINE=1
   and prints the call count before proceeding. Output is gitignored.
"""

from __future__ import annotations

import json
import os

from voice_os.calibration import AUDIENCES, CHANNELS
from voice_os.contexts import GOALS

DEFAULT_SEED = os.path.join("data", "contrast", "synthetic_llm.txt")
DEFAULT_GENERATED = os.path.join("corpus", "contrast", "generated.jsonl")

_GEN_SYSTEM = (
    "You are a generic corporate email assistant. Write in the most "
    "conventional, template-like assistant style: polite openers, formal "
    "transitions, stock closings."
)


def load_contrast(paths: list[str]) -> list[str]:
    """Load contrast passages from .txt (blank-line-separated) and .jsonl
    ({\"text\": ...} per line) files. Missing paths are skipped so the
    generated set stays optional."""
    passages: list[str] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Skip malformed lines: a truncated generated file must
                    # not break mining while the seed corpus is present.
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = entry.get("text", "") if isinstance(entry, dict) else ""
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


def generate_contrast(n: int, out_path: str = DEFAULT_GENERATED) -> int:
    """Generate n contrast passages via Claude. Returns passages written.

    Explicitly refuses offline mode: this is the one mining path that
    costs API money (approved 2026-07-07), and it must never run as a
    silent side effect of an offline workflow.
    """
    if os.environ.get("VOICE_OS_OFFLINE") == "1":
        raise RuntimeError(
            "contrast-gen makes live API calls and VOICE_OS_OFFLINE=1 is set; "
            "unset it to generate the contrast corpus"
        )
    from voice_os import llm

    grid = [
        (channel, audience, goal)
        for channel in CHANNELS
        for audience in AUDIENCES
        for goal in GOALS
        if goal != "unknown"
    ]
    print(f"contrast-gen: {n} passages, {n} API calls (model {llm.DEFAULT_MODEL})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            channel, audience, goal = grid[i % len(grid)]
            prompt = (
                f"Write a short generic {channel} message to a {audience} "
                f"contact whose goal is to {goal}. 40 to 120 words. Use the "
                "most stereotypical assistant phrasing you can. Return only "
                "the message."
            )
            text = llm.complete(_GEN_SYSTEM, prompt, max_tokens=400)
            if not text:
                print(f"contrast-gen: API unavailable after {written} passages; stopping")
                break
            f.write(json.dumps({
                "text": text.strip(),
                "context": {"channel": channel, "audience": audience, "goal": goal},
            }) + "\n")
            written += 1
    print(f"contrast-gen: wrote {written} passages to {out_path}")
    return written
