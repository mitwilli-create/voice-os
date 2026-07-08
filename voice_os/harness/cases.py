"""Held-out case selection and deterministic content briefs.

A case is one real Tier 1 held-out message plus the style-neutralized
brief the pipeline will draft in the same context. Selection is
stratified by (channel, audience), sorted by content hash, RNG-free,
and independent of file order, so the same store and parameters always
yield the same case list (docs/determinism.md, Rules for new modules).

The brief is ALWAYS built by this deterministic neutralizer, in both
offline and live runs, so pipeline inputs never vary by mode. Greeting
and sign-off stripping matches the FIXED lexicons from the evolution
module: nothing name-shaped is matched, so briefs add no new privacy
surface beyond the chunk text they derive from.

Design: docs/eval-harness.md.
"""

from __future__ import annotations

import re

from ..contexts import GOALS, MEDIA, VoiceContext
from ..evolution.patterns import GREETING_LEXICON, SIGNOFF_LEXICON
from ..holdout import is_holdout
from ..store import iter_chunks

MIN_WORDS = 12
MAX_WORDS = 400
DEFAULT_PER_CELL = 6
DEFAULT_CAP = 72
HOLDOUT_PCT = 20  # the repo-standard split; not a tunable

# Style tokens removed from briefs: the hedge/filler lexicon mirrored
# from voice_os/axes.py plus exclamation and emoji stripping below.
# Deliberately duplicated rather than imported from the private axes
# regex, so a scoring-side lexicon change cannot silently change what
# the pipeline receives as input.
_FILLERS = re.compile(
    r"\b(maybe|perhaps|possibly|might|i think|i guess|i feel like|sort of|"
    r"kind of|just|a bit|somewhat|arguably|it seems|we may want to|"
    r"haha|lol)\b",
    re.IGNORECASE,
)
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\ufe0f"
    "]+"
)
_MAX_GREETING_WORDS = 6
_MAX_SIGNOFF_WORDS = 5


def _lexicon_line(line: str, lexicon: tuple[str, ...], max_words: int) -> bool:
    """True when a short line starts with a fixed lexicon entry."""
    stripped = line.strip().lower()
    if not stripped or len(stripped.split()) > max_words:
        return False
    normalized = stripped.strip(" \t,.!;:-")
    return any(
        normalized == entry or normalized.startswith(entry + " ")
        for entry in lexicon
    )


def build_brief(text: str) -> str:
    """Deterministic style-neutralized content brief of a real message.

    Partial by design: lexicon neutralization leaves distinctive
    vocabulary in place. The residual bias is analyzed in
    docs/eval-harness.md; the paired style comparison is the
    discriminating metric, not raw similarity.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and _lexicon_line(lines[0], GREETING_LEXICON, _MAX_GREETING_WORDS):
        lines = lines[1:]
    if lines and _lexicon_line(lines[-1], SIGNOFF_LEXICON, _MAX_SIGNOFF_WORDS):
        lines = lines[:-1]
    joined = " ".join(lines)
    joined = joined.replace("!", ".")
    joined = _EMOJI.sub(" ", joined)
    joined = _FILLERS.sub(" ", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    if not joined:
        # Everything was style (e.g. a bare greeting): fall back to the
        # whitespace-collapsed original so the case still runs.
        joined = re.sub(r"\s+", " ", text).strip()
    return joined


def _case_context(chunk: dict) -> VoiceContext | None:
    """Tolerant context parse, mirroring the scorecard's semantics."""
    context = chunk.get("context", {})
    if not isinstance(context, dict):
        return None
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


def _eligible(chunk: dict) -> bool:
    """Tier 1, held out, well formed, and inside the word-count band."""
    text = chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        if int(chunk.get("tier", 4)) != 1:
            return False
        if not is_holdout(chunk.get("hash", ""), HOLDOUT_PCT):
            return False
    except (TypeError, ValueError):
        return False
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS


def select_cases(
    chunks_dir: str,
    per_cell: int = DEFAULT_PER_CELL,
    cap: int = DEFAULT_CAP,
) -> list[dict]:
    """Deterministic stratified case list from the chunk store.

    Candidates group by (channel, audience); each cell sorts by content
    hash and keeps the first per_cell. Cells are then walked in sorted
    order and the total cap trims the tail, never reordering within a
    cell.
    """
    if per_cell < 1:
        raise ValueError(f"per_cell must be >= 1, got {per_cell}")
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")

    cells: dict[tuple[str, str], list[tuple[str, dict, VoiceContext]]] = {}
    seen_hashes: set[str] = set()
    for chunk in iter_chunks(chunks_dir):
        if not _eligible(chunk):
            continue
        ctx = _case_context(chunk)
        if ctx is None:
            continue
        chunk_hash = chunk["hash"]
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)
        cells.setdefault((ctx.channel, ctx.audience), []).append(
            (chunk_hash, chunk, ctx)
        )

    cases: list[dict] = []
    for cell_key in sorted(cells):
        ranked = sorted(cells[cell_key], key=lambda item: item[0])
        for chunk_hash, chunk, ctx in ranked[:per_cell]:
            text = chunk["text"]
            cases.append(
                {
                    "id": chunk.get("id") or chunk_hash[:16],
                    "hash": chunk_hash,
                    "channel": ctx.channel,
                    "audience": ctx.audience,
                    "medium": ctx.medium,
                    "goal": ctx.goal,
                    "real_text": text,
                    "brief": build_brief(text),
                }
            )
            if len(cases) >= cap:
                return cases
    return cases
