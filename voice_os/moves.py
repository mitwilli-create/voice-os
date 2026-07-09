"""Signature-move detection and the per-batch avoid-list vocabulary.

The 2026-07-08 field report (class 6) documented corpus-level tic
convergence: across 17 independent runs the same signature moves
recurred (fragment date openers, "X. Not Y." constructions, staccato
"No X. No Y." runs, single-fragment and punch-tag closers) until one
page of stories read as a detectable house tic. Each move is fine
alone; a single call cannot see the batch.

This module gives batch callers the two halves of the fix:

- `detect(text)` always runs in the product graph and reports which
  moves an output uses (envelope `signature_moves.detected`), so a
  batch caller can measure convergence across its items;
- `draft(avoid=[...])` opts specific moves out: the generative persona
  is told to avoid them, and a detected avoided move blocks a pass the
  same way a redraft entailment failure does.

Everything is stdlib-only and deterministic. Detectors are calibrated
against the site pass's voiced outputs: the moves the report names in
its receipts (mong-kok's "October 2014." opener, stream-launch's "On
purpose." punch tag, "No satellite truck. No control room. No studio.")
are all detected.
"""

from __future__ import annotations

import re

from .conservation import split_sentences

# One entry per move: caller-facing key -> (guidance for the persona,
# shown when the caller avoids the move).
CATALOG: dict[str, str] = {
    "fragment-date-opener": (
        "do not open with a bare date fragment like 'December 2007.'; "
        "weave the date into a full opening sentence"
    ),
    "x-not-y": (
        "do not use the 'X. Not Y.' construction (a sentence followed by "
        "a short corrective fragment starting with 'Not')"
    ),
    "no-x-run": (
        "do not stack staccato 'No X. No Y.' fragment runs; fold the "
        "negations into one sentence"
    ),
    "punch-tag-closer": (
        "do not close with a punch tag ('That's the whole point.' / "
        "'Every time.' / 'On purpose.'); end on a full sentence"
    ),
    "fragment-closer": (
        "do not close with a short sentence fragment; end on a complete "
        "sentence"
    ),
}

_MONTHS = (
    "january february march april may june july august september "
    "october november december".split()
)

# Fixed emphatic tags the report saw recurring across independent runs.
_PUNCH_TAGS = frozenset(
    (
        "that's the point",
        "that's the whole point",
        "that's the whole game",
        "that's the game",
        "every time",
        "on purpose",
        "not once",
        "full stop",
    )
)

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NOT_FRAGMENT = re.compile(r"(?<=[.!?])\s+(Not\s+[^.!?\n]{1,40}[.!?])")
# Case-sensitive: the move is consecutive capital-"No" fragments; a
# mid-sentence lowercase "no" must not anchor a run.
_NO_RUN = re.compile(r"\bNo\s+[^.!?\n]{1,30}\.\s+No\s+[^.!?\n]{1,30}\.")


def _normalize_tag(sentence: str) -> str:
    return re.sub(r"[^a-z' ]", "", sentence.lower()).strip()


def _is_date_fragment(sentence: str) -> bool:
    words = sentence.split()
    if not words or len(words) > 4 or not sentence.endswith("."):
        return False
    if not _YEAR.search(sentence):
        return False
    # "December 2007." / "October 2014." / "May 2, 2011." style: every
    # word is a month, a number, or a short ordinal; no verbs to be had.
    for word in words:
        bare = word.strip(".,").lower()
        if bare in _MONTHS or re.fullmatch(r"\d{1,4}(st|nd|rd|th)?", bare):
            continue
        return False
    return True


def detect(text: str) -> dict[str, list[str]]:
    """Signature moves present in the text: move key -> instances.

    Keys with no instances are omitted, so an empty dict means a
    move-free text.
    """
    found: dict[str, list[str]] = {}
    sentences = split_sentences(text)
    if not sentences:
        return found

    if _is_date_fragment(sentences[0]):
        found["fragment-date-opener"] = [sentences[0]]

    not_fragments = _NOT_FRAGMENT.findall(text)
    if not_fragments:
        found["x-not-y"] = not_fragments

    no_runs = _NO_RUN.findall(text)
    if no_runs:
        found["no-x-run"] = no_runs

    closer = sentences[-1]
    if _normalize_tag(closer) in _PUNCH_TAGS:
        found["punch-tag-closer"] = [closer]
    elif len(closer.split()) <= 4 and closer.endswith(".") and \
            '"' not in closer:
        found["fragment-closer"] = [closer]

    return found


def validate_avoid(avoid: list[str]) -> list[str]:
    """Canonicalized avoid list; raises on unknown move keys."""
    canonical = []
    for key in avoid:
        slug = str(key).strip().lower().replace("_", "-")
        if slug not in CATALOG:
            raise ValueError(
                f"unknown signature move {key!r}; known moves: "
                + ", ".join(sorted(CATALOG))
            )
        if slug not in canonical:
            canonical.append(slug)
    return canonical


def avoid_guidance(avoid: list[str]) -> list[str]:
    """Persona guidance lines for the avoided moves."""
    return [f"avoid this signature move: {CATALOG[key]}" for key in avoid]
