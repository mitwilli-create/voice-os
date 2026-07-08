"""Pattern-profile fusion: mined Tier 1 pattern statistics distilled
into bounded, prompt-ready guidance strings for the prepare node.

Phase 0 item 1 of docs/voice-reflection-engine.md: the evolution
module's Tier 1 pattern profile (greeting and sign-off distributions,
casual-marker rates, sentence shape) previously existed only for drift
detection. distill_pattern_guidance() renders it as fixed-template
plain-English statements, exactly as distill_kb_guidance() renders the
compact KB (docs/kb-fusion.md).

Privacy invariant, mirroring evolution's fixed-lexicon rule: template
slots iterate the FIXED evolution lexicons (GREETING_LEXICON,
SIGNOFF_LEXICON, DEFAULT_MARKERS), so free vocabulary, including the
"other" bucket, can never render, even from a hand-edited artifact.
Everything else in a rendered line is a template literal or a number.

The distiller never invents content: a missing, mistyped, thin, or
non-finite profile yields fewer lines or an empty list, never an error.
Pure and deterministic for a given profile dict.

Stdlib only. Design: docs/voice-reflection-engine.md section 3 item 1,
implementation record docs/pattern-fusion.md.
"""

from __future__ import annotations

import math

from ..evolution.patterns import (
    DEFAULT_MARKERS,
    GREETING_LEXICON,
    MIN_SUPPORT,
    NEAR_ZERO,
    SHORT_CHUNK_WORDS,
    SIGNOFF_LEXICON,
)

# Bounds, same idea as the KB-fusion caps: the template set can only
# produce PATTERN_GUIDANCE_MAX_ITEMS single-line statements, and the
# word budget counts exactly the text that renders, so the persona
# prompt stays bounded whatever the artifact contains.
PATTERN_GUIDANCE_MAX_ITEMS = 6
PATTERN_GUIDANCE_MAX_WORDS = 120
# Support floors: a profile mined from fewer chunks than this is too
# thin to state percentages as the author's habits (the real Tier 1
# profile is corpus-scale); per-form floors reuse the evolution diff
# constants so fusion and drift agree on what "present" means.
PATTERN_MIN_CHUNKS = 50
PATTERN_TOP_FORMS = 3


def _rate(value) -> float | None:
    """A non-negative finite float, or None. Bools are ints and must
    not become rates; NaN/Infinity survive json.load and arithmetic."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _count(value) -> int:
    """A non-negative int, or 0. Same bool caveat as _rate."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _top_forms(
    profile: dict,
    value_field: str,
    count_field: str,
    lexicon: tuple[str, ...],
) -> list[tuple[float, str]]:
    """The strongest lexicon forms as (rate, form), rate descending then
    form name. Iterates the lexicon, never the artifact's keys: the
    lexicon is the whitelist, so unlisted keys (including "other") can
    never surface. A form needs both the MIN_SUPPORT raw count and a
    rate above NEAR_ZERO, matching the diff layer's presence floors."""
    values = profile.get(value_field)
    counts = profile.get(count_field)
    if not isinstance(values, dict) or not isinstance(counts, dict):
        return []
    picked = []
    for form in lexicon:
        rate = _rate(values.get(form))
        if rate is None or rate <= NEAR_ZERO:
            continue
        if _count(counts.get(form)) < MIN_SUPPORT:
            continue
        picked.append((rate, form))
    picked.sort(key=lambda pair: (-pair[0], pair[1]))
    return picked[:PATTERN_TOP_FORMS]


def distill_pattern_guidance(
    profile: dict | None, max_words: int = PATTERN_GUIDANCE_MAX_WORDS
) -> list[str]:
    """Bounded, prompt-ready voice-pattern statements from a mined Tier 1
    pattern profile (evolution/patterns.py::extract_pattern_profile,
    persisted in the evolution_flags artifact).

    Returns [] when the profile is absent, malformed, or mined from
    fewer than PATTERN_MIN_CHUNKS chunks. Every line is single-line by
    construction (template literals, lexicon slots, numbers); the
    result is capped at max_words total and PATTERN_GUIDANCE_MAX_ITEMS
    lines.
    """
    if not isinstance(profile, dict):
        return []
    if _count(profile.get("n_chunks")) < PATTERN_MIN_CHUNKS:
        return []

    candidates: list[str] = []

    greetings = _top_forms(
        profile, "greetings", "greeting_counts", GREETING_LEXICON
    )
    if greetings:
        candidates.append(
            "Greetings the author actually opens with, most common first: "
            + "; ".join(
                f"'{form}' (in {round(rate * 100)} percent of messages)"
                for rate, form in greetings
            )
        )

    signoffs = _top_forms(
        profile, "signoffs", "signoff_counts", SIGNOFF_LEXICON
    )
    if signoffs:
        candidates.append(
            "Sign-offs the author actually closes with, most common first: "
            + "; ".join(
                f"'{form}' (in {round(rate * 100)} percent of messages)"
                for rate, form in signoffs
            )
        )

    markers = _top_forms(
        profile, "markers_per_100w", "marker_counts", DEFAULT_MARKERS
    )
    if markers:
        candidates.append(
            "Casual markers the author uses, rate per 100 words: "
            + "; ".join(
                f"'{form}' {round(rate, 2)}" for rate, form in markers
            )
        )

    sentence = profile.get("sentence_length")
    if (
        isinstance(sentence, dict)
        and _count(profile.get("n_sentences")) >= MIN_SUPPORT
    ):
        mean = _rate(sentence.get("mean"))
        p50 = _rate(sentence.get("p50"))
        p90 = _rate(sentence.get("p90"))
        if mean and p50 is not None and p90 is not None:
            candidates.append(
                f"Sentences run about {round(mean, 1)} words on average "
                f"(median {round(p50)}, 90th percentile {round(p90)})."
            )
        short_rate = _rate(sentence.get("short_rate"))
        if short_rate is not None and short_rate > NEAR_ZERO:
            candidates.append(
                f"About {round(short_rate * 100)} percent of messages run "
                f"under {SHORT_CHUNK_WORDS} words."
            )

    exclamations = _rate(profile.get("exclamations_per_100w"))
    if exclamations is not None and exclamations > NEAR_ZERO:
        candidates.append(
            f"Exclamation marks appear about {round(exclamations, 1)} "
            "times per 100 words."
        )

    guidance: list[str] = []
    total_words = 0
    for candidate in candidates[:PATTERN_GUIDANCE_MAX_ITEMS]:
        words = len(candidate.split())
        if total_words + words > max_words:
            break
        guidance.append(candidate)
        total_words += words
    return guidance
