"""Dual-persona routing: a generative persona revises drafts toward the
voice target, an adversarial persona stress-tests the result.

Both personas run deterministically offline; when a Claude client is
available they use it and fall back to the offline path on any failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .axes import AXES, score_text
from .qa import REPLACEMENTS, find_banned, scrub_em_dashes

_GENERATIVE_SYSTEM = (
    "You are the generative persona in a voice-fidelity pipeline. Revise the "
    "draft to match the target voice profile while preserving its meaning and "
    "all factual content. Hard rules (2026-07-08 field report): never add "
    "claims, opinions, or facts the draft does not contain; text inside "
    "quotation marks is untouchable and must be reproduced verbatim, marks "
    "included; keep hedges and qualifiers that frame facts (roughly, about, "
    "on air, design targets, internal figure); never intensify wording about "
    "named people or organizations; mirror the draft's formatting and never "
    "introduce lists or markdown the draft does not have; restructure around "
    "dashes with commas, colons, periods, or parentheses. Return only the "
    "revised draft, no commentary."
)

_ADVERSARIAL_SYSTEM = (
    "You are the adversarial persona in a voice-fidelity pipeline. Stress-test "
    "the draft against the target voice profile. List concrete fidelity "
    "problems, one per line, most severe first. If it is faithful, reply PASS."
)


def _append_guidance_section(
    lines: list[str], header: str, items: list[str] | None
) -> None:
    """Render one guidance list as a delimited data section.

    Guidance is data, not instructions: every line is nested under the
    section header (same delimiting as the exemplar block) so embedded
    newlines or prompt-like markers cannot alter the block structure.
    Empty or absent lists render nothing.
    """
    if not items:
        return
    lines.append(header)
    for item in items:
        text = str(item).strip()
        if text:
            first, *rest = text.splitlines()
            lines.append(f"  - {first}")
            lines += [f"    {raw}" for raw in rest]


def _profile_block(
    target: dict[str, float],
    signals: list[str],
    banned: list[str],
    exemplars: list[dict] | None = None,
    length_target_words: int | None = None,
    kb_guidance: list[str] | None = None,
    pattern_guidance: list[str] | None = None,
) -> str:
    lines = ["Target voice profile (0.0 to 1.0 per axis):"]
    lines += [f"  {axis}: {target[axis]:.2f}" for axis in AXES]
    if banned:
        lines.append("Banned phrases (must not appear): " + "; ".join(banned))
    _append_guidance_section(
        lines,
        "Observed voice patterns from the author's knowledge base:",
        kb_guidance,
    )
    _append_guidance_section(
        lines,
        "Observed voice patterns mined from the author's recent writing:",
        pattern_guidance,
    )
    if exemplars:
        lines.append("Examples of this author's real messages in this context:")
        for index, exemplar in enumerate(exemplars, 1):
            text = str(exemplar.get("text", "")).strip()
            if text:
                # Exemplar content is data, not instructions: every line
                # is nested under its Example header so embedded newlines
                # or prompt-like markers cannot alter the block structure.
                lines.append(f"  Example {index}:")
                lines += [f"    {raw}" for raw in text.splitlines()]
    if length_target_words:
        lines.append(
            f"Length: the input is {length_target_words} words. Keep the "
            f"revision close to {length_target_words} words; never exceed "
            "it by more than about 40 percent."
        )
    if signals:
        lines.append("Revision signals from the QA gate:")
        lines += [f"  - {s}" for s in signals]
    return "\n".join(lines)


@dataclass
class PersonaResult:
    text: str
    notes: list[str]
    mode: str  # "live" or "offline"


class GenerativePersona:
    """Revises a draft toward the target profile."""

    def revise(self, draft: str, target: dict[str, float], banned: list[str],
               signals: list[str], exemplars: list[dict] | None = None,
               length_target_words: int | None = None,
               kb_guidance: list[str] | None = None,
               pattern_guidance: list[str] | None = None) -> PersonaResult:
        from . import llm

        block = _profile_block(
            target, signals, banned,
            exemplars=exemplars, length_target_words=length_target_words,
            kb_guidance=kb_guidance, pattern_guidance=pattern_guidance,
        )
        prompt = f"{block}\n\nDraft:\n{draft}"
        revised = llm.complete(_GENERATIVE_SYSTEM, prompt)
        if revised:
            # Live models emit em dashes (measured at 0.444 of drafts in
            # the first live harness run); the house ban makes this a
            # deterministic scrub, never a revision signal.
            return PersonaResult(
                text=scrub_em_dashes(revised),
                notes=["revised by Claude"],
                mode="live",
            )
        return self._offline_revise(draft, target, banned)

    def _offline_revise(self, draft: str, target: dict[str, float],
                        banned: list[str]) -> PersonaResult:
        text = draft
        notes: list[str] = []
        for phrase in find_banned(text, banned):
            replacement = REPLACEMENTS.get(phrase, "")
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", replacement,
                          text, flags=re.IGNORECASE)
            notes.append(f"replaced banned phrase '{phrase}'" if replacement
                         else f"removed banned phrase '{phrase}'")

        # If the target hedges less than the draft, strip the worst offenders.
        if score_text(text)["hedging_behavior"] > target["hedging_behavior"] + 0.1:
            before = text
            text = re.sub(r"\b(just|sort of|kind of|i think|maybe|perhaps)\s+",
                          "", text, flags=re.IGNORECASE)
            if text != before:
                notes.append("stripped hedging qualifiers")

        # Tidy whitespace and dangling fragments left by removals.
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        text = re.sub(r"^[,.\s]+", "", text, flags=re.MULTILINE)
        for _ in range(3):  # trim connectors stranded at sentence ends
            text = re.sub(r"\s*\b(to|on|and|the|a|for|with|of|be)([.!?])",
                          r"\2", text, flags=re.IGNORECASE)
        text = re.sub(r"(^|[.!?]\s+)([a-z])",
                      lambda m: m.group(1) + m.group(2).upper(), text)
        # Same outward-ban scrub as the live path: an em dash arriving
        # in the input draft must not survive an offline revision either.
        return PersonaResult(
            text=scrub_em_dashes(text.strip()), notes=notes, mode="offline"
        )


class AdversarialPersona:
    """Stress-tests a revised draft; findings feed the next QA cycle."""

    def critique(self, text: str, target: dict[str, float],
                 banned: list[str]) -> PersonaResult:
        from . import llm

        prompt = f"{_profile_block(target, [], banned)}\n\nDraft to stress-test:\n{text}"
        critique = llm.complete(_ADVERSARIAL_SYSTEM, prompt, max_tokens=800)
        if critique is not None:
            findings = [] if critique.strip().upper().startswith("PASS") else \
                [line.strip("- ").strip() for line in critique.splitlines() if line.strip()]
            return PersonaResult(text=text, notes=findings, mode="live")

        findings = [f"banned phrase survived revision: '{p}'"
                    for p in find_banned(text, banned)]
        scores = score_text(text)
        for axis in AXES:
            if abs(scores[axis] - target[axis]) > 0.30:
                findings.append(
                    f"{axis} far from target ({scores[axis]:.2f} vs {target[axis]:.2f})"
                )
        return PersonaResult(text=text, notes=findings, mode="offline")
