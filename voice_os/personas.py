"""Dual-persona routing: a generative persona revises drafts toward the
voice target, an adversarial persona stress-tests the result.

Both personas run deterministically offline; when a Claude client is
available they use it and fall back to the offline path on any failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .axes import AXES, score_text
from .qa import REPLACEMENTS, find_banned

_GENERATIVE_SYSTEM = (
    "You are the generative persona in a voice-fidelity pipeline. Revise the "
    "draft to match the target voice profile while preserving its meaning and "
    "all factual content. Return only the revised draft, no commentary."
)

_ADVERSARIAL_SYSTEM = (
    "You are the adversarial persona in a voice-fidelity pipeline. Stress-test "
    "the draft against the target voice profile. List concrete fidelity "
    "problems, one per line, most severe first. If it is faithful, reply PASS."
)


def _profile_block(target: dict[str, float], signals: list[str], banned: list[str]) -> str:
    lines = ["Target voice profile (0.0 to 1.0 per axis):"]
    lines += [f"  {axis}: {target[axis]:.2f}" for axis in AXES]
    if banned:
        lines.append("Banned phrases (must not appear): " + "; ".join(banned))
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
               signals: list[str]) -> PersonaResult:
        from . import llm

        prompt = f"{_profile_block(target, signals, banned)}\n\nDraft:\n{draft}"
        revised = llm.complete(_GENERATIVE_SYSTEM, prompt)
        if revised:
            return PersonaResult(text=revised, notes=["revised by Claude"], mode="live")
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
        return PersonaResult(text=text.strip(), notes=notes, mode="offline")


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
