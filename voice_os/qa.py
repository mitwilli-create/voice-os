"""QA gate: banned-phrase enforcement and pass/cycle decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .axes import AxisProfile

PASS_THRESHOLD = 0.80

# Offline replacement suggestions for common banned phrases. Phrases without
# an entry are simply flagged (and deleted by the offline generative persona).
REPLACEMENTS = {
    "i hope this email finds you well": "",
    "i wanted to reach out": "",
    "just checking in": "",
    "please don't hesitate to reach out": "let me know",
    "leverage": "use",
    "synergy": "alignment",
    "deep dive": "closer look",
    "circle back": "follow up",
    "touch base": "talk",
    "at your earliest convenience": "when you can",
    "moving forward": "from here",
    "per my last email": "as I mentioned",
}


def load_banned_list(path: str) -> list[str]:
    phrases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                phrases.append(line.lower())
    return phrases


def find_banned(text: str, banned: list[str]) -> list[str]:
    lowered = text.lower()
    return [p for p in banned if re.search(r"\b" + re.escape(p) + r"\b", lowered)]


@dataclass
class GateResult:
    decision: str                      # "pass" or "cycle"
    fidelity: float
    per_axis: dict[str, float]
    banned_hits: list[str]
    revision_signals: list[str] = field(default_factory=list)


def gate(
    scores: dict[str, float],
    baseline: AxisProfile,
    target: dict[str, float],
    banned_hits: list[str],
    threshold: float = PASS_THRESHOLD,
) -> GateResult:
    """Pass/cycle decision with a structured revision signal.

    Fidelity is measured against the register-calibrated target (the baseline
    mean adjusted for channel/audience/situation), using the baseline's
    per-axis spread as the tolerance band.
    """
    calibrated = AxisProfile(mean=target, std=baseline.std)
    fidelity, per_axis = calibrated.fidelity(scores)

    signals = [f"remove banned phrase: '{p}'" for p in banned_hits]
    for axis, closeness in sorted(per_axis.items(), key=lambda kv: kv[1]):
        if closeness < 0.5:
            direction = "raise" if scores[axis] < target[axis] else "lower"
            signals.append(
                f"{direction} {axis}: draft {scores[axis]:.2f} vs target {target[axis]:.2f}"
            )

    decision = "pass" if fidelity >= threshold and not banned_hits else "cycle"
    return GateResult(
        decision=decision,
        fidelity=fidelity,
        per_axis=per_axis,
        banned_hits=banned_hits,
        revision_signals=signals,
    )
