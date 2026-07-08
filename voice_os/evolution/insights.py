"""Deterministic insight generation over evolution data.

Surfaces non-obvious voice shifts by audience, medium, era, or goal as
ranked human-readable findings. No LLM anywhere: insights are
threshold-and-effect-size math over pattern profiles, so the scheduled
drift path stays zero-cost and offline-deterministic.

Stdlib only. Design: docs/evolution.md.
"""

from __future__ import annotations

from ..store import iter_chunks
from .patterns import (
    DEFAULT_MARKERS,
    NEAR_ZERO,
    extract_pattern_profile,
)
from .timeline import _chunks_dir, evolution_timeline

# An insight needs this much normalized effect to surface, and only the
# strongest TOP_K survive, so the report stays readable.
MIN_EFFECT = 0.5
TOP_K = 10
MIN_SLICE_CHUNKS = 20

_CONTRAST_FAMILIES = (
    ("greetings", "greeting", "share"),
    ("signoffs", "signoff", "share"),
    ("markers_per_100w", "marker", "per 100w"),
)


def _effect(baseline: float, value: float) -> float:
    return abs(value - baseline) / max(abs(baseline), NEAR_ZERO)


def contrast_insights(
    slices: dict[str, dict], global_profile: dict, kind: str
) -> list[dict]:
    """Slice-vs-global contrasts, e.g. leadership greetings vs global."""
    findings = []
    for label in sorted(slices):
        profile = slices[label]
        for family, noun, unit in _CONTRAST_FAMILIES:
            base_values = global_profile.get(family, {})
            for key in sorted(profile.get(family, {})):
                if key == "other":
                    continue
                value = profile[family][key]
                base = base_values.get(key, 0.0)
                effect = _effect(base, value)
                if effect < MIN_EFFECT:
                    continue
                direction = "over-indexes" if value > base else "under-indexes"
                findings.append(
                    {
                        "kind": f"{kind}-contrast",
                        "subject": label,
                        "effect": round(effect, 4),
                        "text": (
                            f"{kind} '{label}' {direction} on {noun} "
                            f"'{key}' ({value} vs {base} global, {unit})"
                        ),
                    }
                )
        base_mean = global_profile.get("sentence_length", {}).get("mean", 0.0)
        mean = profile.get("sentence_length", {}).get("mean", 0.0)
        effect = _effect(base_mean, mean)
        if effect >= MIN_EFFECT:
            comparative = "longer" if mean > base_mean else "shorter"
            findings.append(
                {
                    "kind": f"{kind}-contrast",
                    "subject": label,
                    "effect": round(effect, 4),
                    "text": (
                        f"{kind} '{label}' runs {comparative} sentences "
                        f"({mean} vs {base_mean} words global)"
                    ),
                }
            )
    return findings


def era_insights(window_groups: list[dict]) -> list[dict]:
    """Window-over-window pattern shifts across the dated history."""
    findings = []
    for earlier, later in zip(window_groups, window_groups[1:]):
        before = earlier.get("patterns", {})
        after = later.get("patterns", {})
        for family, noun, unit in _CONTRAST_FAMILIES:
            base_values = before.get(family, {})
            for key in sorted(set(base_values) | set(after.get(family, {}))):
                if key == "other":
                    continue
                base = base_values.get(key, 0.0)
                value = after.get(family, {}).get(key, 0.0)
                if base <= NEAR_ZERO and value <= NEAR_ZERO:
                    continue
                effect = _effect(base, value)
                if effect < MIN_EFFECT:
                    continue
                direction = "rose" if value > base else "fell"
                findings.append(
                    {
                        "kind": "era-shift",
                        "subject": later.get("group", ""),
                        "effect": round(effect, 4),
                        "text": (
                            f"{noun} '{key}' {direction} in "
                            f"{later.get('group', '')} ({base} -> {value}, "
                            f"{unit})"
                        ),
                    }
                )
    return findings


def _rank(findings: list[dict], top_k: int) -> list[dict]:
    findings.sort(
        key=lambda f: (-f["effect"], f["kind"], f["subject"], f["text"])
    )
    return findings[:top_k]


def generate_insights(
    chunks_dir: str | None = None,
    *,
    top_k: int = TOP_K,
    markers: tuple[str, ...] = DEFAULT_MARKERS,
    min_slice_chunks: int = MIN_SLICE_CHUNKS,
) -> list[dict]:
    """Ranked insights across audience, medium, goal, and era.

    One pass over the chunk store builds the global profile and the
    context slices; the era layer reuses evolution_timeline. Returns
    the top_k findings sorted by effect size (ties broken
    lexicographically, so output is deterministic).
    """
    all_texts: list[tuple[str, str]] = []
    sliced: dict[str, dict[str, list[tuple[str, str]]]] = {
        "audience": {},
        "medium": {},
        "goal": {},
    }
    for chunk in iter_chunks(_chunks_dir(chunks_dir)):
        text = chunk.get("text")
        if not isinstance(text, str) or not text:
            continue
        sort_key = str(chunk.get("hash", ""))
        all_texts.append((sort_key, text))
        context = chunk.get("context", {})
        for kind in sliced:
            value = context.get(kind)
            if isinstance(value, str) and value and value != "unknown":
                sliced[kind].setdefault(value, []).append((sort_key, text))

    if not all_texts:
        return []
    all_texts.sort()
    global_profile = extract_pattern_profile(
        [text for _, text in all_texts], markers
    )

    findings: list[dict] = []
    for kind, groups in sliced.items():
        profiles = {}
        for label, pairs in groups.items():
            if len(pairs) < min_slice_chunks:
                continue
            pairs.sort()
            profiles[label] = extract_pattern_profile(
                [text for _, text in pairs], markers
            )
        findings.extend(contrast_insights(profiles, global_profile, kind))

    windows = evolution_timeline(
        chunks_dir, group_by="window", markers=markers
    )
    findings.extend(era_insights(windows))
    return _rank(findings, top_k)
