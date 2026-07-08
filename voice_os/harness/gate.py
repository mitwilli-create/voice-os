"""The regression gate: compare a current summary against the baseline.

Pure comparison logic plus the baseline file conventions. Tolerances
are tight because the offline path is deterministic: any change is a
real behavior change, not sampling noise. The baseline moves only by
explicit command (--update-baseline), mirroring the evolution module's
stored-baseline semantics.

Design: docs/eval-harness.md.
"""

from __future__ import annotations

import json
import os

from ..product.kb import REPO_ROOT

# Absolute-drop tolerances (increase tolerances for the two rates).
# Rationale per metric in docs/eval-harness.md, "The regression gate".
DEFAULT_TOLERANCES = {
    "alignment_offline": 0.005,
    "style_overall": 0.005,
    "channel_alignment": 0.02,
    "audience_alignment": 0.02,
    "banned_hit_rate": 0.005,
    "em_dash_rate": 0.005,
}
# Cells below this n are reported, never gated: too noisy to block on.
MIN_GATE_N = 4


def var_dir_for(var_dir: str | None) -> str:
    return (
        var_dir
        or os.environ.get("VOICE_OS_VAR_DIR")
        or os.path.join(REPO_ROOT, "var")
    )


def baseline_path(var_dir: str | None = None) -> str:
    return os.path.join(var_dir_for(var_dir), "eval", "baseline.json")


def load_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict) or "overall" not in summary:
        raise ValueError(f"not a harness summary: {path}")
    return summary


def write_baseline(summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def _check(
    checks: list,
    regressions: list,
    metric: str,
    current: float | None,
    baseline: float | None,
    tolerance: float,
    *,
    higher_is_better: bool,
) -> None:
    """One comparison; missing values on either side are skipped
    (recorded as unchecked) rather than failed, so a new channel or an
    empty cell never blocks unrelated work."""
    if current is None or baseline is None:
        checks.append({"metric": metric, "status": "skipped"})
        return
    delta = current - baseline
    failed = delta < -tolerance if higher_is_better else delta > tolerance
    record = {
        "metric": metric,
        "baseline": baseline,
        "current": current,
        "delta": round(delta, 4),
        "tolerance": tolerance,
        "status": "regression" if failed else "ok",
    }
    checks.append(record)
    if failed:
        regressions.append(record)


def gate(
    current: dict, baseline: dict, tolerances: dict | None = None
) -> dict:
    """Compare two summaries. Empty regressions list = pass."""
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)
    checks: list[dict] = []
    regressions: list[dict] = []

    current_overall = current.get("overall", {})
    baseline_overall = baseline.get("overall", {})
    _check(
        checks,
        regressions,
        "overall.alignment_offline",
        current_overall.get("alignment_offline"),
        baseline_overall.get("alignment_offline"),
        tol["alignment_offline"],
        higher_is_better=True,
    )
    _check(
        checks,
        regressions,
        "overall.style_overall",
        current_overall.get("style_overall"),
        baseline_overall.get("style_overall"),
        tol["style_overall"],
        higher_is_better=True,
    )
    for rate in ("banned_hit_rate", "em_dash_rate"):
        _check(
            checks,
            regressions,
            f"overall.{rate}",
            current_overall.get(rate),
            baseline_overall.get(rate),
            tol[rate],
            higher_is_better=False,
        )

    for group, tol_key in (
        ("by_channel", "channel_alignment"),
        ("by_audience", "audience_alignment"),
    ):
        current_groups = current.get(group, {})
        baseline_groups = baseline.get(group, {})
        for key in sorted(set(current_groups) & set(baseline_groups)):
            cur, base = current_groups[key], baseline_groups[key]
            if cur.get("n", 0) < MIN_GATE_N or base.get("n", 0) < MIN_GATE_N:
                checks.append(
                    {"metric": f"{group}.{key}", "status": "skipped-small-n"}
                )
                continue
            _check(
                checks,
                regressions,
                f"{group}.{key}.alignment_offline",
                cur.get("alignment_offline"),
                base.get("alignment_offline"),
                tol[tol_key],
                higher_is_better=True,
            )

    return {
        "status": "regression" if regressions else "pass",
        "regressions": regressions,
        "checks": checks,
    }
