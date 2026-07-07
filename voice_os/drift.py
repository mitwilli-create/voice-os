"""Voice evolution tracking: windowed axis profiles and drift flags.

Pure math over (timestamp, scores) pairs and per-window token counts, so
everything here is unit-testable without file I/O; the batch side that
reads the chunk store and writes the drift_report artifact lives in
mine/drift.py.

Detection reuses the tolerance formula from AxisProfile.fidelity
(max(2 * std, 0.12)): an axis is flagged when its half-year window mean
departs from the trailing four-window mean by more than that tolerance
for two consecutive windows. Lexical marker pairs (like the known 2020
"yea" to "yeah" transition) get per-window frequency series with a
crossover detector.

Everything produced here is a proposal. Tier boundaries in
voice_os/corpus.py remain the single source of truth and change only by
explicit human edit.
"""

from __future__ import annotations

from .axes import AXES

MIN_WINDOW_CHUNKS = 30
TRAILING_WINDOWS = 4
SUSTAIN_WINDOWS = 2
TOLERANCE_FLOOR = 0.12


def window_key(timestamp: str) -> str | None:
    """Half-year bucket for an ISO timestamp: '2020-08-01...' -> '2020H2'.

    Returns None for malformed timestamps (non-numeric year or month,
    month out of range) so callers can skip them instead of crashing.
    """
    if not timestamp or len(timestamp) < 7:
        return None
    year, month_str = timestamp[:4], timestamp[5:7]
    if not (year.isdigit() and month_str.isdigit()):
        return None
    month = int(month_str)
    if not 1 <= month <= 12:
        return None
    return f"{year}H{1 if month <= 6 else 2}"


def window_profiles(
    dated_scores: list[tuple[str, dict[str, float]]],
    min_chunks: int = MIN_WINDOW_CHUNKS,
) -> list[dict]:
    """Per-half-year axis means over (timestamp, axis_scores) pairs.

    Malformed timestamps are skipped. Windows with fewer than min_chunks
    samples are dropped; mine/drift.py reports how many via its
    stats.windows_dropped field. Returns chronologically sorted window
    dicts.
    """
    buckets: dict[str, list[dict[str, float]]] = {}
    for timestamp, scores in dated_scores:
        key = window_key(timestamp)
        if key is None:
            continue
        buckets.setdefault(key, []).append(scores)

    windows = []
    for key in sorted(buckets):
        scores_list = buckets[key]
        if len(scores_list) < min_chunks:
            continue
        mean = {
            axis: round(sum(s[axis] for s in scores_list) / len(scores_list), 4)
            for axis in AXES
        }
        windows.append({"window": key, "n_chunks": len(scores_list), "axis_mean": mean})
    return windows


def flag_shifts(
    windows: list[dict],
    min_windows: int = SUSTAIN_WINDOWS,
    trailing: int = TRAILING_WINDOWS,
) -> list[dict]:
    """Sustained axis shifts against a trailing-window baseline.

    A shift is flagged when |window mean - trailing mean| exceeds
    max(2 * trailing std, 0.12) for min_windows consecutive windows. One
    flag is emitted per sustained run, anchored at the window where the
    run began.
    """
    flags: list[dict] = []
    for axis in AXES:
        run: dict | None = None
        run_length = 0
        for i in range(trailing, len(windows)):
            value = windows[i]["axis_mean"][axis]

            if run is None:
                # Fresh trailing baseline from the windows before this one.
                history = [w["axis_mean"][axis] for w in windows[i - trailing : i]]
                trailing_mean = sum(history) / len(history)
                variance = sum((v - trailing_mean) ** 2 for v in history) / len(history)
                tolerance = max(2.0 * variance ** 0.5, TOLERANCE_FLOOR)
                delta = value - trailing_mean
                if abs(delta) > tolerance:
                    run = {
                        "axis": axis,
                        "window": windows[i]["window"],
                        "from": round(trailing_mean, 4),
                        "to": round(value, 4),
                        "delta": round(delta, 4),
                        "tolerance": round(tolerance, 4),
                        "_baseline": trailing_mean,
                        "_positive": delta > 0,
                    }
                    run_length = 1
                    if run_length == min_windows:
                        flags.append(run)
                continue

            # A run is active: hold the pre-shift baseline fixed, otherwise
            # the shift inflates its own trailing statistics and masks the
            # sustained departure it caused.
            delta = value - run["_baseline"]
            if abs(delta) > run["tolerance"] and (delta > 0) == run["_positive"]:
                run_length += 1
                if run_length == min_windows:
                    flags.append(run)
            else:
                run = None
                run_length = 0

    for flag in flags:
        flag.pop("_baseline", None)
        flag.pop("_positive", None)
    flags.sort(key=lambda f: (f["window"], f["axis"]))
    return flags


def marker_series(
    counts_by_window: dict[str, dict[str, float]],
    markers: list[tuple[str, str]],
    word_totals: dict[str, float],
) -> dict:
    """Per-window frequency series and crossover window per marker pair.

    counts_by_window maps window -> {form: raw count}; word_totals maps
    window -> total words. The crossover is the first window where the
    new form outruns the old one and never falls back behind it.
    """
    series: dict[str, dict] = {}
    for old, new in markers:
        rows = []
        for window in sorted(counts_by_window):
            words = max(word_totals.get(window, 0.0), 1.0)
            rows.append({
                "window": window,
                f"{old}_per_100w": round(
                    counts_by_window[window].get(old, 0.0) * 100 / words, 4
                ),
                f"{new}_per_100w": round(
                    counts_by_window[window].get(new, 0.0) * 100 / words, 4
                ),
            })
        crossover = None
        for i, row in enumerate(rows):
            if row[f"{new}_per_100w"] > row[f"{old}_per_100w"]:
                if all(
                    later[f"{new}_per_100w"] >= later[f"{old}_per_100w"]
                    for later in rows[i:]
                ):
                    crossover = row["window"]
                    break
        series[f"{old}->{new}"] = {"series": rows, "crossover": crossover}
    return series


def suggest_boundaries(flags: list[dict], markers: dict | None = None) -> list[str]:
    """Human-readable proposals. Never applied automatically."""
    suggestions = []
    for flag in flags:
        direction = "rose" if flag["delta"] > 0 else "fell"
        suggestions.append(
            f"{flag['axis']} {direction} by {abs(flag['delta']):.2f} starting "
            f"{flag['window']} (from {flag['from']:.2f} to {flag['to']:.2f}, "
            f"tolerance {flag['tolerance']:.2f}); consider whether the tier "
            "boundaries in voice_os/corpus.py still bracket a stable voice period"
        )
    for name, data in (markers or {}).items():
        if data.get("crossover"):
            suggestions.append(
                f"lexical marker {name} crossed over in {data['crossover']}; "
                "generation should prefer the newer form"
            )
    return suggestions
