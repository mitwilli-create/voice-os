"""Tests for drift detection. Synthetic data only, built in-code."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["VOICE_OS_OFFLINE"] = "1"

from mine.cli import main as mine_main  # noqa: E402
from mine.drift import mine_drift  # noqa: E402
from voice_os.axes import AXES  # noqa: E402
from voice_os.drift import (  # noqa: E402
    flag_shifts,
    marker_series,
    suggest_boundaries,
    window_key,
    window_profiles,
)
from voice_os.mined import load_artifacts  # noqa: E402

from test_mine import make_chunk  # noqa: E402


def scores_at(value: float) -> dict[str, float]:
    return {axis: value for axis in AXES}


def windows_with_shift(shift_at: int = 6, low: float = 0.4, high: float = 0.8):
    """Ten half-year windows; every axis jumps from low to high at index shift_at."""
    dated = []
    for i in range(10):
        year = 2016 + i // 2
        month = "03" if i % 2 == 0 else "09"
        value = high if i >= shift_at else low
        for _ in range(35):
            dated.append((f"{year}-{month}-15T12:00:00", scores_at(value)))
    return window_profiles(dated)


def test_window_key_buckets_half_years():
    assert window_key("2020-06-30T23:59:59") == "2020H1"
    assert window_key("2020-07-01T00:00:00") == "2020H2"


def test_window_profiles_drop_thin_windows():
    dated = [("2020-01-15T12:00:00", scores_at(0.5))] * 35
    dated += [("2020-08-15T12:00:00", scores_at(0.5))] * 5  # below min 30
    windows = window_profiles(dated)
    assert [w["window"] for w in windows] == ["2020H1"]
    assert windows[0]["n_chunks"] == 35


def test_sustained_shift_is_flagged_once_per_axis():
    flags = flag_shifts(windows_with_shift())
    axes_flagged = {f["axis"] for f in flags}
    assert axes_flagged == set(AXES)
    # one flag per axis (one sustained run), anchored at the 2019H1 jump
    assert len(flags) == len(AXES)
    assert all(f["window"] == "2019H1" for f in flags)
    assert all(f["delta"] > 0.12 for f in flags)


def test_brief_blip_is_not_flagged():
    dated = []
    for i in range(10):
        year = 2016 + i // 2
        month = "03" if i % 2 == 0 else "09"
        value = 0.8 if i == 6 else 0.4  # single-window blip, not sustained
        for _ in range(35):
            dated.append((f"{year}-{month}-15T12:00:00", scores_at(value)))
    assert flag_shifts(window_profiles(dated)) == []


def test_marker_crossover_detected():
    counts = {
        "2019H1": {"yea": 10.0, "yeah": 2.0},
        "2019H2": {"yea": 8.0, "yeah": 4.0},
        "2020H1": {"yea": 3.0, "yeah": 9.0},
        "2020H2": {"yea": 1.0, "yeah": 12.0},
    }
    words = {w: 1000.0 for w in counts}
    series = marker_series(counts, [("yea", "yeah")], words)
    assert series["yea->yeah"]["crossover"] == "2020H1"
    assert len(series["yea->yeah"]["series"]) == 4


def test_no_crossover_when_old_form_returns():
    counts = {
        "2019H1": {"yea": 10.0, "yeah": 2.0},
        "2019H2": {"yea": 2.0, "yeah": 8.0},
        "2020H1": {"yea": 9.0, "yeah": 3.0},  # old form comes back
    }
    words = {w: 1000.0 for w in counts}
    series = marker_series(counts, [("yea", "yeah")], words)
    assert series["yea->yeah"]["crossover"] is None


def test_suggestions_cover_flags_and_markers():
    flags = flag_shifts(windows_with_shift())
    markers = {"yea->yeah": {"crossover": "2020H1", "series": []}}
    suggestions = suggest_boundaries(flags, markers)
    assert any("editorial_register" in s for s in suggestions)
    assert any("yea->yeah" in s for s in suggestions)
    assert all("corpus.py" in s or "newer form" in s for s in suggestions)


def _era_chunks():
    """Casual yea-era chunks (2018-19) then formal yeah-era (2020-21).

    Four windows per era so the flag detector has a full trailing baseline
    of pre-shift windows when the 2020H1 era change arrives.
    """
    chunks = []
    for i, year_month in enumerate(
        ["2018-03", "2018-09", "2019-03", "2019-09",
         "2020-03", "2020-09", "2021-03", "2021-09"]
    ):
        old_era = i < 4
        for j in range(35):
            text = (
                f"yea lol sounds good {j}, cya soon haha"
                if old_era
                else f"yeah, agreed on point {j}. Let us proceed with the plan "
                "as discussed and confirm the schedule."
            )
            chunk = make_chunk(text, hint=f"era friend {i}", year=2025)
            chunk["provenance"]["timestamp"] = f"{year_month}-15T12:00:00"
            chunk["tier"] = 3
            chunks.append(chunk)
    return chunks


def test_mine_drift_reproduces_marker_transition():
    artifact = mine_drift(_era_chunks(), markers=[("yea", "yeah")])
    data = artifact["data"]
    assert data["stats"]["windows_kept"] == 8
    assert data["markers"]["yea->yeah"]["crossover"] == "2020H1"
    assert any("yea->yeah" in s for s in data["suggestions"])
    # the era change also registers as sustained axis shifts at 2020H1
    assert data["flags"]
    assert all(f["window"] == "2020H1" for f in data["flags"])


def test_mine_drift_requires_dated_chunks():
    chunk = make_chunk("undated text with no timestamp")
    chunk["provenance"]["timestamp"] = None
    try:
        mine_drift([chunk])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_cli_drift_job(tmp_path):
    chunks_dir = tmp_path / "corpus" / "chunks"
    chunks_dir.mkdir(parents=True)
    with open(chunks_dir / "synthetic.jsonl", "w") as f:
        for chunk in _era_chunks():
            f.write(json.dumps(chunk) + "\n")

    out = tmp_path / "corpus" / "mined"
    code = mine_main([
        "run", "--job", "drift",
        "--corpus-dir", str(tmp_path / "corpus"),
        "--out", str(out),
    ])
    assert code == 0
    loaded = load_artifacts(str(out))
    assert loaded.drift_report is not None
    assert loaded.drift_report["markers"]["yea->yeah"]["crossover"]


def test_window_key_rejects_malformed_timestamps():
    assert window_key("not-a-date") is None
    assert window_key("2020-xx-01") is None
    assert window_key("2020-13-01") is None
    assert window_key("") is None
    assert window_key("2020-06-01") == "2020H1"


def test_malformed_timestamps_skipped_not_crashed():
    chunks = _era_chunks()
    bad = make_chunk("text with a broken timestamp")
    bad["provenance"]["timestamp"] = "20xx-99-zz"
    artifact = mine_drift(chunks + [bad], markers=[("yea", "yeah")])
    assert artifact["data"]["stats"]["undated_or_malformed_skipped"] == 1


def test_markers_cover_only_kept_windows():
    chunks = _era_chunks()
    # A sparse window (below min 30 chunks) full of the new marker form
    # must not influence marker series or crossover detection.
    sparse = []
    for j in range(5):
        chunk = make_chunk(f"yeah yeah yeah sparse {j}")
        chunk["provenance"]["timestamp"] = "2017-03-15T12:00:00"
        chunk["tier"] = 3
        sparse.append(chunk)
    artifact = mine_drift(chunks + sparse, markers=[("yea", "yeah")])
    windows_in_series = {
        row["window"]
        for row in artifact["data"]["markers"]["yea->yeah"]["series"]
    }
    assert "2017H1" not in windows_in_series
    assert artifact["data"]["stats"]["windows_dropped"] == 1
