"""CLI for the evaluation harness.

    python3 -m voice_os.harness run [--per-cell 6] [--cap 72] [--live] [--json]
    python3 -m voice_os.harness gate [--summary PATH] [--baseline PATH]
                                     [--update-baseline] [--force] [--json]
    python3 -m voice_os.harness runs [--json]
    python3 -m voice_os.harness report <run_id> [--json]

Exit codes for gate: 0 pass, 1 regression, 2 no baseline / bad input.

Design: docs/eval-harness.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import (
    baseline_path,
    harness_history,
    harness_run,
    load_summary,
    write_baseline,
)
from .gate import gate as run_gate
from .gate import var_dir_for


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value}"


def render_summary(summary: dict) -> str:
    overall = summary.get("overall", {})
    mode = summary.get("mode", {})
    lines = [
        "Voice OS generation eval",
        "=" * 34,
        f"cases: {summary.get('cases')}  "
        f"personas={mode.get('personas')} judge={mode.get('judge')} "
        f"embed={mode.get('embed_backend')}",
        "",
        f"alignment_offline: {_fmt(overall.get('alignment_offline'))}"
        + (
            f"   alignment_judged: {_fmt(overall.get('alignment_judged'))}"
            if overall.get("alignment_judged") is not None
            else ""
        ),
        f"paired style:      {_fmt(overall.get('style_overall'))}",
        f"similarity:        content={_fmt(overall.get('similarity_content'))} "
        f"surface={_fmt(overall.get('similarity_surface'))}",
        f"pass rate:         {_fmt(overall.get('pass_rate'))}   "
        f"banned={_fmt(overall.get('banned_hit_rate'))} "
        f"emdash={_fmt(overall.get('em_dash_rate'))}",
    ]
    for label, group in (
        ("channel", summary.get("by_channel", {})),
        ("audience", summary.get("by_audience", {})),
    ):
        if not group:
            continue
        lines.append("")
        lines.append(f"alignment_offline by {label}:")
        for key, stats in group.items():
            judged = stats.get("alignment_judged")
            lines.append(
                f"  {key:16} {_fmt(stats.get('alignment_offline'))}"
                f"  style={_fmt(stats.get('style_overall'))}"
                + (f"  judged={judged}" if judged is not None else "")
                + f"  (n={stats.get('n')})"
            )
    return "\n".join(lines)


def render_gate(result: dict) -> str:
    lines = [f"eval gate: {result['status'].upper()}"]
    for record in result["checks"]:
        status = record["status"]
        if status.startswith("skipped"):
            lines.append(f"  {record['metric']:44} {status}")
        else:
            lines.append(
                f"  {record['metric']:44} {record['baseline']} -> "
                f"{record['current']} (delta {record['delta']:+}, "
                f"tol {record['tolerance']}) {status}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice_os.harness", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run one evaluation (default: offline)")
    run_p.add_argument("--per-cell", type=int, default=6)
    run_p.add_argument("--cap", type=int, default=72)
    run_p.add_argument("--max-revisions", type=int, default=2)
    run_p.add_argument("--live", action="store_true")
    run_p.add_argument("--var-dir", default=None)
    run_p.add_argument("--json", action="store_true", dest="as_json")

    gate_p = sub.add_parser("gate", help="compare against the stored baseline")
    gate_p.add_argument("--summary", default=None, help="existing summary JSON")
    gate_p.add_argument("--baseline", default=None)
    gate_p.add_argument("--update-baseline", action="store_true")
    gate_p.add_argument("--force", action="store_true")
    gate_p.add_argument("--var-dir", default=None)
    gate_p.add_argument("--json", action="store_true", dest="as_json")

    runs_p = sub.add_parser("runs", help="list prior eval runs")
    runs_p.add_argument("--var-dir", default=None)
    runs_p.add_argument("--json", action="store_true", dest="as_json")

    report_p = sub.add_parser("report", help="print a stored run summary")
    report_p.add_argument("run_id")
    report_p.add_argument("--var-dir", default=None)
    report_p.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2

    if args.command == "run":
        result = harness_run(
            per_cell=args.per_cell,
            cap=args.cap,
            max_revisions=args.max_revisions,
            live=args.live,
            var_dir=args.var_dir,
        )
        if args.as_json:
            print(json.dumps(result, indent=1, sort_keys=True))
        else:
            print(render_summary(result["summary"]))
            print(f"\nrun_id: {result['run_id']}")
            print(f"summary: {result['summary_path']}")
        return 0

    if args.command == "gate":
        try:
            if args.summary:
                current = load_summary(args.summary)
            else:
                current = harness_run(var_dir=args.var_dir)["summary"]
        except (OSError, ValueError) as exc:
            print(f"eval gate: cannot load summary ({exc})", file=sys.stderr)
            return 2

        path = args.baseline or baseline_path(args.var_dir)
        if not os.path.exists(path):
            if args.update_baseline:
                write_baseline(current, path)
                print(f"eval gate: baseline established at {path}")
                return 0
            print(
                f"eval gate: no baseline at {path}; establish one with "
                "--update-baseline",
                file=sys.stderr,
            )
            return 2

        try:
            result = run_gate(current, load_summary(path))
        except ValueError as exc:
            print(f"eval gate: bad baseline ({exc})", file=sys.stderr)
            return 2
        result["baseline_path"] = path
        print(
            json.dumps(result, indent=1, sort_keys=True)
            if args.as_json
            else render_gate(result)
        )
        if args.update_baseline:
            if result["status"] == "pass" or args.force:
                write_baseline(current, path)
                print(f"eval gate: baseline updated at {path}")
            else:
                print(
                    "eval gate: refusing to move the baseline over a "
                    "regression (pass --force to override)",
                    file=sys.stderr,
                )
                return 1
        return 0 if result["status"] == "pass" else 1

    if args.command == "runs":
        runs = harness_history(var_dir=args.var_dir)
        if args.as_json:
            print(json.dumps(runs, indent=1, sort_keys=True))
        elif not runs:
            print("no eval runs recorded")
        else:
            for record in runs:
                print(
                    f"{record['run_id']}  cases={record['cases']} "
                    f"align_off={_fmt(record['alignment_offline'])} "
                    f"style={_fmt(record['style_overall'])}"
                )
        return 0

    # report
    directory = os.path.join(var_dir_for(args.var_dir), "eval", "reports")
    path = os.path.join(directory, f"{args.run_id}.summary.json")
    try:
        summary = load_summary(path)
    except (OSError, ValueError) as exc:
        print(f"report: cannot load {path} ({exc})", file=sys.stderr)
        return 2
    print(
        json.dumps(summary, indent=1, sort_keys=True)
        if args.as_json
        else render_summary(summary)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
