"""Evolution CLI: python3 -m voice_os.evolution <command>.

Commands:
    timeline   grouped evolution over the chunk store
    insights   ranked deterministic insights
    check      pure drift check against the stored baseline
    baseline   establish/update the pattern baseline from Tier 1 data
    drift-run  one checkpointed graph run (writes evolution_flags.json)
    runs       prior drift runs (all, or one run's steps with --run-id)

The scheduled entry point is `drift-run`; this repo carries no launchd
plists, so scheduling stays external (career-ops conventions) and
points at that command.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import (
    check_drift,
    ensure_baseline,
    evolution_timeline,
    extract_pattern_profile,
    generate_insights,
    tier1_texts,
)


def _print(payload, as_json: bool) -> None:
    if as_json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        print()
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "text" in item:
                print(f"[{item.get('effect', '')}] {item['text']}")
            else:
                print(json.dumps(item, sort_keys=True))
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def _common_options(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Shared options, accepted both before and after the subcommand.

    The subcommand copies use SUPPRESS defaults so an omitted
    post-command flag never overwrites a value given pre-command.
    """
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("--chunks-dir", default=default)
    parser.add_argument("--var-dir", default=default)
    parser.add_argument("--mined-dir", default=default)
    if suppress:
        parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    else:
        parser.add_argument("--json", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m voice_os.evolution")
    _common_options(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_command(name: str) -> argparse.ArgumentParser:
        command = sub.add_parser(name)
        _common_options(command, suppress=True)
        return command

    timeline_parser = add_command("timeline")
    timeline_parser.add_argument(
        "--group-by", choices=("window", "year", "tier"), default="window"
    )
    timeline_parser.add_argument("--audience", default=None)
    timeline_parser.add_argument("--medium", default=None)
    timeline_parser.add_argument("--goal", default=None)
    timeline_parser.add_argument("--min-chunks", type=int, default=1)

    insights_parser = add_command("insights")
    insights_parser.add_argument("--top-k", type=int, default=10)

    add_command("check")

    add_command("baseline")

    drift_parser = add_command("drift-run")
    drift_parser.add_argument("--update-baseline", action="store_true")

    runs_parser = add_command("runs")
    runs_parser.add_argument("--run-id", default=None)

    args = parser.parse_args(argv)

    if args.command == "timeline":
        slice_by = {
            key: value
            for key, value in (
                ("audience", args.audience),
                ("medium", args.medium),
                ("goal", args.goal),
            )
            if value
        }
        _print(
            evolution_timeline(
                args.chunks_dir,
                group_by=args.group_by,
                slice_by=slice_by or None,
                min_chunks=args.min_chunks,
            ),
            args.json,
        )
    elif args.command == "insights":
        _print(generate_insights(args.chunks_dir, top_k=args.top_k), args.json)
    elif args.command == "check":
        _print(check_drift(args.chunks_dir, args.var_dir), args.json)
    elif args.command == "baseline":
        profile = extract_pattern_profile(tier1_texts(args.chunks_dir))
        windows = evolution_timeline(args.chunks_dir, group_by="window")
        manifest, created = ensure_baseline(
            {"profile": profile, "windows": windows}, var_dir=args.var_dir
        )
        _print({"manifest": manifest, "created": created}, args.json)
    elif args.command == "drift-run":
        from . import drift_run

        _print(
            drift_run(
                args.chunks_dir,
                args.var_dir,
                args.mined_dir,
                update_baseline=args.update_baseline,
            ),
            args.json,
        )
    elif args.command == "runs":
        from . import drift_run_history

        _print(drift_run_history(args.run_id, var_dir=args.var_dir), args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
