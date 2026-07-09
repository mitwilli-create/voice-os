"""CLI for the callable product layer.

    python3 -m voice_os draft [--channel email] [--audience peer]
                              [--situation standard] [--goal unknown]
                              [--stakes STAKES] [--medium MEDIUM]
                              [--max-revisions 2] [--run-id ID]
                              [--file PATH] [--text-only]
                              [--corpus PATH] [--chunks-dir PATH]
                              [--mined-dir PATH] [--banned-path PATH]
                              [--kb-dir PATH] [--var-dir PATH]
    python3 -m voice_os history <run_id> [--var-dir PATH]
    python3 -m voice_os graph

Draft text arrives on stdin by default (heredoc-friendly for shell
callers) or from --file. The full JSON envelope prints to stdout;
--text-only prints just the drafted text. Friendly context aliases
(audience="boss", situation="high_stakes") are accepted; the same
normalization as voice_os.draft() applies.

Exit codes for draft: 0 decision pass, 1 decision reject (envelope
still printed with the best-effort draft), 2 usage, validation, or
dependency error.

Design: docs/callable-layer.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import describe_graph, draft, run_history


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m voice_os",
        description="Draft text in the calibrated voice through the full "
        "generate, critique, gate, revise pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_draft = sub.add_parser("draft", help="draft text in the target voice")
    p_draft.add_argument("--channel", default="email")
    p_draft.add_argument("--audience", default="peer")
    p_draft.add_argument("--situation", default="standard")
    p_draft.add_argument("--goal", default="unknown")
    p_draft.add_argument("--stakes", default=None)
    p_draft.add_argument("--medium", default=None)
    p_draft.add_argument("--max-revisions", type=int, default=2)
    p_draft.add_argument(
        "--redraft",
        action="store_true",
        help="the input is finished writing being re-voiced: output "
        "sentences the input does not entail block a pass",
    )
    p_draft.add_argument("--run-id", default=None)
    p_draft.add_argument(
        "--file", default=None, help="read draft text from a file instead of stdin"
    )
    p_draft.add_argument(
        "--text-only",
        action="store_true",
        help="print only the drafted text, not the JSON envelope",
    )
    p_draft.add_argument("--corpus", default=None, dest="corpus_path")
    p_draft.add_argument("--chunks-dir", default=None)
    p_draft.add_argument("--mined-dir", default=None)
    p_draft.add_argument("--banned-path", default=None)
    p_draft.add_argument("--kb-dir", default=None)
    p_draft.add_argument("--var-dir", default=None)

    p_history = sub.add_parser("history", help="checkpoint summaries for a run")
    p_history.add_argument("run_id")
    p_history.add_argument("--var-dir", default=None)

    sub.add_parser("graph", help="print the compiled graph structure (mermaid)")

    return parser


def _emit(text: str) -> None:
    """Print to stdout, tolerating a consumer that closed the pipe.

    A downstream `head` or broken pipe must not turn a finished draft
    into a spurious exit 2: swallow BrokenPipeError and point stdout at
    devnull so interpreter shutdown does not raise it again.
    """
    try:
        print(text)
        sys.stdout.flush()
    except BrokenPipeError:
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except (OSError, ValueError, AttributeError):
            pass


def _read_text(args: argparse.Namespace) -> str:
    if args.file is not None:
        with open(args.file, encoding="utf-8") as handle:
            return handle.read()
    return sys.stdin.read()


def _cmd_draft(args: argparse.Namespace) -> int:
    text = _read_text(args)
    result = draft(
        text,
        channel=args.channel,
        audience=args.audience,
        situation=args.situation,
        goal=args.goal,
        stakes=args.stakes,
        medium=args.medium,
        max_revisions=args.max_revisions,
        redraft=args.redraft,
        run_id=args.run_id,
        corpus_path=args.corpus_path,
        chunks_dir=args.chunks_dir,
        mined_dir=args.mined_dir,
        banned_path=args.banned_path,
        kb_dir=args.kb_dir,
        var_dir=args.var_dir,
    )
    if args.text_only:
        _emit(result["output_text"])
    else:
        _emit(json.dumps(result, indent=2))
    return 0 if result["decision"] == "pass" else 1


def _cmd_history(args: argparse.Namespace) -> int:
    _emit(json.dumps(run_history(args.run_id, var_dir=args.var_dir), indent=2))
    return 0


def _cmd_graph() -> int:
    _emit(describe_graph())
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        # argparse exits for usage errors (2) and --help (0); keep the
        # main(argv) -> int contract for programmatic callers.
        return exc.code if isinstance(exc.code, int) else 2
    try:
        if args.command == "draft":
            return _cmd_draft(args)
        if args.command == "history":
            return _cmd_history(args)
        return _cmd_graph()
    except (ValueError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
