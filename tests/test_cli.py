"""Offline tests for the package CLI (python3 -m voice_os).

Everything runs offline-deterministic on synthetic fixtures only: the
sample corpus, the fictional Test Person mined artifacts, and a fake KB
under tmp_path. Graph-backed commands skip cleanly when langgraph is
not installed, matching tests/test_product.py conventions.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ["VOICE_OS_OFFLINE"] = "1"

from voice_os.product.cli import main  # noqa: E402

CORPUS = str(REPO_ROOT / "data" / "sample_corpus.txt")
BANNED = str(REPO_ROOT / "data" / "banned_list.txt")
MINED = str(REPO_ROOT / "tests" / "fixtures" / "mined")


def _draft_argv(tmp_path, *extra):
    return [
        "draft",
        "--corpus",
        CORPUS,
        "--mined-dir",
        MINED,
        "--banned-path",
        BANNED,
        "--kb-dir",
        str(tmp_path / "kb"),
        "--var-dir",
        str(tmp_path / "var"),
        *extra,
    ]


def _run(argv, stdin_text=None, capsys=None):
    if stdin_text is not None:
        old_stdin = sys.stdin
        sys.stdin = io.StringIO(stdin_text)
        try:
            code = main(argv)
        finally:
            sys.stdin = old_stdin
    else:
        code = main(argv)
    out, err = capsys.readouterr() if capsys else ("", "")
    return code, out, err


# ---------------------------------------------------------------- draft


def test_draft_stdin_prints_envelope(tmp_path, capsys):
    pytest.importorskip("langgraph")
    code, out, err = _run(
        _draft_argv(tmp_path), stdin_text="quick note about the plan", capsys=capsys
    )
    envelope = json.loads(out)
    assert envelope["decision"] in ("pass", "reject")
    assert code == (0 if envelope["decision"] == "pass" else 1)
    assert envelope["output_text"]
    assert envelope["mode"] == "offline"
    for key in ("run_id", "fidelity", "revisions", "context", "kb", "trace"):
        assert key in envelope


def test_draft_file_input_matches_stdin_shape(tmp_path, capsys):
    pytest.importorskip("langgraph")
    source = tmp_path / "input.txt"
    source.write_text("quick note about the plan", encoding="utf-8")
    code, out, _ = _run(
        _draft_argv(tmp_path, "--file", str(source)), capsys=capsys
    )
    envelope = json.loads(out)
    assert envelope["decision"] in ("pass", "reject")
    assert code in (0, 1)


def test_draft_text_only_prints_plain_text(tmp_path, capsys):
    pytest.importorskip("langgraph")
    code, out, _ = _run(
        _draft_argv(tmp_path, "--text-only"),
        stdin_text="quick note about the plan",
        capsys=capsys,
    )
    assert code in (0, 1)
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
    assert out.strip()


def test_draft_normalizes_friendly_aliases(tmp_path, capsys):
    pytest.importorskip("langgraph")
    _, out, _ = _run(
        _draft_argv(
            tmp_path, "--audience", "boss", "--situation", "high_stakes"
        ),
        stdin_text="quick note about the plan",
        capsys=capsys,
    )
    ctx = json.loads(out)["context"]
    assert ctx["audience"] == "leadership"
    assert ctx["situation"] == "standard"
    assert ctx["stakes"] == "high"


def test_draft_unknown_audience_exits_2(tmp_path, capsys):
    pytest.importorskip("langgraph")
    code, out, err = _run(
        _draft_argv(tmp_path, "--audience", "alien-overlord"),
        stdin_text="hello",
        capsys=capsys,
    )
    assert code == 2
    assert not out
    assert "error:" in err


def test_draft_empty_stdin_exits_2(tmp_path, capsys):
    pytest.importorskip("langgraph")
    code, _, err = _run(_draft_argv(tmp_path), stdin_text="   ", capsys=capsys)
    assert code == 2
    assert "error:" in err


def test_draft_negative_max_revisions_exits_2(tmp_path, capsys):
    pytest.importorskip("langgraph")
    code, _, err = _run(
        _draft_argv(tmp_path, "--max-revisions", "-1"),
        stdin_text="hello",
        capsys=capsys,
    )
    assert code == 2
    assert "error:" in err


# ------------------------------------------------------- history / graph


def test_history_returns_checkpoints_for_run(tmp_path, capsys):
    pytest.importorskip("langgraph")
    _run(
        _draft_argv(tmp_path, "--run-id", "cli-test-run"),
        stdin_text="quick note about the plan",
        capsys=capsys,
    )
    code, out, _ = _run(
        ["history", "cli-test-run", "--var-dir", str(tmp_path / "var")],
        capsys=capsys,
    )
    assert code == 0
    checkpoints = json.loads(out)
    assert isinstance(checkpoints, list)
    assert checkpoints


def test_graph_prints_mermaid(capsys):
    pytest.importorskip("langgraph")
    code, out, _ = _run(["graph"], capsys=capsys)
    assert code == 0
    assert "generate" in out.lower()


# ------------------------------------------------------- exit-code contract


def test_help_returns_0_without_raising(capsys):
    code = main(["draft", "--help"])
    out, _ = capsys.readouterr()
    assert code == 0
    assert "--channel" in out


def test_unknown_flag_returns_2_without_raising(capsys):
    code = main(["draft", "--no-such-flag"])
    capsys.readouterr()
    assert code == 2


def test_broken_pipe_keeps_decision_exit_code(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")

    class _BrokenStdout:
        def write(self, _):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

    source = tmp_path / "input.txt"
    source.write_text("quick note about the plan", encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", _BrokenStdout())
    code = main(_draft_argv(tmp_path, "--file", str(source)))
    assert code in (0, 1)


# ---------------------------------------------------------------- import


def test_main_module_import_is_stdlib_only():
    """python3 -m voice_os must not require langgraph at import time."""
    import subprocess

    probe = (
        "import sys; sys.modules['langgraph'] = None; "
        "import voice_os.__main__; import voice_os.product.cli; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
