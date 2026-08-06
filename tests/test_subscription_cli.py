"""Unit tests for voice_os.subscription_cli.

These mock subprocess.run so the suite never shells out to a real `claude`
or `codex` binary (see the autouse fixture in tests/conftest.py, which also
holds VOICE_OS_SUBSCRIPTION_FIRST off for every other test in the suite).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from voice_os import subscription_cli


class _CompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _request(**overrides):
    base = {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "system": "system",
        "prompt": "draft",
        "max_tokens": 64,
    }
    base.update(overrides)
    return base


def test_anthropic_subscription_complete_parses_json_result(monkeypatch):
    monkeypatch.setenv("VOICE_OS_CLAUDE_CLI_BIN", "/usr/bin/true")
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _CompletedProcess(
            returncode=0,
            stdout=json.dumps({"is_error": False, "result": "CANARY_OK"}),
        )

    monkeypatch.setattr(subscription_cli.subprocess, "run", fake_run)

    text = subscription_cli.anthropic_subscription_complete(_request())

    assert text == "CANARY_OK"
    assert "--safe-mode" in captured["argv"]
    assert "--strict-mcp-config" in captured["argv"]
    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL


def test_anthropic_subscription_complete_never_bypasses_wrapper_on_key_absence(
    monkeypatch,
):
    """Regardless of ANTHROPIC_API_KEY, the CLI binary invoked is the
    configured wrapper path; this adapter never reads or forwards an API
    key itself."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("VOICE_OS_CLAUDE_CLI_BIN", "/usr/bin/true")
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _CompletedProcess(
            returncode=0,
            stdout=json.dumps({"is_error": False, "result": "CANARY_OK"}),
        )

    monkeypatch.setattr(subscription_cli.subprocess, "run", fake_run)

    text = subscription_cli.anthropic_subscription_complete(_request())

    assert text == "CANARY_OK"
    assert captured["argv"][0] == "/usr/bin/true"


def test_anthropic_subscription_complete_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setenv("VOICE_OS_CLAUDE_CLI_BIN", "/usr/bin/true")
    monkeypatch.setattr(
        subscription_cli.subprocess,
        "run",
        lambda *a, **k: _CompletedProcess(returncode=1, stderr="boom"),
    )

    with pytest.raises(subscription_cli.SubscriptionUnavailable, match="exited 1"):
        subscription_cli.anthropic_subscription_complete(_request())


def test_anthropic_subscription_complete_raises_on_malformed_json(monkeypatch):
    monkeypatch.setenv("VOICE_OS_CLAUDE_CLI_BIN", "/usr/bin/true")
    monkeypatch.setattr(
        subscription_cli.subprocess,
        "run",
        lambda *a, **k: _CompletedProcess(returncode=0, stdout="not json"),
    )

    with pytest.raises(subscription_cli.SubscriptionUnavailable, match="non-JSON"):
        subscription_cli.anthropic_subscription_complete(_request())


def test_anthropic_subscription_complete_raises_on_is_error_payload(monkeypatch):
    monkeypatch.setenv("VOICE_OS_CLAUDE_CLI_BIN", "/usr/bin/true")
    monkeypatch.setattr(
        subscription_cli.subprocess,
        "run",
        lambda *a, **k: _CompletedProcess(
            returncode=0,
            stdout=json.dumps({"is_error": True, "result": "refused"}),
        ),
    )

    with pytest.raises(subscription_cli.SubscriptionUnavailable, match="reported an error"):
        subscription_cli.anthropic_subscription_complete(_request())


def test_anthropic_subscription_complete_raises_on_empty_result(monkeypatch):
    monkeypatch.setenv("VOICE_OS_CLAUDE_CLI_BIN", "/usr/bin/true")
    monkeypatch.setattr(
        subscription_cli.subprocess,
        "run",
        lambda *a, **k: _CompletedProcess(
            returncode=0, stdout=json.dumps({"is_error": False, "result": "   "})
        ),
    )

    with pytest.raises(subscription_cli.SubscriptionUnavailable, match="empty result"):
        subscription_cli.anthropic_subscription_complete(_request())


def test_anthropic_subscription_complete_raises_on_timeout(monkeypatch):
    monkeypatch.setenv("VOICE_OS_CLAUDE_CLI_BIN", "/usr/bin/true")

    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr(subscription_cli.subprocess, "run", fake_run)

    with pytest.raises(subscription_cli.SubscriptionUnavailable, match="failed to run"):
        subscription_cli.anthropic_subscription_complete(_request())


def test_openai_subscription_complete_reads_output_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICE_OS_CODEX_CLI_BIN", "/usr/bin/true")
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        out_index = argv.index("-o") + 1
        out_path = argv[out_index]
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write("CANARY_OK\n")
        return _CompletedProcess(returncode=0)

    monkeypatch.setattr(subscription_cli.subprocess, "run", fake_run)

    text = subscription_cli.openai_subscription_complete(
        _request(provider="openai", model="gpt-5.6-sol")
    )

    assert text == "CANARY_OK"
    assert "--sandbox" in captured["argv"]
    assert "read-only" in captured["argv"]
    assert "-m" in captured["argv"]
    assert "gpt-5.6-sol" in captured["argv"]
    # The temp output file is cleaned up after reading.
    out_index = captured["argv"].index("-o") + 1
    import os

    assert not os.path.exists(captured["argv"][out_index])


def test_openai_subscription_complete_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setenv("VOICE_OS_CODEX_CLI_BIN", "/usr/bin/true")
    monkeypatch.setattr(
        subscription_cli.subprocess,
        "run",
        lambda *a, **k: _CompletedProcess(returncode=1, stderr="boom"),
    )

    with pytest.raises(subscription_cli.SubscriptionUnavailable, match="exited 1"):
        subscription_cli.openai_subscription_complete(
            _request(provider="openai", model="gpt-5.6-sol")
        )


def test_resolve_binary_prefers_env_override(monkeypatch):
    monkeypatch.setenv("VOICE_OS_CLAUDE_CLI_BIN", "/custom/path/claude")
    assert (
        subscription_cli._resolve_binary("VOICE_OS_CLAUDE_CLI_BIN", "claude")
        == "/custom/path/claude"
    )


def test_resolve_binary_raises_when_nothing_found(monkeypatch):
    monkeypatch.delenv("VOICE_OS_CLAUDE_CLI_BIN", raising=False)
    monkeypatch.setattr(subscription_cli.shutil, "which", lambda _name: None)

    with pytest.raises(subscription_cli.SubscriptionUnavailable):
        subscription_cli._resolve_binary(
            "VOICE_OS_CLAUDE_CLI_BIN", "definitely-not-a-real-binary-xyz"
        )
