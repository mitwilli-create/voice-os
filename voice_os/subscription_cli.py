"""CLI-subscription completion routes.

Billing rule (Mitchell, 2026-08): a live draft call must prefer a paid CLI
SUBSCRIPTION over a metered API key, wherever a subscription route exists.
This module is the subscription side of that choice; ``llm.py`` decides,
per provider, whether to try it before its metered API adapter.

This is a BILLING mechanism only. It never changes which provider or model
the six-axis policy router selected -- it only changes how the already-
selected call is paid for. Both adapters here take and return the same
``dict -> str`` shape as the metered adapters in ``llm.py`` so they compose
into ``ProviderPolicyRouter`` without it knowing CLI routing exists.

Known, deliberate limitations of a CLI shell-out versus a direct API call:
  - No temperature, structured-output schema, or token-accounting fields.
  - ``max_tokens`` cannot be enforced; it is passed only as a soft, best-
    effort instruction in the prompt text.
  - Refusal is inferred from empty/blank output, not a typed stop-reason
    (the Anthropic API adapter can detect ``stop_reason == "refusal"``
    directly; the CLI cannot).
  - Latency is higher (a full CLI process start, not a bare HTTP call).
Both are used only for short revision-length text completions, which this
pipeline's personas already produce, so these limits are tolerable here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile


class SubscriptionUnavailable(RuntimeError):
    """The subscription CLI route could not service this request.

    Callers (llm.py's billing-aware adapters) catch this and fall back to
    the same provider's metered API adapter. It must never propagate to
    the policy router as a provider-selection failure.
    """


def _default_timeout() -> float:
    raw = os.environ.get("VOICE_OS_SUBSCRIPTION_TIMEOUT_SECONDS", "120")
    try:
        value = float(raw)
    except ValueError:
        value = 120.0
    return value if value > 0 else 120.0


def _resolve_binary(env_var: str, *candidates: str) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        expanded = os.path.expanduser(candidate)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded
    raise SubscriptionUnavailable(f"no CLI binary found for {env_var}")


def _soft_budget_note(max_tokens: int) -> str:
    return (
        f"Keep the response under roughly {max_tokens} tokens. "
        "This is a soft budget, not a hard cutoff."
    )


def anthropic_subscription_complete(request: dict) -> str:
    """Route one completion through the ``claude`` CLI on Mitchell's Max plan.

    Never bypasses the ~/.claude/bin/claude wrapper: it strips
    ANTHROPIC_API_KEY from the child environment so this call cannot
    silently fall through to metered billing. --safe-mode disables
    CLAUDE.md/skills/hooks (so voice-drafting output stays clean of
    unrelated global instructions) while leaving OAuth subscription auth
    intact; --bare would disable CLAUDE.md too but forces API-key-only
    auth, so it is never used here.
    """
    binary = _resolve_binary("VOICE_OS_CLAUDE_CLI_BIN", "~/.claude/bin/claude", "claude")
    system = request["system"] + "\n\n" + _soft_budget_note(request["max_tokens"])
    argv = [
        binary,
        "-p",
        "--safe-mode",
        "--strict-mcp-config",
        "--system-prompt",
        system,
        "--model",
        request["model"],
        "--output-format",
        "json",
        request["prompt"],
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_default_timeout(),
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SubscriptionUnavailable(f"claude CLI failed to run: {exc}") from exc
    if proc.returncode != 0:
        raise SubscriptionUnavailable(
            f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SubscriptionUnavailable(f"claude CLI returned non-JSON output: {exc}") from exc
    if payload.get("is_error"):
        raise SubscriptionUnavailable(
            f"claude CLI reported an error: {payload.get('result', '')[:500]}"
        )
    text = payload.get("result")
    if not isinstance(text, str) or not text.strip():
        raise SubscriptionUnavailable("claude CLI returned an empty result")
    return text.strip()


def openai_subscription_complete(request: dict) -> str:
    """Route one completion through the ``codex exec`` CLI on Mitchell's ChatGPT plan.

    ``codex exec`` has no separate system-prompt flag, so system and user
    text are concatenated into one prompt. ``-o <file>`` writes just the
    agent's final message (no banner, no token-usage footer), which is
    read back and the temp file removed.
    """
    binary = _resolve_binary("VOICE_OS_CODEX_CLI_BIN", "codex")
    prompt = (
        f"System instructions:\n{request['system']}\n\n"
        f"{_soft_budget_note(request['max_tokens'])}\n\n"
        f"User request:\n{request['prompt']}"
    )
    fd, out_path = tempfile.mkstemp(prefix="voice-os-codex-", suffix=".txt")
    os.close(fd)
    try:
        argv = [
            binary,
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "-m",
            request["model"],
            "-o",
            out_path,
            prompt,
        ]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=_default_timeout(),
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SubscriptionUnavailable(f"codex CLI failed to run: {exc}") from exc
        if proc.returncode != 0:
            raise SubscriptionUnavailable(
                f"codex CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )
        try:
            with open(out_path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise SubscriptionUnavailable(f"codex CLI output file unreadable: {exc}") from exc
        if not text.strip():
            raise SubscriptionUnavailable("codex CLI returned an empty result")
        return text.strip()
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
