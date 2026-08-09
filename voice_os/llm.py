"""Optional live completion client.

Every stage has a deterministic offline implementation. Live work is routed
through the provider policy and the subscription-first frontier failover chain.

Privacy: in live mode the draft text, target profile, banned phrases, and
revision signals are sent to the selected calibrated provider. Set
VOICE_OS_OFFLINE=1 to force offline mode for sensitive drafts even when
credentials are present.
"""

from __future__ import annotations

import os
import sys

import httpx

# Mitchell's current frontier failover order is Claude subscription, ChatGPT/
# Codex subscription, Antigravity/Gemini subscription, then Grok subscription.
# Metered Google and xAI APIs remain later, explicitly configured fallbacks.
DEFAULT_MODEL = os.environ.get("VOICE_OS_MODEL", "opus")
DEFAULT_PROVIDER = os.environ.get("VOICE_OS_PROVIDER", "claude_cli")

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

_client = None
_client_checked = False
_warned = False


class RoutedText(str):
    """A normal string carrying prompt-free provider provenance."""

    def __new__(
        cls,
        value: str,
        *,
        provider: str,
        model: str,
        policy_outcome: str,
        fallback_reason: str | None = None,
        requested_slot: str | None = None,
        resolved_model: str | None = None,
    ):
        instance = super().__new__(cls, value)
        instance.provider = provider
        instance.model = model
        instance.policy_outcome = policy_outcome
        instance.fallback_reason = fallback_reason
        instance.requested_slot = requested_slot
        instance.resolved_model = resolved_model
        return instance


class ProviderRefusalError(RuntimeError):
    """A successful provider response that declined to generate output."""


def _warn_once(message: str) -> None:
    global _warned
    if not _warned:
        print(f"voice_os: {message} (falling back to offline mode)", file=sys.stderr)
        _warned = True


def get_client():
    """Return the legacy Anthropic client for explicit compatibility only.

    VOICE_OS_OFFLINE is checked on every call, before the client cache, so
    the privacy override holds even when the flag is set mid-process after
    a client has already been created.
    """
    global _client, _client_checked
    if os.environ.get("VOICE_OS_OFFLINE"):
        return None
    if _client_checked:
        return _client
    _client_checked = True
    if anthropic is None:
        return None
    try:
        _client = anthropic.Anthropic()
    except Exception as exc:
        _warn_once(f"could not initialize the Anthropic client: {exc}")
        _client = None
    return _client


def complete(system: str, prompt: str, max_tokens: int = 2000) -> str | None:
    """Route one live completion through the non-Anthropic provider policy.

    Failures are not silent: the first live-call failure prints a warning to
    stderr so a misconfigured key or model does not quietly demote every run
    to offline mode.
    """
    if os.environ.get("VOICE_OS_OFFLINE"):
        return None
    try:
        return _route_live_completion(
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        _warn_once(f"live persona call failed ({type(exc).__name__}: {exc})")
        return None


def _claude_cli_adapter(request: dict) -> str:
    """Complete via the Claude Code CLI so the call bills the subscription.

    Mitchell ruled on 2026-08-06: use the subscription, not the metered key.
    Every other adapter in this module spends a metered API key, Google
    included, because the Gemini CLI stopped serving individual users on
    2026-06-18 and left no subscription route on that vendor.

    The wrapper at ~/.claude/bin/claude unsets ANTHROPIC_API_KEY before exec,
    which is what forces the call onto subscription OAuth. We invoke that path
    explicitly rather than a bare "claude" off PATH, because a bare claude can
    resolve to an install that keeps the key and silently meters the run.

    The environment is scrubbed of ANTHROPIC_API_KEY here as well. That is
    belt and braces: if the wrapper is ever replaced by a plain binary, a
    missing key makes the call FAIL rather than quietly bill.
    """
    import subprocess

    wrapper = os.path.expanduser(
        os.environ.get("VOICE_OS_CLAUDE_CLI_PATH", "~/.claude/bin/claude")
    )
    if not os.path.exists(wrapper):
        raise RuntimeError(f"claude CLI wrapper not found at {wrapper}")

    child_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    # The parent Voice OS gate has a 300-second ceiling. Keep one seat bounded
    # so a hung plan-limit response cannot consume the whole ceiling before
    # Fable, Opus, Sonnet, Sol, Terra, and Luna have each been tried.
    timeout_s = int(os.environ.get("VOICE_OS_CLAUDE_CLI_TIMEOUT", "45"))
    argv = [
        wrapper,
        "-p",
        request["prompt"],
        "--model",
        request["model"],
        "--append-system-prompt",
        request["system"],
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=child_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"claude CLI timed out after {timeout_s}s") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()[-300:]
        raise RuntimeError(f"claude CLI exited {completed.returncode}: {stderr}")

    text = (completed.stdout or "").strip()
    if not text:
        raise RuntimeError("claude CLI returned empty output")
    # Surface a refusal as a refusal so the policy can fall back, matching how
    # _anthropic_adapter treats stop_reason "refusal".
    if text.lower().startswith("i can't") or text.lower().startswith("i cannot"):
        raise ProviderRefusalError("claude CLI model refusal")
    return text


def _codex_cli_adapter(request: dict) -> str:
    """Complete via the Codex CLI so the call bills the ChatGPT subscription.

    This is the SECONDARY subscription route per Mitchell's 2026-08-06 ruling,
    behind claude_cli and ahead of every metered provider.

    Verified 2026-08-06: ~/.codex/auth.json reports auth_mode "chatgpt" with an
    OAuth token set and no OPENAI_API_KEY, and an exec run with a deliberately
    bogus OPENAI_API_KEY still returned a correct completion. As with the
    claude_cli adapter, OPENAI_API_KEY is scrubbed from the child environment
    so a broken subscription fails loudly instead of silently metering.

    Run with --skip-git-repo-check and a read-only sandbox: this is a scoring
    call, so it must never be able to write. --cd points at a scratch dir so
    codex does not pick up the AGENTS.md of whatever tree the caller happens to
    be in, which is the same contamination class that disqualified agy.
    """
    import subprocess
    import tempfile

    binary = os.environ.get("VOICE_OS_CODEX_CLI_PATH", "codex")
    child_env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    timeout_s = int(os.environ.get("VOICE_OS_CODEX_CLI_TIMEOUT", "540"))

    with tempfile.TemporaryDirectory(prefix="voice-os-codex-") as scratch:
        argv = [
            binary,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-C",
            scratch,
            "-m",
            request["model"],
            f"{request['system']}\n\n{request['prompt']}",
        ]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=child_env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"codex CLI not found: {binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"codex CLI timed out after {timeout_s}s") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()[-300:]
        raise RuntimeError(f"codex CLI exited {completed.returncode}: {stderr}")

    text = _strip_codex_chrome(completed.stdout or "")
    if not text:
        raise RuntimeError("codex CLI returned empty output")
    return text


def _antigravity_cli_adapter(request: dict) -> str:
    """Complete through the Antigravity/Gemini subscription CLI."""
    import subprocess

    binary = os.environ.get("VOICE_OS_ANTIGRAVITY_CLI_PATH", "agy")
    child_env = {
        k: v for k, v in os.environ.items()
        if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
    }
    timeout_s = int(os.environ.get("VOICE_OS_ANTIGRAVITY_CLI_TIMEOUT", "540"))
    composed = f"{request['system']}\n\n---\n\n{request['prompt']}"
    argv = [
        binary,
        "--output-format", "text",
        "--model", request["model"],
        "--effort", "high",
        "--sandbox",
        "--print", composed,
    ]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, env=child_env
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"antigravity CLI not found: {binary}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"antigravity CLI timed out after {timeout_s}s") from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()[-300:]
        raise RuntimeError(f"antigravity CLI exited {completed.returncode}: {stderr}")
    text = (completed.stdout or "").strip()
    if not text:
        raise RuntimeError("antigravity CLI returned empty output")
    return text


def _grok_cli_adapter(request: dict) -> str:
    """Complete through the Grok subscription CLI."""
    import subprocess

    binary = os.environ.get("VOICE_OS_GROK_CLI_PATH", "grok")
    child_env = {
        k: v for k, v in os.environ.items()
        if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"}
    }
    timeout_s = int(os.environ.get("VOICE_OS_GROK_CLI_TIMEOUT", "540"))
    composed = f"{request['system']}\n\n---\n\n{request['prompt']}"
    argv = [
        binary,
        "--single", composed,
        "--output-format", "plain",
        "--model", request["model"],
        "--no-plan",
        "--no-subagents",
        "--permission-mode", "plan",
    ]
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, env=child_env
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"grok CLI not found: {binary}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"grok CLI timed out after {timeout_s}s") from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()[-300:]
        raise RuntimeError(f"grok CLI exited {completed.returncode}: {stderr}")
    text = (completed.stdout or "").strip()
    if not text:
        raise RuntimeError("grok CLI returned empty output")
    return text


def _strip_codex_chrome(raw: str) -> str:
    """Drop the wrapper lines `codex exec` prints around the model's answer.

    Measured 2026-08-06, a one-line completion came back as:
        warning: Skill descriptions were shortened ...
        codex
        CODEX_OK
        tokens used
        7,365
        CODEX_OK

    so the payload is echoed twice and fenced by chrome. Taking the text after
    the LAST "tokens used" block is wrong when the answer itself is multi-line,
    so instead drop known chrome lines and collapse a duplicated trailing echo.
    """
    lines = [line.rstrip() for line in raw.splitlines()]
    skip_prefixes = ("warning:", "tokens used")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "codex":
            continue
        if any(stripped.lower().startswith(p) for p in skip_prefixes):
            continue
        # the token count sits on its own line directly after "tokens used"
        if stripped.replace(",", "").isdigit():
            continue
        cleaned.append(line)
    if len(cleaned) >= 2 and cleaned[-1].strip() == cleaned[0].strip():
        cleaned = cleaned[:-1]
    return "\n".join(cleaned).strip()


def _anthropic_adapter(request: dict) -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("Anthropic client unavailable")
    options = {
        "model": request["model"],
        "max_tokens": request["max_tokens"],
        "system": request["system"],
        "messages": [{"role": "user", "content": request["prompt"]}],
    }
    if request["model"] == "claude-fable-5":
        options["output_config"] = {
            "effort": os.environ.get("VOICE_OS_FABLE_EFFORT", "low"),
        }
    response = client.messages.create(
        **options,
    )
    if getattr(response, "stop_reason", None) == "refusal":
        raise ProviderRefusalError("Anthropic model refusal")
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

def _openai_adapter(request: dict) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI credentials unavailable")
    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": request["model"],
            "max_output_tokens": request["max_tokens"],
            "store": False,
            "input": [
                {"role": "system", "content": request["system"]},
                {"role": "user", "content": request["prompt"]},
            ],
        },
        timeout=120.0,
    )
    response.raise_for_status()
    payload = response.json()
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts).strip()

def _google_adapter(request: dict) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Google credentials unavailable")
    response = httpx.post(
        (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{request['model']}:generateContent"
        ),
        params={"key": key},
        headers={"Content-Type": "application/json"},
        json={
            "system_instruction": {
                "parts": [{"text": request["system"]}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": request["prompt"]}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": request["max_tokens"],
                "thinkingConfig": {
                    "thinkingLevel": "minimal",
                },
            },
        },
        timeout=120.0,
    )
    response.raise_for_status()
    payload = response.json()
    return "".join(
        part.get("text", "")
        for candidate in payload.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
        if isinstance(part.get("text"), str) and part.get("thought") is not True
    ).strip()


def _xai_adapter(request: dict) -> str:
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("xAI credentials unavailable")
    response = httpx.post(
        "https://api.x.ai/v1/responses",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": request["model"],
            "max_output_tokens": request["max_tokens"],
            "store": False,
            "reasoning": {"effort": "low"},
            "input": [
                {"role": "system", "content": request["system"]},
                {"role": "user", "content": request["prompt"]},
            ],
        },
        timeout=120.0,
    )
    response.raise_for_status()
    payload = response.json()
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return "".join(
        content.get("text", "")
        for item in payload.get("output", [])
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    ).strip()


def _openrouter_adapter(request: dict) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OpenRouter credentials unavailable")
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mitwilli-create/career-ops",
            "X-Title": "career-ops voice-os",
        },
        json={
            "model": request["model"],
            "messages": [
                {"role": "system", "content": request["system"]},
                {"role": "user", "content": request["prompt"]},
            ],
            "max_tokens": request["max_tokens"],
        },
        timeout=120.0,
    )
    response.raise_for_status()
    payload = response.json()
    return str(
        payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    ).strip()


def _route_live_completion(
    *, system: str, prompt: str, max_tokens: int
) -> RoutedText:
    from .provider_policy import ProviderPolicyHardStop, ProviderPolicyRouter

    router = ProviderPolicyRouter(
        adapters={
            "claude_cli": _claude_cli_adapter,
            "codex_cli": _codex_cli_adapter,
            "antigravity_cli": _antigravity_cli_adapter,
            "grok_cli": _grok_cli_adapter,
            "anthropic": _anthropic_adapter,
            "openai": _openai_adapter,
            "google": _google_adapter,
            "xai": _xai_adapter,
            "openrouter": _openrouter_adapter,
        },
    )
    openai_model = os.environ.get("VOICE_OS_OPENAI_MODEL", "gpt-5.6-sol")
    gemini_model = os.environ.get("VOICE_OS_GEMINI_MODEL", "gemini-3.6-flash")
    xai_model = os.environ.get("VOICE_OS_XAI_MODEL", "grok-4.5")
    openrouter_model = os.environ.get(
        "VOICE_OS_OPENROUTER_MODEL", "openai/gpt-oss-120b"
    )
    openrouter_fallback_models = [
        value.strip()
        for value in os.environ.get(
            "VOICE_OS_OPENROUTER_FALLBACK_MODELS",
            "deepseek/deepseek-v4-flash,qwen/qwen3-coder,moonshotai/kimi-k2.6,minimax/minimax-m3",
        ).split(",")
        if value.strip()
    ]
    anthropic_fallback_model = os.environ.get(
        "VOICE_OS_ANTHROPIC_FALLBACK_MODEL",
        "claude-opus-4-8",
    )
    non_anthropic_only = os.environ.get("VOICE_OS_NON_ANTHROPIC_ONLY", "").lower() in {
        "1", "true", "yes", "on",
    }
    allowed_providers = {
        value.strip()
        for value in os.environ.get(
            "VOICE_OS_ALLOWED_PROVIDERS",
            "claude_cli,codex_cli,antigravity_cli,grok_cli,google,xai",
        ).split(",")
        if value.strip()
    }
    if non_anthropic_only:
        allowed_providers.discard("anthropic")
        if DEFAULT_PROVIDER == "anthropic":
            raise ProviderPolicyHardStop(
                "anthropic_provider_prohibited",
                kind="policy",
            )
    candidates = []
    # Subscription ladder, Mitchell's explicit ordering on 2026-08-08:
    #   1. Fable         claude_cli:fable
    #   2. Opus          claude_cli:opus
    #   3. Sonnet        claude_cli:sonnet
    #   4. GPT-5.6 Sol   codex_cli:gpt-5.6-sol
    #   5. Terra         codex_cli:gpt-5.6-terra
    #   6. Luna          codex_cli:gpt-5.6-luna
    #   7. Gemini Pro     antigravity_cli:gemini-3.1-pro
    #   8. Grok 4        grok_cli:grok-4
    #
    # These are appended BEFORE every metered branch below, so a subscription
    # seat is always spent before an API key. Their gate is the subscription
    # flag rather than an API key, because requiring a key here would make the
    # metered credential a precondition for the free route, which is backwards.
    for _sub_provider, _sub_model in (
        ("claude_cli", "fable"),
        ("claude_cli", "opus"),
        ("claude_cli", "sonnet"),
        ("codex_cli", "gpt-5.6-sol"),
        ("codex_cli", "gpt-5.6-terra"),
        ("codex_cli", "gpt-5.6-luna"),
        ("antigravity_cli", "gemini-3.1-pro"),
        ("grok_cli", "grok-4"),
    ):
        if _sub_provider in allowed_providers and (_sub_provider, _sub_model) not in candidates:
            candidates.append((_sub_provider, _sub_model))

    if DEFAULT_PROVIDER not in {"claude_cli", "codex_cli"}:
        candidates.insert(0, (DEFAULT_PROVIDER, DEFAULT_MODEL))
    if (
        DEFAULT_PROVIDER == "anthropic"
        and DEFAULT_MODEL != anthropic_fallback_model
        and "anthropic" in allowed_providers
        and os.environ.get("ANTHROPIC_API_KEY")
    ):
        candidates.append(("anthropic", anthropic_fallback_model))
    if (
        DEFAULT_PROVIDER != "google"
        and "google" in allowed_providers
        and os.environ.get("GEMINI_API_KEY")
    ):
        candidates.append(("google", gemini_model))
    if (
        DEFAULT_PROVIDER != "openai"
        and "openai" in allowed_providers
        and os.environ.get("OPENAI_API_KEY")
    ):
        candidates.append(("openai", openai_model))
    if (
        DEFAULT_PROVIDER != "xai"
        and "xai" in allowed_providers
        and os.environ.get("XAI_API_KEY")
    ):
        candidates.append(("xai", xai_model))
    if (
        DEFAULT_PROVIDER != "openrouter"
        and "openrouter" in allowed_providers
        and os.environ.get("OPENROUTER_API_KEY")
    ):
        candidates.append(("openrouter", openrouter_model))
    if "openrouter" in allowed_providers and os.environ.get("OPENROUTER_API_KEY"):
        for model in openrouter_fallback_models:
            if ("openrouter", model) not in candidates:
                candidates.append(("openrouter", model))
    subscription_enabled = os.environ.get(
        "CAREER_OPS_SUBSCRIPTION_CLI_ENABLED", ""
    ).lower() in {"1", "true", "yes", "on"}
    allow_degraded = os.environ.get(
        "VOICE_OS_ALLOW_DEGRADED",
        "true" if subscription_enabled else "",
    ).lower() in {"1", "true", "yes", "on"}
    result = router.route_candidates(
        candidates=candidates,
        system=system,
        prompt=prompt,
        max_tokens=max_tokens,
        allow_degraded=allow_degraded,
        allowed_providers=allowed_providers,
    )
    return RoutedText(
        result.text,
        provider=result.route.provider,
        model=result.route.resolved_model or result.route.model,
        policy_outcome=result.route.outcome,
        fallback_reason=result.fallback_reason,
        requested_slot=result.route.requested_slot or f"{result.route.provider}:{result.route.model}",
        resolved_model=result.route.resolved_model or result.route.model,
    )
