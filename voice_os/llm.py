"""Optional live completion client.

The anthropic SDK is an optional dependency: every stage of the pipeline has
a deterministic offline implementation, and Claude is layered on top when
credentials resolve (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
`ant auth login` profile).

Privacy: in live mode the draft text, target profile, banned phrases, and
revision signals are sent to the selected calibrated provider. Set
VOICE_OS_OFFLINE=1 to force offline mode for sensitive drafts even when
credentials are present.
"""

from __future__ import annotations

import os
import sys

import httpx

DEFAULT_MODEL = os.environ.get("VOICE_OS_MODEL", "claude-fable-5")
DEFAULT_PROVIDER = os.environ.get("VOICE_OS_PROVIDER", "anthropic")

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
    ):
        instance = super().__new__(cls, value)
        instance.provider = provider
        instance.model = model
        instance.policy_outcome = policy_outcome
        instance.fallback_reason = fallback_reason
        return instance


class ProviderRefusalError(RuntimeError):
    """A successful provider response that declined to generate output."""


def _warn_once(message: str) -> None:
    global _warned
    if not _warned:
        print(f"voice_os: {message} (falling back to offline mode)", file=sys.stderr)
        _warned = True


def get_client():
    """Return an Anthropic client, or None when unavailable or opted out.

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
    """One Claude completion; returns None on failure so callers fall back offline.

    Failures are not silent: the first live-call failure prints a warning to
    stderr so a misconfigured key or model does not quietly demote every run
    to offline mode.
    """
    if os.environ.get("VOICE_OS_OFFLINE"):
        return None
    if os.environ.get("VOICE_OS_PROVIDER_POLICY_ENABLED", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return _route_live_completion(
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
        )

    client = get_client()
    if client is None:
        return None
    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as exc:
        _warn_once(f"live persona call failed ({type(exc).__name__}: {exc})")
        return None


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


def _route_live_completion(
    *, system: str, prompt: str, max_tokens: int
) -> RoutedText:
    from .provider_policy import ProviderPolicyRouter

    router = ProviderPolicyRouter(
        adapters={
            "anthropic": _anthropic_adapter,
            "openai": _openai_adapter,
            "google": _google_adapter,
            "xai": _xai_adapter,
        },
    )
    openai_model = os.environ.get("VOICE_OS_OPENAI_MODEL", "gpt-5.6-sol")
    gemini_model = os.environ.get("VOICE_OS_GEMINI_MODEL", "gemini-3.6-flash")
    xai_model = os.environ.get("VOICE_OS_XAI_MODEL", "grok-4.5")
    anthropic_fallback_model = os.environ.get(
        "VOICE_OS_ANTHROPIC_FALLBACK_MODEL",
        "claude-opus-4-8",
    )
    allowed_providers = {
        value.strip()
        for value in os.environ.get(
            "VOICE_OS_ALLOWED_PROVIDERS",
            DEFAULT_PROVIDER,
        ).split(",")
        if value.strip()
    }
    candidates = [(DEFAULT_PROVIDER, DEFAULT_MODEL)]
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
    allow_degraded = os.environ.get(
        "VOICE_OS_ALLOW_DEGRADED",
        "",
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
        model=result.route.model,
        policy_outcome=result.route.outcome,
        fallback_reason=result.fallback_reason,
    )
