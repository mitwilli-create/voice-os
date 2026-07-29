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

DEFAULT_MODEL = os.environ.get("VOICE_OS_MODEL", "claude-opus-4-8")
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
    ):
        instance = super().__new__(cls, value)
        instance.provider = provider
        instance.model = model
        instance.policy_outcome = policy_outcome
        return instance


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
    response = client.messages.create(
        model=request["model"],
        max_tokens=request["max_tokens"],
        system=request["system"],
        messages=[{"role": "user", "content": request["prompt"]}],
    )
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def _route_live_completion(
    *, system: str, prompt: str, max_tokens: int
) -> RoutedText:
    from .provider_policy import ProviderPolicyRouter

    router = ProviderPolicyRouter(
        adapters={"anthropic": _anthropic_adapter},
    )
    result = router.route(
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL,
        system=system,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    return RoutedText(
        result.text,
        provider=result.route.provider,
        model=result.route.model,
        policy_outcome=result.route.outcome,
    )
