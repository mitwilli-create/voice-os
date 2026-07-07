"""Optional Claude client.

The anthropic SDK is an optional dependency: every stage of the pipeline has
a deterministic offline implementation, and Claude is layered on top when
credentials resolve (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
`ant auth login` profile).

Privacy: in live mode the draft text, target profile, banned phrases, and
revision signals are sent to the Anthropic API. Set VOICE_OS_OFFLINE=1 to
force offline mode for sensitive drafts even when credentials are present.
"""

from __future__ import annotations

import os
import sys

DEFAULT_MODEL = os.environ.get("VOICE_OS_MODEL", "claude-opus-4-8")

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

_client = None
_client_checked = False
_warned = False


def _warn_once(message: str) -> None:
    global _warned
    if not _warned:
        print(f"voice_os: {message} (falling back to offline mode)", file=sys.stderr)
        _warned = True


def get_client():
    """Return an Anthropic client, or None when unavailable or opted out."""
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    if os.environ.get("VOICE_OS_OFFLINE"):
        return None
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
