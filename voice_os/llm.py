"""Optional Claude client.

The anthropic SDK is an optional dependency: every stage of the pipeline has
a deterministic offline implementation, and Claude is layered on top when
credentials resolve (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
`ant auth login` profile).
"""

from __future__ import annotations

import os

DEFAULT_MODEL = os.environ.get("VOICE_OS_MODEL", "claude-opus-4-8")

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

_client = None
_client_checked = False


def get_client():
    """Return an Anthropic client, or None when the SDK or credentials are absent."""
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    if anthropic is None:
        return None
    try:
        _client = anthropic.Anthropic()
    except Exception:
        _client = None
    return _client


def complete(system: str, prompt: str, max_tokens: int = 2000) -> str | None:
    """One Claude completion; returns None on any failure so callers fall back offline."""
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
    except Exception:
        return None
