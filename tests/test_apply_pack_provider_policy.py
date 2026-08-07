from __future__ import annotations

import pytest

from voice_os.provider_policy import ProviderPolicyHardStop, ProviderPolicyRouter


def test_apply_pack_mode_hard_stops_anthropic_before_adapter():
    calls = []
    router = ProviderPolicyRouter(
        adapters={"anthropic": lambda request: calls.append(request) or "unsafe"},
        env={
            "ANTHROPIC_API_KEY": "present",
            "VOICE_OS_NON_ANTHROPIC_ONLY": "true",
        },
    )

    with pytest.raises(ProviderPolicyHardStop, match="anthropic_provider_prohibited"):
        router.route(
            provider="anthropic",
            model="claude-opus-4-8",
            system="system",
            prompt="draft",
            max_tokens=100,
        )
    assert calls == []
