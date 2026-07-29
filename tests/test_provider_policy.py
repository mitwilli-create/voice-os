from __future__ import annotations

import pytest

from voice_os import _last_provider_route
from voice_os import llm
from voice_os.personas import GenerativePersona, PersonaResult
from voice_os.product.graph import _stamp_live_model
from voice_os.provider_policy import (
    ProviderPolicyHardStop,
    ProviderPolicyRouter,
    ProviderRoute,
)


def test_offline_privacy_override_skips_router(monkeypatch):
    calls = []
    monkeypatch.setenv("VOICE_OS_OFFLINE", "1")
    monkeypatch.setattr(
        llm,
        "_route_live_completion",
        lambda **kwargs: calls.append(kwargs),
    )

    assert llm.complete("system", "private draft") is None
    assert calls == []


def test_default_calibrated_route_is_equivalent():
    calls = []
    router = ProviderPolicyRouter(
        adapters={
            "anthropic": lambda request: calls.append(request) or "revised",
        },
        env={"ANTHROPIC_API_KEY": "present"},
    )

    result = router.route(
        provider="anthropic",
        model="claude-opus-4-8",
        system="system",
        prompt="draft",
        max_tokens=100,
    )

    assert result.text == "revised"
    assert result.route == ProviderRoute(
        provider="anthropic",
        model="claude-opus-4-8",
        outcome="equivalent",
    )
    assert len(calls) == 1


def test_uncalibrated_alternate_hard_stops_before_adapter_call():
    calls = []
    router = ProviderPolicyRouter(
        adapters={"openai": lambda request: calls.append(request) or "nope"},
        env={"OPENAI_API_KEY": "present"},
    )

    with pytest.raises(ProviderPolicyHardStop, match="not calibrated"):
        router.route(
            provider="openai",
            model="gpt-5",
            system="system",
            prompt="private draft",
            max_tokens=100,
        )
    assert calls == []


def test_registered_hard_stop_never_calls_adapter():
    calls = []
    route = ProviderRoute(
        provider="anthropic",
        model="blocked-model",
        outcome="hard_stop",
    )
    router = ProviderPolicyRouter(
        adapters={
            "anthropic": lambda request: calls.append(request) or "nope",
        },
        env={"ANTHROPIC_API_KEY": "present"},
        registry={("anthropic", "blocked-model"): route},
    )

    with pytest.raises(ProviderPolicyHardStop, match="hard-stopped"):
        router.route(
            provider="anthropic",
            model="blocked-model",
            system="system",
            prompt="private draft",
            max_tokens=100,
        )
    assert calls == []


def test_invalid_registry_entries_fail_closed_at_construction():
    invalid_outcome = ProviderRoute(
        provider="anthropic",
        model="model-a",
        outcome="disabled",
    )
    mismatched_key = ProviderRoute(
        provider="anthropic",
        model="model-b",
        outcome="equivalent",
    )

    with pytest.raises(ValueError, match="registry entry"):
        ProviderPolicyRouter(
            adapters={},
            registry={("anthropic", "model-a"): invalid_outcome},
        )
    with pytest.raises(ValueError, match="registry entry"):
        ProviderPolicyRouter(
            adapters={},
            registry={("anthropic", "model-a"): mismatched_key},
        )


def test_provider_failure_never_silently_becomes_offline():
    def fail(_request):
        raise RuntimeError("provider unavailable")

    router = ProviderPolicyRouter(
        adapters={"anthropic": fail},
        env={"ANTHROPIC_API_KEY": "present"},
    )

    with pytest.raises(ProviderPolicyHardStop, match="provider failed"):
        router.route(
            provider="anthropic",
            model="claude-opus-4-8",
            system="system",
            prompt="draft",
            max_tokens=100,
        )


def test_explain_is_deterministic_and_calls_no_adapter():
    calls = []
    router = ProviderPolicyRouter(
        adapters={"anthropic": lambda request: calls.append(request)},
        env={"ANTHROPIC_API_KEY": "present"},
    )
    first = router.explain(provider="anthropic", model="claude-opus-4-8")
    second = router.explain(provider="anthropic", model="claude-opus-4-8")
    assert first == second
    assert first["outcome"] == "equivalent"
    assert calls == []


def test_llm_routed_text_provenance_reaches_persona(monkeypatch):
    routed = llm.RoutedText(
        "Revised text.",
        provider="anthropic",
        model="claude-opus-4-8",
        policy_outcome="equivalent",
    )
    monkeypatch.setattr(llm, "complete", lambda *args, **kwargs: routed)
    target = {
        "rhetorical_pace": 0.5,
        "risk_tolerance": 0.5,
        "sentence_rhythm": 0.5,
        "escalation_pattern": 0.5,
        "hedging_behavior": 0.5,
        "editorial_register": 0.5,
    }

    result = GenerativePersona().revise("Draft.", target, [], [])

    assert result.mode == "live"
    assert result.provider == "anthropic"
    assert result.model == "claude-opus-4-8"
    assert result.policy_outcome == "equivalent"


def test_live_route_stamps_checkpoint_provenance():
    state = {"provenance": {"voice_os_version": "test", "live_model": None}}
    result = PersonaResult(
        text="Draft.",
        notes=[],
        mode="live",
        provider="anthropic",
        model="claude-opus-4-8",
        policy_outcome="equivalent",
    )

    update = _stamp_live_model(state, result)

    assert update["provenance"] == {
        "voice_os_version": "test",
        "live_model": "claude-opus-4-8",
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "policy_outcome": "equivalent",
    }


def test_shared_pipeline_uses_the_actual_last_route():
    cycles = [
        {
            "provider_routes": [
                {
                    "provider": "anthropic",
                    "model": "model-a",
                    "policy_outcome": "equivalent",
                },
                {
                    "provider": "anthropic",
                    "model": "model-b",
                    "policy_outcome": "degraded",
                },
            ]
        }
    ]
    assert _last_provider_route(cycles) == {
        "provider": "anthropic",
        "model": "model-b",
        "policy_outcome": "degraded",
    }
