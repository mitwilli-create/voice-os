from __future__ import annotations

import pytest

from voice_os import _last_provider_route
from voice_os import llm
from voice_os.personas import GenerativePersona, PersonaResult
from voice_os.product.graph import _stamp_live_model
from voice_os.provider_policy import (
    CompletionResult,
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


def test_registered_hard_stop_explains_that_it_is_blocked():
    route = ProviderRoute(
        provider="anthropic",
        model="blocked-model",
        outcome="hard_stop",
    )
    router = ProviderPolicyRouter(
        adapters={},
        registry={("anthropic", "blocked-model"): route},
    )
    plan = router.explain(provider="anthropic", model="blocked-model")
    assert plan["outcome"] == "hard_stop"
    assert plan["reason"] == "hard_stopped"


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

def test_retryable_capacity_failure_uses_explicitly_allowed_degraded_fallback():
    calls = []

    class CapacityError(RuntimeError):
        status_code = 429

    def anthropic(_request):
        calls.append("anthropic")
        raise CapacityError("usage limit reached")

    def openai(_request):
        calls.append("openai")
        return "CANARY_OK"

    router = ProviderPolicyRouter(
        adapters={"anthropic": anthropic, "openai": openai},
        env={
            "ANTHROPIC_API_KEY": "present",
            "OPENAI_API_KEY": "present",
        },
    )

    result = router.route_candidates(
        candidates=[
            ("anthropic", "claude-opus-4-8"),
            ("openai", "gpt-5.6-sol"),
        ],
        system="system",
        prompt="synthetic draft",
        max_tokens=16,
        allow_degraded=True,
        allowed_providers={"anthropic", "openai"},
    )

    assert result == CompletionResult(
        text="CANARY_OK",
        route=ProviderRoute(
            provider="openai",
            model="gpt-5.6-sol",
            outcome="degraded",
        ),
        fallback_reason="provider_rate_quota",
    )
    assert calls == ["anthropic", "openai"]


def test_degraded_fallback_requires_flag_and_provider_allowlist():
    calls = []

    class CapacityError(RuntimeError):
        status_code = 429

    router = ProviderPolicyRouter(
        adapters={
            "anthropic": lambda _request: (_ for _ in ()).throw(
                CapacityError("usage limit reached")
            ),
            "openai": lambda request: calls.append(request) or "CANARY_OK",
        },
        env={
            "ANTHROPIC_API_KEY": "present",
            "OPENAI_API_KEY": "present",
        },
    )
    candidates = [
        ("anthropic", "claude-opus-4-8"),
        ("openai", "gpt-5.6-sol"),
    ]

    with pytest.raises(ProviderPolicyHardStop):
        router.route_candidates(
            candidates=candidates,
            system="system",
            prompt="synthetic draft",
            max_tokens=16,
            allow_degraded=False,
            allowed_providers={"anthropic", "openai"},
        )
    with pytest.raises(ProviderPolicyHardStop):
        router.route_candidates(
            candidates=candidates,
            system="system",
            prompt="synthetic draft",
            max_tokens=16,
            allow_degraded=True,
            allowed_providers={"anthropic"},
        )
    assert calls == []


def test_invalid_request_does_not_cross_provider_fallback():
    calls = []

    class InvalidRequestError(RuntimeError):
        status_code = 400

    router = ProviderPolicyRouter(
        adapters={
            "anthropic": lambda _request: (_ for _ in ()).throw(
                InvalidRequestError("invalid request")
            ),
            "openai": lambda request: calls.append(request) or "CANARY_OK",
        },
        env={
            "ANTHROPIC_API_KEY": "present",
            "OPENAI_API_KEY": "present",
        },
    )

    with pytest.raises(ProviderPolicyHardStop, match="provider_invalid_request"):
        router.route_candidates(
            candidates=[
                ("anthropic", "claude-opus-4-8"),
                ("openai", "gpt-5.6-sol"),
            ],
            system="system",
            prompt="synthetic draft",
            max_tokens=16,
            allow_degraded=True,
            allowed_providers={"anthropic", "openai"},
        )
    assert calls == []

def test_live_completion_uses_authorized_openai_fallback_and_stamps_reason(
    monkeypatch,
):
    calls = []

    class CapacityError(RuntimeError):
        status_code = 429

    monkeypatch.setattr(llm, "DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "DEFAULT_MODEL", "claude-opus-4-8")
    monkeypatch.setattr(
        llm,
        "_anthropic_adapter",
        lambda _request: calls.append("anthropic")
        or (_ for _ in ()).throw(CapacityError("usage limit reached")),
    )
    monkeypatch.setattr(
        llm,
        "_openai_adapter",
        lambda _request: calls.append("openai") or "CANARY_OK",
        raising=False,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    monkeypatch.setenv("VOICE_OS_ALLOW_DEGRADED", "true")
    monkeypatch.setenv("VOICE_OS_ALLOWED_PROVIDERS", "anthropic,openai")
    monkeypatch.setenv("VOICE_OS_OPENAI_MODEL", "gpt-5.6-sol")

    result = llm._route_live_completion(
        system="system",
        prompt="synthetic draft",
        max_tokens=16,
    )

    assert result == "CANARY_OK"
    assert result.provider == "openai"
    assert result.model == "gpt-5.6-sol"
    assert result.policy_outcome == "degraded"
    assert result.fallback_reason == "provider_rate_quota"
    assert calls == ["anthropic", "openai"]


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


def test_legacy_plain_text_completion_records_actual_anthropic_provider(
    monkeypatch,
):
    monkeypatch.setattr(llm, "DEFAULT_PROVIDER", "openai")
    monkeypatch.setattr(
        llm,
        "complete",
        lambda *args, **kwargs: "Legacy Anthropic result.",
    )
    target = {
        "rhetorical_pace": 0.5,
        "risk_tolerance": 0.5,
        "sentence_rhythm": 0.5,
        "escalation_pattern": 0.5,
        "hedging_behavior": 0.5,
        "editorial_register": 0.5,
    }

    result = GenerativePersona().revise("Draft.", target, [], [])

    assert result.provider == "anthropic"
    assert result.policy_outcome is None


def test_live_route_stamps_checkpoint_provenance():
    state = {"provenance": {"voice_os_version": "test", "live_model": None}}
    result = PersonaResult(
        text="Draft.",
        notes=[],
        mode="live",
        provider="openai",
        model="gpt-5.6-sol",
        policy_outcome="degraded",
        fallback_reason="provider_rate_quota",
    )

    update = _stamp_live_model(state, result)

    assert update["provenance"] == {
        "voice_os_version": "test",
        "live_model": "gpt-5.6-sol",
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "policy_outcome": "degraded",
        "fallback_reason": "provider_rate_quota",
    }


def test_unrouted_live_call_clears_stale_policy_outcome():
    state = {
        "provenance": {
            "provider": "anthropic",
            "model": "model-a",
            "policy_outcome": "equivalent",
        }
    }
    result = PersonaResult(
        text="Draft.",
        notes=[],
        mode="live",
        provider="anthropic",
        model="model-a",
        policy_outcome=None,
    )

    update = _stamp_live_model(state, result)

    assert "policy_outcome" not in update["provenance"]


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
