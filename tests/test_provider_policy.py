from __future__ import annotations

import os

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


def test_default_openweight_route_is_available_for_toil():
    calls = []
    router = ProviderPolicyRouter(
        adapters={
            "openrouter": lambda request: calls.append(request) or "revised",
        },
        env={"OPENROUTER_API_KEY": "present"},
    )

    result = router.route(
        provider="openrouter",
        model="openai/gpt-oss-120b",
        system="system",
        prompt="draft",
        max_tokens=100,
        allow_degraded=True,
    )

    assert result.text == "revised"
    assert result.route == ProviderRoute(
        provider="openrouter",
        model="openai/gpt-oss-120b",
        outcome="degraded",
    )
    assert len(calls) == 1


def test_default_voice_route_is_subscription_billed_not_metered():
    """The default route must cost nothing at the margin.

    REPLACES test_default_voice_route_is_openweight_and_non_anthropic, which
    asserted a google default and a blanket non-Anthropic rule. Mitchell
    reversed that policy on 2026-08-06: Claude subscription primary, OpenAI
    subscription fallback, Google deprecated for this pipeline.

    The old test's PROTECTIVE intent was billing, not vendor identity: keep the
    default off a metered Anthropic API key. That intent is preserved and made
    stricter here. The default is now asserted to be a subscription route by
    name, and "anthropic" (the metered SDK provider, distinct from the
    subscription "claude_cli" route) is asserted NOT to be the default. A
    regression that silently repointed the default at the metered key would
    fail this test, which is the failure the old one existed to catch.
    """
    router = ProviderPolicyRouter(
        adapters={},
        env={"OPENROUTER_API_KEY": "present"},
    )

    route = router.explain(provider="openrouter", model="openai/gpt-oss-120b")

    subscription_providers = {"claude_cli", "codex_cli"}
    assert llm.DEFAULT_PROVIDER in subscription_providers
    # The metered Anthropic SDK path must never be the default.
    assert llm.DEFAULT_PROVIDER != "anthropic"
    from voice_os.provider_policy import CALIBRATED_ROUTES as _ROUTES

    assert (llm.DEFAULT_PROVIDER, llm.DEFAULT_MODEL) in _ROUTES
    assert route["outcome"] == "degraded"
    assert route["calibrated"] is True


def test_subscription_routes_do_not_depend_on_a_metered_api_key():
    """A subscription route must not become eligible via a metered key.

    Guards the billing inversion directly: if claude_cli's credential were
    ANTHROPIC_API_KEY, then having that key present would make the "free"
    route eligible while the actual spend landed on the metered account.
    """
    router = ProviderPolicyRouter(
        adapters={},
        env={"ANTHROPIC_API_KEY": "present", "OPENAI_API_KEY": "present"},
    )

    assert router.explain(provider="claude_cli", model="opus")[
        "credential_present"
    ] is False
    assert router.explain(provider="codex_cli", model="gpt-5.6-sol")[
        "credential_present"
    ] is False

    enabled = ProviderPolicyRouter(
        adapters={},
        env={"CAREER_OPS_SUBSCRIPTION_CLI_ENABLED": "true"},
    )
    assert enabled.explain(provider="claude_cli", model="opus")[
        "credential_present"
    ] is True


def test_anthropic_requires_explicit_compatibility_opt_in():
    router = ProviderPolicyRouter(
        adapters={"anthropic": lambda _request: "should not run"},
        env={"ANTHROPIC_API_KEY": "present"},
    )

    with pytest.raises(ProviderPolicyHardStop, match="anthropic_provider_prohibited"):
        router.route(
            provider="anthropic",
            model="claude-opus-4-8",
            system="system",
            prompt="draft",
            max_tokens=16,
        )


def test_anthropic_adapter_uses_low_effort_for_fable(monkeypatch):
    monkeypatch.setenv("VOICE_OS_FABLE_EFFORT", "low")
    captured = {}

    class Response:
        stop_reason = "end_turn"
        content = [type("Block", (), {"type": "text", "text": "CANARY_OK"})()]

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Response()

    class Client:
        messages = Messages()

    monkeypatch.setattr(llm, "get_client", lambda: Client())

    result = llm._anthropic_adapter(
        {
            "model": "claude-fable-5",
            "system": "system",
            "prompt": "synthetic draft",
            "max_tokens": 64,
        }
    )

    assert result == "CANARY_OK"
    assert captured["output_config"] == {"effort": "low"}


def test_anthropic_adapter_surfaces_fable_refusal(monkeypatch):
    class Response:
        stop_reason = "refusal"
        content = []

    class Messages:
        def create(self, **_kwargs):
            return Response()

    class Client:
        messages = Messages()

    monkeypatch.setattr(llm, "get_client", lambda: Client())

    with pytest.raises(llm.ProviderRefusalError):
        llm._anthropic_adapter(
            {
                "model": "claude-fable-5",
                "system": "system",
                "prompt": "synthetic draft",
                "max_tokens": 64,
            }
        )


def test_live_path_always_uses_provider_policy(monkeypatch):
    calls = []
    monkeypatch.delenv("VOICE_OS_OFFLINE", raising=False)
    monkeypatch.setattr(
        llm,
        "_route_live_completion",
        lambda **kwargs: calls.append(kwargs) or "CANARY_OK",
    )

    result = llm.complete("system", "synthetic draft", max_tokens=64)

    assert result == "CANARY_OK"
    assert calls == [{"system": "system", "prompt": "synthetic draft", "max_tokens": 64}]


def test_additional_voice_fallbacks_are_registered_as_degraded():
    router = ProviderPolicyRouter(
        adapters={},
        env={
            "GEMINI_API_KEY": "present",
            "XAI_API_KEY": "present",
        },
    )
    google = router.explain(provider="google", model="gemini-3.6-flash")
    xai = router.explain(provider="xai", model="grok-4.5")
    assert google["outcome"] == "degraded"
    assert google["calibrated"] is True
    assert xai["outcome"] == "degraded"
    assert xai["calibrated"] is True


def test_openrouter_voice_fallback_is_registered_as_degraded():
    router = ProviderPolicyRouter(
        adapters={"openrouter": lambda request: "CANARY_OK"},
        env={"OPENROUTER_API_KEY": "present"},
    )

    route = router.explain(provider="openrouter", model="openai/gpt-oss-120b")

    assert route["outcome"] == "degraded"
    assert route["calibrated"] is True
    assert route["credential_present"] is True


def test_openrouter_adapter_uses_openai_compatible_payload(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "CANARY_OK"}}]}

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    monkeypatch.setattr(llm.httpx, "post", post)

    result = llm._openrouter_adapter(
        {
            "model": "openai/gpt-oss-120b",
            "system": "system",
            "prompt": "synthetic draft",
            "max_tokens": 64,
        }
    )

    assert result == "CANARY_OK"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer present"
    assert captured["json"]["model"] == "openai/gpt-oss-120b"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "synthetic draft"},
    ]


def test_google_adapter_excludes_thought_summary_and_requests_minimal_thinking(
    monkeypatch,
):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "internal summary", "thought": True},
                                {"text": "CANARY_OK"},
                            ]
                        }
                    }
                ]
            }

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    monkeypatch.setenv("GEMINI_API_KEY", "present")
    monkeypatch.setattr(llm.httpx, "post", post)

    result = llm._google_adapter(
        {
            "model": "gemini-3.6-flash",
            "system": "system",
            "prompt": "synthetic draft",
            "max_tokens": 64,
        }
    )

    assert result == "CANARY_OK"
    assert captured["json"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "minimal"
    }


@pytest.mark.parametrize(
    ("adapter_name", "credential_key", "model"),
    [
        ("_openai_adapter", "OPENAI_API_KEY", "gpt-5.6-sol"),
        ("_xai_adapter", "XAI_API_KEY", "grok-4.5"),
    ],
)
def test_responses_adapters_disable_storage_and_minimize_reasoning(
    monkeypatch,
    adapter_name,
    credential_key,
    model,
):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": "CANARY_OK"}

    def post(_url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setenv(credential_key, "present")
    monkeypatch.setattr(llm.httpx, "post", post)

    result = getattr(llm, adapter_name)(
        {
            "model": model,
            "system": "system",
            "prompt": "synthetic draft",
            "max_tokens": 64,
        }
    )

    assert result == "CANARY_OK"
    assert captured["json"]["store"] is False
    if adapter_name == "_xai_adapter":
        assert captured["json"]["reasoning"] == {"effort": "low"}


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
        env={"ANTHROPIC_API_KEY": "present", "VOICE_OS_ALLOW_ANTHROPIC": "true"},
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
        env={"ANTHROPIC_API_KEY": "present", "VOICE_OS_ALLOW_ANTHROPIC": "true"},
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
            "VOICE_OS_ALLOW_ANTHROPIC": "true",
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


def test_anthropic_credit_balance_400_uses_explicitly_allowed_fallback():
    calls = []

    class CreditBalanceError(RuntimeError):
        status_code = 400

    def anthropic(_request):
        calls.append("anthropic")
        raise CreditBalanceError(
            "Your credit balance is too low to access the Anthropic API."
        )

    def openai(_request):
        calls.append("openai")
        return "CANARY_OK"

    router = ProviderPolicyRouter(
        adapters={"anthropic": anthropic, "openai": openai},
        env={
            "ANTHROPIC_API_KEY": "present",
            "OPENAI_API_KEY": "present",
            "VOICE_OS_ALLOW_ANTHROPIC": "true",
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

    assert result.text == "CANARY_OK"
    assert result.fallback_reason == "provider_rate_quota"
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
            "VOICE_OS_ALLOW_ANTHROPIC": "true",
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
            "VOICE_OS_ALLOW_ANTHROPIC": "true",
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


@pytest.mark.parametrize("status_code", [429, 500])
def test_httpx_retryable_status_uses_next_provider(status_code):
    calls = []

    def openai(_request):
        calls.append("openai")
        request = llm.httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = llm.httpx.Response(status_code, request=request)
        raise llm.httpx.HTTPStatusError(
            "provider failure",
            request=request,
            response=response,
        )

    router = ProviderPolicyRouter(
        adapters={
            "openai": openai,
            "google": lambda _request: calls.append("google") or "CANARY_OK",
        },
        env={
            "OPENAI_API_KEY": "present",
            "GEMINI_API_KEY": "present",
        },
    )

    result = router.route_candidates(
        candidates=[
            ("openai", "gpt-5.6-sol"),
            ("google", "gemini-3.6-flash"),
        ],
        system="system",
        prompt="synthetic draft",
        max_tokens=16,
        allow_degraded=True,
        allowed_providers={"openai", "google"},
    )

    assert result.text == "CANARY_OK"
    assert result.route.provider == "google"
    assert calls == ["openai", "google"]


def test_empty_provider_output_uses_next_allowed_provider():
    calls = []
    router = ProviderPolicyRouter(
        adapters={
            "openai": lambda _request: calls.append("openai") or "",
            "google": lambda _request: calls.append("google") or "CANARY_OK",
        },
        env={
            "OPENAI_API_KEY": "present",
            "GEMINI_API_KEY": "present",
        },
    )

    result = router.route_candidates(
        candidates=[
            ("openai", "gpt-5.6-sol"),
            ("google", "gemini-3.6-flash"),
        ],
        system="system",
        prompt="synthetic draft",
        max_tokens=16,
        allow_degraded=True,
        allowed_providers={"openai", "google"},
    )

    assert result.text == "CANARY_OK"
    assert result.fallback_reason == "provider_malformed_response"
    assert calls == ["openai", "google"]


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
    monkeypatch.setenv("VOICE_OS_ALLOW_ANTHROPIC", "true")
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


@pytest.mark.parametrize(
    ("failure", "fallback_reason"),
    [
        (
            type(
                "CapacityError",
                (RuntimeError,),
                {"status_code": 429},
            )("usage limit reached"),
            "provider_rate_quota",
        ),
        (
            llm.ProviderRefusalError("Fable refused the request"),
            "provider_refusal",
        ),
    ],
)
def test_fable_failure_uses_equivalent_opus_before_cross_provider(
    monkeypatch,
    failure,
    fallback_reason,
):
    calls = []

    def anthropic_adapter(request):
        calls.append(request["model"])
        if request["model"] == "claude-fable-5":
            raise failure
        return "CANARY_OK"

    monkeypatch.setattr(llm, "DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "DEFAULT_MODEL", "claude-fable-5")
    monkeypatch.setattr(llm, "_anthropic_adapter", anthropic_adapter)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.setenv("VOICE_OS_ALLOW_ANTHROPIC", "true")
    monkeypatch.setenv(
        "VOICE_OS_ANTHROPIC_FALLBACK_MODEL",
        "claude-opus-4-8",
    )
    monkeypatch.setenv("VOICE_OS_ALLOW_DEGRADED", "true")
    monkeypatch.setenv("VOICE_OS_ALLOWED_PROVIDERS", "anthropic")

    result = llm._route_live_completion(
        system="system",
        prompt="synthetic draft",
        max_tokens=16,
    )

    assert result == "CANARY_OK"
    assert result.provider == "anthropic"
    assert result.model == "claude-opus-4-8"
    assert result.policy_outcome == "equivalent"
    assert result.fallback_reason == fallback_reason
    assert calls == ["claude-fable-5", "claude-opus-4-8"]


def test_live_completion_prefers_google_then_uses_openai_if_unavailable(
    monkeypatch,
):
    calls = []

    class CapacityError(RuntimeError):
        status_code = 429

    class UnavailableError(RuntimeError):
        status_code = 503

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
        "_google_adapter",
        lambda _request: calls.append("google")
        or (_ for _ in ()).throw(UnavailableError("service unavailable")),
    )
    monkeypatch.setattr(
        llm,
        "_openai_adapter",
        lambda _request: calls.append("openai") or "CANARY_OK",
        raising=False,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.setenv("VOICE_OS_ALLOW_ANTHROPIC", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    monkeypatch.setenv("GEMINI_API_KEY", "present")
    monkeypatch.setenv("VOICE_OS_ALLOW_DEGRADED", "true")
    monkeypatch.setenv(
        "VOICE_OS_ALLOWED_PROVIDERS",
        "anthropic,openai,google",
    )
    monkeypatch.setenv("VOICE_OS_OPENAI_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("VOICE_OS_GEMINI_MODEL", "gemini-3.6-flash")

    result = llm._route_live_completion(
        system="system",
        prompt="synthetic draft",
        max_tokens=16,
    )

    assert result == "CANARY_OK"
    assert result.provider == "openai"
    assert result.model == "gpt-5.6-sol"
    assert result.policy_outcome == "degraded"
    assert result.fallback_reason == "provider_unavailable"
    assert calls == ["anthropic", "google", "openai"]


def test_live_completion_omits_credentialed_but_disallowed_provider(monkeypatch):
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
        lambda _request: calls.append("openai") or "WRONG_PROVIDER",
    )
    monkeypatch.setattr(
        llm,
        "_google_adapter",
        lambda _request: calls.append("google") or "CANARY_OK",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")
    monkeypatch.setenv("VOICE_OS_ALLOW_ANTHROPIC", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "present")
    monkeypatch.setenv("GEMINI_API_KEY", "present")
    monkeypatch.setenv("VOICE_OS_ALLOW_DEGRADED", "true")
    monkeypatch.setenv("VOICE_OS_ALLOWED_PROVIDERS", "anthropic,google")

    result = llm._route_live_completion(
        system="system",
        prompt="synthetic draft",
        max_tokens=16,
    )

    assert result == "CANARY_OK"
    assert result.provider == "google"
    assert calls == ["anthropic", "google"]


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


def test_plain_text_completion_records_non_anthropic_default_provider(
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

    assert result.provider == "openrouter"
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
