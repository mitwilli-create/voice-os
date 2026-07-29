"""Privacy-first provider policy for live Voice OS completions.

Only independently calibrated provider and model pairs may receive a draft.
Unknown or uncalibrated alternates hard-stop before prompt text reaches an
adapter. Offline mode is handled by llm.py before this module is called.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping

OUTCOMES = ("equivalent", "degraded", "hard_stop")


@dataclass(frozen=True)
class ProviderRoute:
    provider: str
    model: str
    outcome: str


@dataclass(frozen=True)
class CompletionResult:
    text: str
    route: ProviderRoute
    fallback_reason: str | None = None


class ProviderPolicyHardStop(RuntimeError):
    """A live request that policy refuses to send or silently demote."""

    def __init__(self, message: str, *, kind: str = "policy", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


CALIBRATED_ROUTES = {
    ("anthropic", "claude-opus-4-8"): ProviderRoute(
        provider="anthropic",
        model="claude-opus-4-8",
        outcome="equivalent",
    ),
    ("openai", "gpt-5.6-sol"): ProviderRoute(
        provider="openai",
        model="gpt-5.6-sol",
        outcome="degraded",
    ),
}

_CREDENTIAL_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "xai": ("XAI_API_KEY",),
}

def _provider_error_kind(exc: Exception) -> tuple[str, bool]:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    signal = f"{type(exc).__name__} {exc}".lower()
    if (
        status in {402, 429}
        or "usage limit" in signal
        or "rate limit" in signal
        or "quota" in signal
    ):
        return "rate_quota", True
    if status == 408 or "timeout" in signal or "timed out" in signal:
        return "timeout", True
    if status in {401, 403}:
        return "credential", False
    if status == 400:
        return "invalid_request", False
    if status == 529 or (isinstance(status, int) and status >= 500):
        return "unavailable", True
    if "connect" in signal or "unavailable" in signal:
        return "unavailable", True
    return "unknown", False


class ProviderPolicyRouter:
    def __init__(
        self,
        *,
        adapters: Mapping[str, Callable[[dict], str]],
        env: Mapping[str, str] | None = None,
        registry: Mapping[tuple[str, str], ProviderRoute] | None = None,
    ) -> None:
        self.adapters = dict(adapters)
        self.env = os.environ if env is None else env
        self.registry = dict(CALIBRATED_ROUTES if registry is None else registry)
        for (provider, model), route in self.registry.items():
            if (
                route.outcome not in OUTCOMES
                or route.provider != provider
                or route.model != model
            ):
                raise ValueError("invalid provider-policy registry entry")

    def explain(self, *, provider: str, model: str) -> dict:
        route = self.registry.get((provider, model))
        credential_present = any(
            bool(self.env.get(key)) for key in _CREDENTIAL_KEYS.get(provider, ())
        )
        if route is None:
            return {
                "provider": provider,
                "model": model,
                "outcome": "hard_stop",
                "reason": "not_calibrated",
                "calibrated": False,
                "credential_present": credential_present,
                "adapter_present": provider in self.adapters,
            }
        reason = "hard_stopped" if route.outcome == "hard_stop" else "eligible"
        return {
            "provider": provider,
            "model": model,
            "outcome": route.outcome,
            "reason": reason,
            "calibrated": True,
            "credential_present": credential_present,
            "adapter_present": provider in self.adapters,
        }

    def route(
        self,
        *,
        provider: str,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int,
        allow_degraded: bool = False,
        allowed_providers: set[str] | None = None,
    ) -> CompletionResult:
        plan = self.explain(provider=provider, model=model)
        if allowed_providers is not None and provider not in allowed_providers:
            raise ProviderPolicyHardStop(
                "provider_policy_denied",
                kind="policy",
            )
        if plan["outcome"] == "hard_stop" and plan["calibrated"]:
            raise ProviderPolicyHardStop(
                f"provider route {provider}:{model} is hard-stopped"
            )
        if not plan["calibrated"]:
            raise ProviderPolicyHardStop(
                f"provider route {provider}:{model} is not calibrated"
            )
        if plan["outcome"] == "degraded" and not allow_degraded:
            raise ProviderPolicyHardStop(
                "provider_degraded_not_allowed",
                kind="policy",
            )
        if not plan["credential_present"]:
            raise ProviderPolicyHardStop(
                f"provider route {provider}:{model} has no credentials"
            )
        adapter = self.adapters.get(provider)
        if adapter is None:
            raise ProviderPolicyHardStop(
                f"provider route {provider}:{model} has no adapter"
            )

        try:
            text = adapter(
                {
                    "provider": provider,
                    "model": model,
                    "system": system,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                }
            )
        except ProviderPolicyHardStop:
            raise
        except Exception as exc:
            kind, retryable = _provider_error_kind(exc)
            raise ProviderPolicyHardStop(
                f"provider failed: provider_{kind}",
                kind=kind,
                retryable=retryable,
            ) from None

        if not isinstance(text, str) or not text.strip():
            raise ProviderPolicyHardStop(
                "provider_malformed_response",
                kind="malformed_response",
            )
        route = self.registry[(provider, model)]
        return CompletionResult(text=text.strip(), route=route)

    def route_candidates(
        self,
        *,
        candidates: list[tuple[str, str]],
        system: str,
        prompt: str,
        max_tokens: int,
        allow_degraded: bool = False,
        allowed_providers: set[str] | None = None,
    ) -> CompletionResult:
        last_error: ProviderPolicyHardStop | None = None
        fallback_reason: str | None = None
        for provider, model in candidates:
            try:
                result = self.route(
                    provider=provider,
                    model=model,
                    system=system,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    allow_degraded=allow_degraded,
                    allowed_providers=allowed_providers,
                )
                if fallback_reason is None:
                    return result
                return CompletionResult(
                    text=result.text,
                    route=result.route,
                    fallback_reason=fallback_reason,
                )
            except ProviderPolicyHardStop as exc:
                last_error = exc
                if not exc.retryable:
                    raise
                fallback_reason = f"provider_{exc.kind}"
        if last_error is not None:
            raise last_error
        raise ProviderPolicyHardStop("provider_no_candidates", kind="policy")
