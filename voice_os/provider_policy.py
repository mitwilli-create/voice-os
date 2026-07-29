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


class ProviderPolicyHardStop(RuntimeError):
    """A live request that policy refuses to send or silently demote."""


CALIBRATED_ROUTES = {
    ("anthropic", "claude-opus-4-8"): ProviderRoute(
        provider="anthropic",
        model="claude-opus-4-8",
        outcome="equivalent",
    ),
}

_CREDENTIAL_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "xai": ("XAI_API_KEY",),
}


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
    ) -> CompletionResult:
        plan = self.explain(provider=provider, model=model)
        if plan["outcome"] == "hard_stop" and plan["calibrated"]:
            raise ProviderPolicyHardStop(
                f"provider route {provider}:{model} is hard-stopped"
            )
        if not plan["calibrated"]:
            raise ProviderPolicyHardStop(
                f"provider route {provider}:{model} is not calibrated"
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
            raise ProviderPolicyHardStop(
                f"provider failed for {provider}:{model}: {type(exc).__name__}"
            ) from None

        if not isinstance(text, str) or not text.strip():
            raise ProviderPolicyHardStop(
                f"provider failed for {provider}:{model}: empty response"
            )
        route = self.registry[(provider, model)]
        return CompletionResult(text=text.strip(), route=route)
