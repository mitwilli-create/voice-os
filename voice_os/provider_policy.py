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
ACCOUNT_TYPES = ("subscription", "metered_api")
FAILURE_LEDGER_MAX = 16


@dataclass(frozen=True)
class ProviderRoute:
    provider: str
    model: str
    outcome: str
    account_type: str | None = None
    requested_slot: str | None = None
    resolved_model: str | None = None
    compatibility_label: str | None = None


@dataclass(frozen=True)
class CompletionResult:
    text: str
    route: ProviderRoute
    fallback_reason: str | None = None
    failure_ledger: tuple[dict, ...] = ()


class ProviderPolicyHardStop(RuntimeError):
    """A live request that policy refuses to send or silently demote."""

    def __init__(self, message: str, *, kind: str = "policy", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


CALIBRATED_ROUTES = {
    # Subscription-billed Fable, added 2026-08-07 on Mitchell's ruling: "use the
    # subscription, not the metered key". It is the first seat in the explicit
    # cascade, followed by Opus, Sonnet, Sol, Terra, and Luna.
    ("claude_cli", "fable"): ProviderRoute(
        provider="claude_cli",
        model="fable",
        outcome="equivalent",
        account_type="subscription",
    ),
    # Subscription-billed Opus, added 2026-08-06 on Mitchell's ruling: "use the
    # subscription, not the metered key". It is listed FIRST because it is the
    # only route in this table that costs nothing at the margin. Every other
    # entry here, Google included, bills a metered API key.
    #
    # It reaches the model through ~/.claude/bin/claude, a wrapper that strips
    # ANTHROPIC_API_KEY so the call goes over subscription OAuth. Verified
    # 2026-08-06: invoking that wrapper with a deliberately bogus
    # ANTHROPIC_API_KEY still returns a correct completion, which a metered
    # call could not do.
    #
    # Marked "equivalent" rather than "degraded" on the same basis as the two
    # anthropic entries below: it resolves to the same Opus family, not a
    # smaller model. It is a BILLING path change, not a capability change.
    ("claude_cli", "opus"): ProviderRoute(
        provider="claude_cli",
        model="opus",
        outcome="equivalent",
        account_type="subscription",
    ),
    # SECONDARY subscription route, same ruling. Billed to the ChatGPT
    # subscription via ~/.codex/auth.json auth_mode "chatgpt". Marked
    # "degraded" rather than "equivalent" because, unlike claude_cli, it is a
    # different model family from the one this pipeline's thresholds were
    # calibrated against, so it must not be treated as a like for like swap.
    # It runs only as a fallback, which is exactly the intent.
    ("codex_cli", "gpt-5.6-sol"): ProviderRoute(
        provider="codex_cli",
        model="gpt-5.6-sol",
        outcome="degraded",
        account_type="subscription",
    ),
    # THIRD, same subscription and same wrapper as claude_cli:opus, just a
    # smaller model. "sonnet" is a verified alias of the claude CLI, which
    # advertises fable, opus and sonnet.
    ("claude_cli", "sonnet"): ProviderRoute(
        provider="claude_cli",
        model="sonnet",
        outcome="degraded",
        account_type="subscription",
    ),
    # FOURTH. GPT-5.6 Terra, the balanced middle tier of the same family,
    # added 2026-08-06 on Mitchell's request after the Sol/Terra/Luna family
    # was resolved. Slotted between sonnet and luna because that is the only
    # placement consistent with both his stated order and capability descent.
    # Verified on this machine the same way as the others: codex exec with
    # -m gpt-5.6-terra and a bogus OPENAI_API_KEY returned a completion.
    ("codex_cli", "gpt-5.6-terra"): ProviderRoute(
        provider="codex_cli",
        model="gpt-5.6-terra",
        outcome="degraded",
        account_type="subscription",
    ),
    # FIFTH and last, per Mitchell's 2026-08-06 ordering. "Luna" resolved to
    # GPT-5.6 Luna, the fast and cheap tier of OpenAI's GPT-5.6 family
    # (Sol flagship, Terra balanced, Luna fastest), released 2026-07-09.
    # Confirmed by web search, then verified on this machine: codex exec with
    # -m gpt-5.6-luna and a deliberately bogus OPENAI_API_KEY returned a
    # correct completion, so the id is real AND it bills the subscription.
    # Degraded because it is the smallest model in the ladder.
    ("codex_cli", "gpt-5.6-luna"): ProviderRoute(
        provider="codex_cli",
        model="gpt-5.6-luna",
        outcome="degraded",
        account_type="subscription",
    ),
    ("anthropic", "claude-fable-5"): ProviderRoute(
        provider="anthropic",
        model="claude-fable-5",
        outcome="equivalent",
        account_type="metered_api",
    ),
    ("anthropic", "claude-opus-4-8"): ProviderRoute(
        provider="anthropic",
        model="claude-opus-4-8",
        outcome="equivalent",
        account_type="metered_api",
    ),
    ("openai", "gpt-5.6-sol"): ProviderRoute(
        provider="openai",
        model="gpt-5.6-sol",
        outcome="degraded",
        account_type="metered_api",
    ),
    ("google", "gemini-3.1-pro"): ProviderRoute(
        provider="google",
        model="gemini-3.1-pro",
        outcome="degraded",
        account_type="metered_api",
        requested_slot="google:gemini-3.1-pro",
        resolved_model="gemini-3.1-pro-preview",
    ),
    ("google", "gemini-3.6-flash"): ProviderRoute(
        provider="google",
        model="gemini-3.6-flash",
        outcome="degraded",
        account_type="metered_api",
        requested_slot="google:gemini-3.6-flash",
        resolved_model="gemini-3.6-flash",
    ),
    # Compatibility slots. Keep them callable for older configuration, but
    # route the adapter to the current model and expose the distinction.
    ("google", "gemini-2.5-pro"): ProviderRoute(
        provider="google",
        model="gemini-2.5-pro",
        outcome="degraded",
        account_type="metered_api",
        requested_slot="google:gemini-2.5-pro",
        resolved_model="gemini-3.1-pro-preview",
        compatibility_label="COMPATIBILITY SLOT: current Gemini 3.1 Pro Preview",
    ),
    ("google", "gemini-3-flash"): ProviderRoute(
        provider="google",
        model="gemini-3-flash",
        outcome="degraded",
        account_type="metered_api",
        requested_slot="google:gemini-3-flash",
        resolved_model="gemini-3.6-flash",
        compatibility_label="COMPATIBILITY SLOT: current stable Gemini 3.6 Flash",
    ),
    ("google", "gemini-2.5-flash"): ProviderRoute(
        provider="google",
        model="gemini-2.5-flash",
        outcome="degraded",
        account_type="metered_api",
        requested_slot="google:gemini-2.5-flash",
        resolved_model="gemini-3.6-flash",
        compatibility_label="COMPATIBILITY SLOT: current stable Gemini 3.6 Flash",
    ),
    ("xai", "grok-4.5"): ProviderRoute(
        provider="xai",
        model="grok-4.5",
        outcome="degraded",
        account_type="metered_api",
    ),
    ("openrouter", "openai/gpt-oss-120b"): ProviderRoute(
        provider="openrouter",
        model="openai/gpt-oss-120b",
        outcome="degraded",
        account_type="metered_api",
    ),
    ("openrouter", "deepseek/deepseek-v4-flash"): ProviderRoute(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        outcome="degraded",
        account_type="metered_api",
    ),
    ("openrouter", "qwen/qwen3-coder"): ProviderRoute(
        provider="openrouter",
        model="qwen/qwen3-coder",
        outcome="degraded",
        account_type="metered_api",
    ),
    ("openrouter", "moonshotai/kimi-k2.6"): ProviderRoute(
        provider="openrouter",
        model="moonshotai/kimi-k2.6",
        outcome="degraded",
        account_type="metered_api",
    ),
    ("openrouter", "minimax/minimax-m3"): ProviderRoute(
        provider="openrouter",
        model="minimax/minimax-m3",
        outcome="degraded",
        account_type="metered_api",
    ),
}

_CREDENTIAL_KEYS = {
    # The subscription route has no API key by design, so its "credential" is
    # the explicit opt-in flag that says the subscription CLI is available.
    # Deliberately NOT ANTHROPIC_API_KEY: that key being present must never be
    # what makes this route eligible, or a misconfigured run would bill the
    # metered key while reporting itself as a subscription call.
    "claude_cli": ("CAREER_OPS_SUBSCRIPTION_CLI_ENABLED", "CLAUDE_CODE_OAUTH_TOKEN"),
    "codex_cli": ("CAREER_OPS_SUBSCRIPTION_CLI_ENABLED", "CODEX_OAUTH_TOKEN"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}

def _provider_error_kind(exc: Exception) -> tuple[str, bool]:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    signal = f"{type(exc).__name__} {exc}".lower()
    if "refusal" in signal or "refused" in signal:
        return "refusal", True
    if (
        status in {402, 429}
        or "usage limit" in signal
        or "weekly limit" in signal
        or "plan limit" in signal
        or "rate limit" in signal
        or "quota" in signal
    ):
        return "rate_quota", True
    if status == 408 or "timeout" in signal or "timed out" in signal:
        return "timeout", True
    if status == 401 or any(
        marker in signal
        for marker in (
            "authentication failed",
            "not authenticated",
            "invalid credential",
            "invalid token",
        )
    ):
        return "credential", True
    if status == 403 or any(
        marker in signal
        for marker in (
            "permission denied",
            "not authorized",
            "forbidden",
        )
    ):
        return "authorization", False
    # Command-line adapters have no HTTP status. An otherwise unclassified
    # process exit or missing binary is an availability failure. Quota and
    # timeout markers are classified above so their bounded reason stays exact.
    if any(
        marker in signal
        for marker in (
            "cli exited",
            "cli wrapper not found",
            "cli not found",
        )
    ):
        return "unavailable", True
    # Anthropic reports exhausted prepaid credits as HTTP 400 rather than a
    # quota-shaped 402/429. Treat that specific billing response as retryable
    # so a configured fallback can carry the request. Generic 400s remain
    # terminal because they usually mean the request or model is invalid.
    if status == 400 and any(
        marker in signal
        for marker in (
            "credit balance is too low",
            "insufficient credits",
            "purchase credits",
        )
    ):
        return "rate_quota", True
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
                or (
                    route.outcome != "hard_stop"
                    and route.account_type not in ACCOUNT_TYPES
                )
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
                "requested_slot": f"{provider}:{model}",
                "resolved_model": None,
                "account_type": None,
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
            "requested_slot": route.requested_slot or f"{provider}:{model}",
            "resolved_model": route.resolved_model or model,
            "account_type": route.account_type,
            "compatibility_label": route.compatibility_label,
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
        if provider == "anthropic" and self.env.get(
            "VOICE_OS_ALLOW_ANTHROPIC", ""
        ).lower() not in {"1", "true", "yes", "on"}:
            raise ProviderPolicyHardStop(
                "anthropic_provider_prohibited",
                kind="policy",
            )
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
                f"provider route {provider}:{model} has no credentials",
                kind="credential",
                retryable=True,
            )
        adapter = self.adapters.get(provider)
        if adapter is None:
            raise ProviderPolicyHardStop(
                f"provider route {provider}:{model} has no adapter"
            )

        route = self.registry[(provider, model)]
        requested_slot = route.requested_slot or f"{provider}:{model}"
        resolved_model = route.resolved_model or model
        try:
            text = adapter(
                {
                    "provider": provider,
                    "model": resolved_model,
                    "requested_slot": requested_slot,
                    "resolved_model": resolved_model,
                    "account_type": route.account_type,
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
                retryable=True,
            )
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
        failure_ledger: list[dict] = []
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
                    failure_ledger=tuple(failure_ledger),
                )
            except ProviderPolicyHardStop as exc:
                last_error = exc
                if not exc.retryable:
                    raise
                fallback_reason = f"provider_{exc.kind}"
                plan = self.explain(provider=provider, model=model)
                failure_ledger.append(
                    {
                        "requested_slot": plan["requested_slot"],
                        "provider": provider,
                        "resolved_model": plan["resolved_model"],
                        "account_type": plan["account_type"],
                        "reason": fallback_reason,
                        "retryable": True,
                    }
                )
                if len(failure_ledger) > FAILURE_LEDGER_MAX:
                    failure_ledger = failure_ledger[-FAILURE_LEDGER_MAX:]
        if last_error is not None:
            raise last_error
        raise ProviderPolicyHardStop("provider_no_candidates", kind="policy")
