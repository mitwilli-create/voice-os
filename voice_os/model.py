"""VoiceModel: the queryable facade over the extended voice model.

Loads the corpus baseline, the hand-curated banned list, mined artifacts,
and (optionally) the chunk store, then answers context queries with a
calibrated target profile, tone norms, merged banned list, exemplars, and
persona-ready guidance strings. Degrades gracefully: on a fresh clone with
no chunks or mined artifacts everything falls back to the hand tables and
the sources map says so. Offline-deterministic either way.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from . import __version__, load_corpus, run_cycles
from .axes import AxisProfile, score_text
from .calibration import calibrate_extended
from .contexts import VoiceContext
from .mined import MinedArtifacts, group_profile, load_artifacts
from .qa import GateResult, find_banned, gate_extended, load_banned_list
from .tone import ToneProfile, derive_metrics, tone_signals

# Exemplar selection: consider the most recent candidates only, so query
# stays fast on a large chunk store while remaining deterministic.
EXEMPLAR_CANDIDATES = 200
EXEMPLAR_K = 5


@dataclass
class QueryResult:
    """Everything the callable voice module needs for one context."""

    context: dict
    target_profile: dict[str, float]
    tone: ToneProfile | None
    banned: list[str]
    exemplars: list[dict]
    guidance: list[str]
    sources: dict[str, str]
    meta: dict = field(default_factory=dict)


class VoiceModel:
    def __init__(
        self,
        baseline: AxisProfile,
        banned: list[str],
        mined: MinedArtifacts,
        chunks_dir: str | None,
    ) -> None:
        self.baseline = baseline
        self.hand_banned = banned
        self.mined = mined
        self.chunks_dir = chunks_dir

    @classmethod
    def load(
        cls,
        corpus_path: str = os.path.join("corpus", "voice_corpus.txt"),
        *,
        chunks_dir: str | None = os.path.join("corpus", "chunks"),
        mined_dir: str | None = os.path.join("corpus", "mined"),
        banned_path: str | None = os.path.join("data", "banned_list.txt"),
    ) -> "VoiceModel":
        """Load a model. Only the corpus file is required; every other
        input is optional and its absence degrades to hand tables."""
        baseline = load_corpus(corpus_path)
        banned = (
            load_banned_list(banned_path)
            if banned_path and os.path.exists(banned_path)
            else []
        )
        mined = load_artifacts(mined_dir)
        if chunks_dir and not os.path.isdir(chunks_dir):
            chunks_dir = None
        return cls(baseline, banned, mined, chunks_dir)

    @property
    def banned(self) -> list[str]:
        """Hand list merged with mined n-grams, deduped, hand list first."""
        merged = list(self.hand_banned)
        seen = set(merged)
        for phrase in self.mined.ngram_banned:
            if phrase not in seen:
                merged.append(phrase)
                seen.add(phrase)
        return merged

    def query(self, **context_kwargs) -> QueryResult:
        """Resolve a communication context into generation targets."""
        ctx = VoiceContext(**context_kwargs)
        target, sources = calibrate_extended(self.baseline, ctx, mined=self.mined)

        tone, tone_source = self._tone_profile(ctx)
        sources["tone"] = tone_source

        guidance: list[str] = []
        if ctx.goal != "unknown":
            guidance.append(f"communication goal: {ctx.goal}")
        if ctx.stakes != "routine":
            guidance.append(f"stakes level: {ctx.stakes}")

        exemplars = self._exemplars(ctx, target)
        sources["exemplars"] = "mined" if exemplars else "absent"

        meta = {
            "voice_os_version": __version__,
            "artifacts": self.mined.meta,
        }
        if self.mined.drift_report:
            meta["drift_flags"] = self.mined.drift_report.get("flags", [])

        return QueryResult(
            context=ctx.as_dict(),
            target_profile=target,
            tone=tone,
            banned=self.banned,
            exemplars=exemplars,
            guidance=guidance,
            sources=sources,
            meta=meta,
        )

    def gate_draft(self, draft: str, q: QueryResult) -> GateResult:
        """Gate a draft against a query's target, banned list, and tone."""
        scores = score_text(draft)
        hits = find_banned(draft, q.banned)
        observed = derive_metrics(tone_signals(draft)) if q.tone else None
        return gate_extended(
            scores, self.baseline, q.target_profile, hits,
            tone_observed=observed, tone_profile=q.tone,
        )

    def run(self, draft: str, max_cycles: int = 2, **context_kwargs) -> dict:
        """Full dual-persona pipeline for a context. JSON-safe dict."""
        q = self.query(**context_kwargs)
        cycles, result, text, modes = run_cycles(
            self.baseline, q.target_profile, draft, q.banned, max_cycles,
            extra_signals=q.guidance, tone_profile=q.tone,
        )
        return {
            "meta": {
                **q.meta,
                "mode": "live" if "live" in modes else "offline",
            },
            "classification": q.context,
            "sources": q.sources,
            "baseline": {"mean": self.baseline.mean, "std": self.baseline.std},
            "target_profile": q.target_profile,
            "cycles": cycles,
            "final": {
                "decision": result.decision,
                "fidelity": result.fidelity,
                "output_text": text,
            },
        }

    def _tone_profile(self, ctx: VoiceContext) -> tuple[ToneProfile | None, str]:
        """Most specific mined tone norms for the context: pair, then
        audience, then medium, then goal, then the mined global."""
        profiles = self.mined.context_profiles
        if not profiles:
            return None, "absent"
        lookups = []
        if ctx.medium:
            lookups.append(("pairs", f"{ctx.audience}|{ctx.medium}"))
        lookups.append(("audiences", ctx.audience))
        if ctx.medium:
            lookups.append(("media", ctx.medium))
        if ctx.goal != "unknown":
            lookups.append(("goals", ctx.goal))
        for kind, key in lookups:
            profile = group_profile(profiles, kind, key)
            if profile:
                return (
                    ToneProfile(mean=profile["tone_mean"], std=profile["tone_std"]),
                    "mined",
                )
        global_profile = profiles.get("global", {})
        if global_profile.get("tone_mean"):
            return (
                ToneProfile(
                    mean=global_profile["tone_mean"],
                    std=global_profile["tone_std"],
                ),
                "mined",
            )
        return None, "absent"

    def _iter_chunks(self):
        """Local JSONL chunk iterator; voice_os stays independent of ingest.

        Malformed lines are skipped rather than raised: the chunk store is
        optional input to the facade, and one corrupt line must not break
        query() (graceful-degradation contract).
        """
        import glob

        for path in sorted(glob.glob(os.path.join(self.chunks_dir, "*.jsonl"))):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk, dict):
                        yield chunk

    def _exemplars(self, ctx: VoiceContext, target: dict[str, float]) -> list[dict]:
        """Up to EXEMPLAR_K held-in tier 1/2 chunks matching the context,
        ranked by fidelity to the target then recency.

        Scores only the most recent EXEMPLAR_CANDIDATES matching chunks, so
        selection is deterministic and fast on a large store. Exemplar text
        is personal data: it feeds live persona prompts and is never
        persisted by this module.
        """
        if not self.chunks_dir:
            return []
        import heapq

        from .holdout import is_holdout

        def matching():
            for chunk in self._iter_chunks():
                if chunk.get("tier") not in (1, 2):
                    continue
                if is_holdout(chunk["hash"]):
                    continue
                context = chunk.get("context", {})
                if context.get("audience") != ctx.audience:
                    continue
                if ctx.medium and context.get("medium") != ctx.medium:
                    continue
                if ctx.goal != "unknown" and context.get("goal") != ctx.goal:
                    continue
                timestamp = chunk.get("provenance", {}).get("timestamp") or ""
                yield timestamp, chunk

        # One streaming pass with a bounded heap: O(N log K) with K =
        # EXEMPLAR_CANDIDATES, no full-store sort or materialization.
        recent = heapq.nlargest(
            EXEMPLAR_CANDIDATES, matching(), key=lambda pair: pair[0]
        )
        target_profile = AxisProfile(mean=target, std=self.baseline.std)
        scored = []
        for timestamp, chunk in recent:
            fit, _ = target_profile.fidelity(score_text(chunk["text"]))
            scored.append((fit, timestamp, chunk))
        scored.sort(key=lambda triple: (triple[0], triple[1]), reverse=True)
        return [
            {"id": chunk["id"], "text": chunk["text"], "tier": chunk["tier"],
             "fit": fit}
            for fit, _, chunk in scored[:EXEMPLAR_K]
        ]
