"""Generated-vs-real scoring: similarity, paired style, judge, composites.

Everything the regression gate consumes is deterministic and stdlib
only: lexical embedding similarity, paired style fidelity on the six
canonical axes, tone deltas, and safety counts. The LLM judge and the
optional semantic embedding backend are live-mode evidence layers with
honest labeling; their numbers never enter the gated composite.

Design: docs/eval-harness.md.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

from .. import llm
from ..axes import AXES, AxisProfile, score_text
from ..tone import TONE_METRICS, derive_metrics, tone_signals

EMBED_BACKEND_ENV = "VOICE_OS_EMBED_BACKEND"
EMBED_MODEL_ENV = "VOICE_OS_EMBED_MODEL"
DEFAULT_EMBED_MODEL = "voyage-3"

# Composite weights, one place, with the rationale from the design doc:
# the brief derives from the real message, so content similarity is
# partly by construction; paired style fidelity is the discriminating
# metric and carries the most weight. Changing these is a design
# decision, not a tweak (docs/eval-harness.md, Scoring).
OFFLINE_WEIGHTS = {"style": 0.60, "content": 0.25, "surface": 0.15}
JUDGED_WEIGHTS = {"style": 0.50, "content": 0.20, "judge": 0.30}

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before
    but by can could did do does for from had has have he her him his how
    i if in into is it its just me my no not of on or our out she so some
    than that the their them then there they this to up us was we were
    what when which who will with would you your
    """.split()
)


def _content_counts(text: str) -> Counter:
    return Counter(
        token
        for token in _WORD_RE.findall(text.lower())
        if token not in _STOPWORDS
    )


def _surface_counts(text: str) -> Counter:
    normalized = " ".join(text.lower().split())
    return Counter(
        normalized[i : i + 3] for i in range(max(len(normalized) - 2, 0))
    )


def cosine(a: dict, b: dict) -> float:
    """Cosine similarity of two sparse count vectors; 0.0 when either is empty."""
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(term, 0) for term, count in a.items())
    norm_a = math.sqrt(sum(count * count for count in a.values()))
    norm_b = math.sqrt(sum(count * count for count in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_similarity(real: str, generated: str) -> tuple[float, str]:
    """Optional live embedding backend; hard error, never silent fallback."""
    try:
        import voyageai
    except ImportError as exc:
        raise RuntimeError(
            f"{EMBED_BACKEND_ENV}=voyage requires the voyageai package: "
            "pip install voyageai"
        ) from exc
    model = os.environ.get(EMBED_MODEL_ENV, DEFAULT_EMBED_MODEL)
    client = voyageai.Client()
    result = client.embed([real, generated], model=model)
    vec_a, vec_b = result.embeddings
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm = math.sqrt(sum(x * x for x in vec_a)) * math.sqrt(
        sum(y * y for y in vec_b)
    )
    return (dot / norm if norm else 0.0), f"voyage:{model}"


def embed_similarity(real: str, generated: str) -> dict:
    """Similarity block: lexical always (the gated numbers), semantic
    only when VOICE_OS_EMBED_BACKEND=voyage is set."""
    block = {
        "content": round(
            cosine(_content_counts(real), _content_counts(generated)), 4
        ),
        "surface": round(
            cosine(_surface_counts(real), _surface_counts(generated)), 4
        ),
        "backend": "lexical",
    }
    if os.environ.get(EMBED_BACKEND_ENV) == "voyage":
        semantic, backend = _semantic_similarity(real, generated)
        block["semantic"] = round(semantic, 4)
        block["backend"] = f"lexical+{backend}"
    return block


def paired_style(real: str, generated: str, baseline_std: dict) -> dict:
    """Fidelity of the generated text TO THE REAL MESSAGE: the real
    message's axis scores are the target, the shared corpus std is the
    tolerance normalizer."""
    profile = AxisProfile(mean=score_text(real), std=dict(baseline_std))
    overall, per_axis = profile.fidelity(score_text(generated))
    return {"overall": overall, "per_axis": per_axis}


def tone_mae(real: str, generated: str) -> dict:
    observed_real = derive_metrics(tone_signals(real))
    observed_gen = derive_metrics(tone_signals(generated))
    return {
        metric: round(
            abs(observed_gen.get(metric, 0.0) - observed_real.get(metric, 0.0)), 4
        )
        for metric in TONE_METRICS
    }


# ------------------------------------------------------------------- judge

_JUDGE_SYSTEM = (
    "You are an impartial writing-style judge. Compare two texts on six "
    "stylistic dimensions and answer with strict JSON only: no prose, no "
    "markdown fences."
)


def _judge_prompt(real: str, generated: str) -> str:
    axis_lines = "\n".join(f'  "{axis}": <1-5>,' for axis in AXES)
    return (
        "Text A is a real message by an author. Text B attempts to write "
        "the same content in the same author's voice.\n\n"
        f"Text A:\n{real}\n\nText B:\n{generated}\n\n"
        "Rate 1-5 how closely Text B matches Text A on each dimension "
        "(5 = indistinguishable), and same_author 1-5 for how plausibly "
        "the author of A wrote B. Respond with exactly this JSON object:\n"
        "{\n" + axis_lines + '\n  "same_author": <1-5>\n}'
    )


def _clamp_rating(value) -> int | None:
    try:
        rating = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, rating))


def _parse_judge(raw: str) -> dict | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    axes = {}
    for axis in AXES:
        rating = _clamp_rating(parsed.get(axis))
        if rating is None:
            return None
        axes[axis] = rating
    same_author = _clamp_rating(parsed.get("same_author"))
    if same_author is None:
        return None
    return {"axes": axes, "same_author": same_author}


def _fidelity_rating(fidelity: float) -> int:
    return max(1, min(5, 1 + round(4 * fidelity)))


def judge_case(real: str, generated: str, style: dict, live: bool) -> dict:
    """LLM judge in live mode; deterministic fallback otherwise.

    The offline rating is DERIVED from the paired style fidelities and
    is never an independent opinion: mode makes that unmistakable.
    """
    if live:
        raw = llm.complete(_JUDGE_SYSTEM, _judge_prompt(real, generated))
        if raw:
            parsed = _parse_judge(raw)
            if parsed:
                return {**parsed, "mode": "live"}
    return {
        "axes": {
            axis: _fidelity_rating(style["per_axis"].get(axis, 0.0))
            for axis in AXES
        },
        "same_author": _fidelity_rating(style.get("overall", 0.0)),
        "mode": "offline",
    }


# -------------------------------------------------------------- case score


def score_case(case: dict, envelope: dict, baseline_std: dict, live: bool) -> dict:
    """One result record: generated (envelope) vs real (case)."""
    real = case["real_text"]
    generated = envelope.get("output_text", "")
    similarity = embed_similarity(real, generated)
    style = paired_style(real, generated, baseline_std)
    judge = judge_case(real, generated, style, live)

    alignment_offline = round(
        OFFLINE_WEIGHTS["style"] * style["overall"]
        + OFFLINE_WEIGHTS["content"] * similarity["content"]
        + OFFLINE_WEIGHTS["surface"] * similarity["surface"],
        4,
    )
    alignment_judged = None
    if judge["mode"] == "live":
        alignment_judged = round(
            JUDGED_WEIGHTS["style"] * style["overall"]
            + JUDGED_WEIGHTS["content"] * similarity["content"]
            + JUDGED_WEIGHTS["judge"] * (judge["same_author"] - 1) / 4,
            4,
        )

    real_words = len(real.split())
    generated_words = len(generated.split())
    return {
        "case_id": case["id"],
        "hash": case["hash"],
        "channel": case["channel"],
        "audience": case["audience"],
        "medium": case["medium"],
        "goal": case["goal"],
        "real_text": real,
        "brief": case["brief"],
        "output_text": generated,
        "inner_run_id": envelope.get("run_id"),
        "decision": envelope.get("decision"),
        "revisions": envelope.get("revisions", 0),
        "mode": envelope.get("mode", "offline"),
        "pipeline_fidelity": (envelope.get("fidelity") or {}).get("overall"),
        "provenance": dict(envelope.get("provenance") or {}),
        "similarity": similarity,
        "style": style,
        "tone_mae": tone_mae(real, generated),
        "judge": judge,
        "banned_hits": list(envelope.get("banned_hits") or []),
        "em_dash_hits": generated.count("\u2014"),
        "real_words": real_words,
        "generated_words": generated_words,
        "length_ratio": round(generated_words / real_words, 4) if real_words else None,
        "alignment_offline": alignment_offline,
        "alignment_judged": alignment_judged,
    }


# ------------------------------------------------------------- aggregation


def _mean(values: list[float]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return round(sum(cleaned) / len(cleaned), 4)


def _rate(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return round(sum(1 for flag in flags if flag) / len(flags), 4)


def _aggregate(records: list[dict]) -> dict:
    return {
        "n": len(records),
        "alignment_offline": _mean([r["alignment_offline"] for r in records]),
        "alignment_judged": _mean([r["alignment_judged"] for r in records]),
        "style_overall": _mean([r["style"]["overall"] for r in records]),
        "style_axes": {
            axis: _mean([r["style"]["per_axis"].get(axis) for r in records])
            for axis in AXES
        },
        "similarity_content": _mean(
            [r["similarity"]["content"] for r in records]
        ),
        "similarity_surface": _mean(
            [r["similarity"]["surface"] for r in records]
        ),
        "tone_mae": {
            metric: _mean([r["tone_mae"].get(metric) for r in records])
            for metric in TONE_METRICS
        },
        "judge_same_author": _mean(
            [r["judge"]["same_author"] for r in records]
        ),
        "pass_rate": _rate([r["decision"] == "pass" for r in records]),
        "banned_hit_rate": _rate([bool(r["banned_hits"]) for r in records]),
        "em_dash_rate": _rate([r["em_dash_hits"] > 0 for r in records]),
        "length_ratio": _mean([r["length_ratio"] for r in records]),
    }


def _modes(values: list[str]) -> str:
    unique = sorted(set(values))
    if not unique:
        return "none"
    return unique[0] if len(unique) == 1 else "mixed"


def summarize(results: list[dict]) -> dict:
    """Numbers-only summary: overall, per channel, per audience, per cell.

    Contains metric values, counts, and context labels; never message
    text. This is the shape the regression gate consumes and the only
    harness output safe to quote outside var/.
    """
    by_channel: dict[str, list[dict]] = {}
    by_audience: dict[str, list[dict]] = {}
    by_cell: dict[str, list[dict]] = {}
    for record in results:
        by_channel.setdefault(record["channel"], []).append(record)
        by_audience.setdefault(record["audience"], []).append(record)
        cell = f"{record['channel']}|{record['audience']}"
        by_cell.setdefault(cell, []).append(record)

    live_models = sorted(
        {
            record["provenance"].get("live_model")
            for record in results
            if record["provenance"].get("live_model")
        }
    )
    return {
        "schema_version": "1.0",
        "cases": len(results),
        "mode": {
            "personas": _modes([record["mode"] for record in results]),
            "judge": _modes([record["judge"]["mode"] for record in results]),
            "embed_backend": _modes(
                [record["similarity"]["backend"] for record in results]
            ),
        },
        "live_models": live_models,
        "weights": {"offline": OFFLINE_WEIGHTS, "judged": JUDGED_WEIGHTS},
        "overall": _aggregate(results),
        "by_channel": {
            key: _aggregate(group) for key, group in sorted(by_channel.items())
        },
        "by_audience": {
            key: _aggregate(group) for key, group in sorted(by_audience.items())
        },
        "by_cell": {
            key: _aggregate(group) for key, group in sorted(by_cell.items())
        },
    }
