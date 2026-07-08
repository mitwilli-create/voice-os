# voice-os

A personal voice operating system: six-axis voice scoring, register calibration mined from a personal corpus, dual-persona drafting, calibrated QA gates, and a measured evaluation harness. Built on Claude. `import voice_os; voice_os.draft(...)` returns a checkpointed, gate-approved draft in the author's voice.

The engine began as the executive "Voice DNA" RAG pipeline I built for the Office of Engineering Strategy (OES) inside a large engineering organization, a digital twin for a VP-level executive's communications calibrated on 6.9M+ words. That system is the provenance; this repo has since become the personal successor: the same architecture, recalibrated and extended on my own corpus (tens of thousands of provenance-tagged chunks spanning email, messages, social, professional documents, and on-camera transcripts).

---

## What it does

Ingests a personal voice corpus with full provenance, mines per-context style statistics from it, and builds a queryable voice model. Drafts route through a LangGraph state machine: a generative persona writes with real exemplars of the author's messages in context, an adversarial persona stress-tests the result, and a QA gate calibrated to what the author's own text actually scores decides pass, revise, or reject. Every run is checkpointed to SQLite and stamped with corpus, artifact, and model provenance.

---

## Why it matters

Most "voice matching" is prompt engineering with a few examples. This is a calibrated, measured system. The six axes catch what vibes-based prompting misses: rhetorical pace, risk tolerance, sentence rhythm, escalation pattern, hedging behavior, and editorial register. The banned layer combines a curated refusal list with statistically mined anti-patterns. And the evaluation harness drafts real held-out messages through the full pipeline and scores generated against real, so changes are measured, not vibed: the QA gate's thresholds themselves are calibrated per context cell to percentiles of what the author's real text scores.

---

## Quick Start

Requires Python 3.10 or newer.

```bash
# Clone the repo
git clone https://github.com/mitwilli-create/voice-os.git
cd voice-os

# Optional: enables the live Claude personas and the LangGraph layer.
# The pipeline runs deterministically offline without it.
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here

# Run a scoring pass against the sample corpus
python score.py --corpus data/sample_corpus.txt --draft data/sample_draft.txt

# Run the full dual-persona pipeline with QA gate
python pipeline.py \
  --corpus data/sample_corpus.txt \
  --banned-list data/banned_list.txt \
  --draft data/sample_draft.txt \
  --output output/scored_draft.json

# Or call the shipped product layer directly
python -c "
import voice_os
result = voice_os.draft(
    'quick note to say the launch plan looks right, one timing question',
    channel='email', audience='boss', situation='high_stakes',
    goal='set_expectations',
)
print(result['decision'], result['output_text'])
"
```

Output includes axis scores, the register-calibrated target profile, QA gate decision, and a full revision trace per cycle. Context flags (`--channel`, `--audience`, `--situation`) select the register calibration applied to the target; the callable layer accepts natural aliases ("boss" maps to leadership, "high_stakes" to stakes high).

---

## Architecture

| Layer | Function | Module |
|---|---|---|
| Source ingestion | Extracts, deduplicates, and provenance-tags local data exports (email, Instagram, Facebook, iMessage, documents by doc_type, video transcripts with pacing); renders the scoring corpus. Incremental by design. See [docs/ingestion.md](docs/ingestion.md) and [docs/doc-types.md](docs/doc-types.md) | `ingest/` |
| Corpus baseline | Parses dated corpus entries; builds a temporal-tier-weighted axis baseline | `voice_os/corpus.py` |
| Six-axis scorer | Evaluates drafts against the baseline across six stylistic axes | `voice_os/axes.py` |
| Register calibration | Channel x audience x situation x goal x stakes deltas produce the generation target; mined per-context profiles blend over hand tables. See [docs/extended-model.md](docs/extended-model.md) | `voice_os/calibration.py`, `voice_os/contexts.py`, `voice_os/tone.py` |
| Mining layer | Statistically mines recipient deltas, per-context tone norms, n-gram anti-patterns, drift flags, and per-cell gate calibration from the chunk store (held-in split only) | `mine/`, loaded by `voice_os/mined.py` |
| Voice model facade | One queryable object: context in, calibrated target + tone norms + merged banned list + ranked real exemplars + guidance out | `voice_os/model.py` |
| Dual-persona router | Generative persona revises with exemplars, KB voice patterns, and a length budget; adversarial persona stress-tests voice match | `voice_os/personas.py` |
| QA gate | Blocks output below a per-cell calibrated threshold; returns structured revision signals. See [docs/live-alignment.md](docs/live-alignment.md) | `voice_os/qa.py` |
| Callable product layer | `voice_os.draft()`: a LangGraph generate, critique, gate, revise state machine with SQLite-checkpointed runs, KB snapshotting, and machine-neutral provenance. LangGraph loads lazily; `import voice_os` stays stdlib-only. See [docs/callable-layer.md](docs/callable-layer.md) | `voice_os/product/` |
| Evolution tracking | Stored baselines and insights on how the voice shifts over time | `voice_os/evolution/` |
| Evaluation harness | Drafts real held-out messages through the pipeline, scores generated vs real (style, similarity, live LLM judge), gates regressions | `voice_os/harness/`, `voice_os/eval.py` |

### The canonical six axes

One canonical axis set is used everywhere (scoring baseline, calibration target, QA gate): **rhetorical pace, risk tolerance, sentence rhythm, escalation pattern, hedging behavior, editorial register.** An earlier iteration of this system (documented in `docs/architecture.md`) expressed register calibration on a second dimension set; those are now re-expressed as deltas on the canonical axes. The mapping lives in [`voice_os/axes.py`](voice_os/axes.py).

---

## What this demonstrates

- **Production RAG design.** The predecessor ran at executive scale inside a large engineering organization; this successor runs on a personal corpus with the same discipline.
- **Evaluation rigor.** A harness that drafts real held-out messages and measures alignment offline and with a live LLM judge, with an explicitly governed regression baseline. Offline numbers gate; judged numbers are never gated, only reported.
- **Agentic architecture.** A LangGraph state machine with conditional routing, checkpointed runs, and a bounded revision loop, contained behind a stdlib-only public API.
- **Data engineering.** Provenance-tagged incremental ingestion across seven source types, a held-in/held-out split keyed on content hashes, and mined artifacts with versioned envelopes.
- **Domain depth.** A decade in newsrooms and eight years inside a large engineering organization built the editorial judgment that makes the scoring axes meaningful.

---

## Status

Shipped and measured: the callable layer (`voice_os.draft()`), the extended context model (audience, medium, goal, stakes, tone), mined per-context calibration including data-driven QA-gate thresholds, exemplar and KB fusion into live persona prompts, an evaluation harness with a locked regression baseline, and provenance-stamped, SQLite-checkpointed runs.

The personal corpus itself (chunk stores under `corpus/`, raw exports under `sources/`, mined artifacts, KB snapshots) is local-only and gitignored; layered privacy gates (gitignore, pre-commit and pre-push hooks) keep personal data out of the repo. Sample data in `data/` is synthetic but structurally representative, and the whole test suite runs against it.

Every stage has a deterministic offline implementation, so scoring and gating are reproducible without an API key; with credentials, the generative and adversarial personas run on Claude.

**Privacy note:** in live mode the draft text, target profile, banned phrases, revision signals, selected real exemplar messages (bounded, held-in only), and distilled KB voice patterns are sent to the Anthropic API. Set `VOICE_OS_OFFLINE=1` to force offline mode for sensitive drafts even when credentials are present. Checkpoints contain draft and exemplar text and live under the gitignored `var/` directory.

Tests: `python -m pytest tests/` runs the full suite (offline, no API key needed). The core scoring tests also run dependency-free via `python -m unittest discover -s tests -v`.

Evaluation harness + regression gate: `python3 -m voice_os.harness run`
drafts real held-out Tier 1 messages through the full pipeline and
scores generated vs real (embedding similarity, paired style fidelity
on the six axes, LLM judge in live mode), reporting per-channel and
per-audience fidelity. The gate blocks modeling and pipeline changes
that regress the deterministic offline numbers, in three layers:
the fixture-corpus golden lock runs inside the standard pytest suite;
`.githooks/pre-push` re-runs it (plus the real-corpus gate when a
local baseline exists) on every push from a clone that has enabled
hooks (`git config core.hooksPath .githooks`); and the baseline moves
only by explicit `python3 -m voice_os.harness gate --update-baseline`.
Design and honesty constraints: `docs/eval-harness.md`; measured live
alignment history: `docs/live-alignment.md`.

---

## Built with

- Claude (Anthropic): generation, adversarial critique, live judging
- Python, LangGraph (contained in `voice_os/product/`)
- Custom scoring, mining, and evaluation layers

---

Mitchell Williams · [LinkedIn](https://linkedin.com/in/mitwilli) · [GitHub](https://github.com/mitwilli-create) · [thestorytellermitch.com](https://thestorytellermitch.com)
