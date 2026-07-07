# voice-os

Six-axis voice scoring. Dual-persona routing. QA gates that hold drafts in-voice before they ever reach a human. Built on Claude. Calibrated on 6.9M+ words of executive communication.

This is the engine behind the executive "Voice DNA" RAG pipeline I built for the Office of Engineering Strategy (OES) inside a large engineering organization: a digital twin for a VP-level executive's communications. Every draft is scored against the 6.9M+ word calibration corpus across six stylistic axes, and nothing ships until it clears the QA gate.

---

## What it does

Ingests a voice corpus, builds a scored representation across six stylistic axes, and routes drafts through dual personas (generative + adversarial) before a QA gate decides whether output clears or cycles back. The result is drafts that sound like the person they're supposed to sound like, not a generic LLM.

---

## Why it matters

Most "voice matching" is prompt engineering with a few examples. This is a calibrated scoring system. The six axes catch what vibes-based prompting misses: rhetorical pace, risk tolerance, sentence rhythm, escalation pattern, hedging behavior, and editorial register. The banned-phrase checklist, a curated set of rejected drafts, teaches the system what the voice refuses to do, not just what it does.

---

## Quick Start

Requires Python 3.10 or newer.

```bash
# Clone the repo
git clone https://github.com/mitwilli-create/voice-os.git
cd voice-os

# Optional: enables the live Claude personas. The pipeline runs
# deterministically offline without it (no API key required).
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

# Same pipeline on a draft that is already in-voice (gate returns "pass")
python pipeline.py \
  --corpus data/sample_corpus.txt \
  --banned-list data/banned_list.txt \
  --draft data/sample_draft_good.txt
```

Output includes axis scores, the register-calibrated target profile, QA gate decision (pass / cycle), and a full revision trace per cycle. Exit code 0 means the gate passed; 1 means the draft still needs a cycle. Context flags (`--channel`, `--audience`, `--situation`) select the register calibration applied to the target.

---

## Architecture

| Layer | Function | Module |
|---|---|---|
| Source ingestion | Extracts, deduplicates, and provenance-tags local data exports; renders the scoring corpus. Incremental by design. See [docs/ingestion.md](docs/ingestion.md) | `ingest/` |
| Corpus ingestion | Parses dated corpus entries; builds a temporal-tier-weighted axis baseline | `voice_os/corpus.py` |
| Six-axis scorer | Evaluates drafts against the baseline across six stylistic axes | `voice_os/axes.py` |
| Register calibration | Channel x audience x situation deltas produce the generation target | `voice_os/calibration.py` |
| Dual-persona router | Generative persona revises; adversarial persona stress-tests voice match | `voice_os/personas.py` |
| QA gate | Blocks output below threshold; returns structured revision signal | `voice_os/qa.py` |
| Banned-phrase enforcement | Flags patterns the voice explicitly rejects | `voice_os/qa.py` |

### The canonical six axes

One canonical axis set is used everywhere (scoring baseline, calibration target, QA gate): **rhetorical pace, risk tolerance, sentence rhythm, escalation pattern, hedging behavior, editorial register.** An earlier iteration of this system (documented in `docs/architecture.md`) expressed register calibration on a second dimension set (directness, structure, warmth, formality, precision, assertiveness); those are now re-expressed as deltas on the canonical axes. The mapping lives in [`voice_os/axes.py`](voice_os/axes.py).

The `voice_os` package is importable directly (`from voice_os import run_pipeline, score_draft`), which is the foundation for the callable voice module and MCP interface on the roadmap.

---

## What this demonstrates

- **Production RAG design.** Not a demo. A system that ran at executive scale inside a large engineering organization.
- **Evaluation rigor.** Quantified six-axis voice scoring, not vibes.
- **Agentic architecture.** Multi-step pipeline with conditional routing and gate logic.
- **Domain depth.** A decade in newsrooms and eight years inside a large engineering organization built the editorial judgment that makes the scoring axes meaningful.

---

## Status

The pipeline (`voice_os/` package, `score.py`, `pipeline.py`) runs end to end against the synthetic sample data in `data/`. Core corpus and VP-identity data are not included. That's proprietary. Sample data is synthetic but structurally representative.

Every stage has a deterministic offline implementation, so scoring and gating are reproducible without an API key; with credentials, the generative and adversarial personas run on Claude.

**Privacy note:** in live mode the draft text, target profile, banned phrases, and revision signals are sent to the Anthropic API. Set `VOICE_OS_OFFLINE=1` to force offline mode for sensitive drafts even when credentials are present. Corpus text itself is never sent; only its computed axis profile is.

Tests: `python -m pytest tests/` runs the full suite, scoring plus ingestion (offline, no API key needed). The core scoring tests also run dependency-free via `python -m unittest discover -s tests -v`.

CI/evaluation harness: in progress.

---

## Built with

- Claude (Anthropic): generation and adversarial persona
- Python
- Custom embedding + scoring layer

---

Mitchell Williams · [LinkedIn](https://linkedin.com/in/mitwilli) · [GitHub](https://github.com/mitwilli-create) · [thestorytellermitch.com](https://thestorytellermitch.com)
