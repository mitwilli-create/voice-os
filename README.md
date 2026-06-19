# voice-os

Six-axis voice scoring. Dual-persona routing. QA gates that enforce stylistic fidelity before a draft ever reaches a human. Built on Claude. Calibrated on 6.9M+ words of executive communication.

This is the engine behind the executive "Voice DNA" RAG pipeline I built for the Office of Engineering Strategy (OES) inside a large engineering organization: a digital twin for a VP-level executive's communications. It cut executive drafting time substantially while holding high stylistic fidelity across production volume.

---

## What it does

Ingests a voice corpus, builds a scored representation across six stylistic axes, and routes drafts through dual personas (generative + adversarial) before a QA gate decides whether output clears or cycles back. The result is drafts that sound like the person they're supposed to sound like, not a generic LLM.

---

## Why it matters

Most "voice matching" is prompt engineering with a few examples. This is a calibrated scoring system. The six axes catch what vibes-based prompting misses: rhetorical pace, risk tolerance, sentence rhythm, escalation pattern, hedging behavior, and editorial register. The banned-phrase checklist, a curated set of rejected drafts, teaches the system what the voice refuses to do, not just what it does.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/mitwilli-create/voice-os.git
cd voice-os

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=your_key_here

# Run a scoring pass against the sample corpus
python score.py --corpus data/sample_corpus.txt --draft data/sample_draft.txt

# Run the full dual-persona pipeline with QA gate
python pipeline.py \
  --corpus data/sample_corpus.txt \
  --banned-list data/banned_list.txt \
  --draft data/sample_draft.txt \
  --output output/scored_draft.json
```

Output includes axis scores, persona deltas, QA gate decision (pass / cycle), and a revision trace.

---

## Architecture

| Layer | Function |
|---|---|
| Corpus ingestion | Chunks and embeds voice corpus; builds axis score baseline |
| Six-axis scorer | Evaluates drafts against baseline across six stylistic dimensions |
| Dual-persona router | Generative persona drafts; adversarial persona stress-tests fidelity |
| QA gate | Blocks output below threshold; returns structured revision signal |
| Banned-phrase enforcement | Flags patterns the voice explicitly rejects |

---

## What this demonstrates

- **Production RAG design.** Not a demo. A system that ran at executive scale inside a large engineering organization.
- **Evaluation rigor.** Quantified fidelity scoring, not vibes.
- **Agentic architecture.** Multi-step pipeline with conditional routing and gate logic.
- **Domain depth.** A decade in newsrooms and eight years inside a large engineering organization built the editorial judgment that makes the scoring axes meaningful.

---

## Status

Pipeline architecture and scoring logic are documented here. Core corpus and VP-identity data are not included. That's proprietary. Sample data is synthetic but structurally representative.

CI/evaluation harness: in progress.

---

## Built with

- Claude (Anthropic): generation and adversarial persona
- Python
- Custom embedding + scoring layer

---

Mitchell Williams · [LinkedIn](https://linkedin.com/in/mitwilli) · [GitHub](https://github.com/mitwilli-create) · [thestorytellermitch.com](https://thestorytellermitch.com)
