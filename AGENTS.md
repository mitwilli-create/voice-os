# AGENTS.md - voice-os

Read `~/Documents/mission-control/WORKSPACE.md` first: it defines the multi-agent lane rules for this machine. Your lane here (Codex) is building; Claude Code reviews your output and owns orchestration/memory. CodeRabbit reviews commits and PRs automatically.

Then read the `HANDOFF-*.md` files at the repo root: they carry current project state, measured baselines, and the next work item (the reflection engine, design complete, implementation not started).

## What this repo is

Mitchell's voice operating system: scores and calibrates drafts against his actual writing voice on six axes (rhetorical pace, risk tolerance, sentence rhythm, escalation pattern, hedging, editorial register), grounded in a 143k-chunk provenance-tagged personal corpus. A LangGraph dual-persona pipeline (generative persona drafts with real exemplars, adversarial persona stress-tests, corpus-calibrated QA gate decides pass/revise/reject) is callable from other agents via `python3 -m voice_os draft`.

## Hard constraints

- **The corpus is deeply personal data. Never commit:** `corpus/`, `sources/` (21 GB raw email/iMessage/Instagram/Facebook exports), `var/` (checkpoints and eval state contain draft + exemplar text), `ingest.local.json`, `knowledge_base/`, any non-fixture `*.json`/`*.jsonl`. Git hooks enforce this (`git config core.hooksPath .githooks`); do not bypass them.
- **Determinism lock:** the golden-file tests must pass byte-identical every run; the pre-push hook re-runs them. Frozen fixtures in `tests/fixtures/` and `data/` are the only corpus tests may touch.
- **The live baseline moves only explicitly:** `var/eval/baseline.json` (on-disk state under gitignored `var/`, never committed) gates measured changes and is updated solely via `python3 -m voice_os.harness gate --update-baseline`, never as a side effect.
- **Honest measurement discipline:** `alignment_judged` (LLM-judged, live) is the truth metric; `alignment_offline` only gates regressions and is never reported as an approved number.
- **No em dashes anywhere in this repo.** Banned outright.
- **Privacy switch:** live mode sends draft text + exemplars to the Anthropic API, and live runs spend real money. Develop and test with `VOICE_OS_OFFLINE=1`; never trigger live runs (harness `run`, live `draft`) on your own initiative; Mitchell asks for them.
- **Process:** design-first PRs. The merge gate is the Qodo review loop to Bugs(0); CodeRabbit's automatic PR review also runs, but Qodo is the blocker.

## Commands

- Tests (run before declaring done): `python -m pytest tests/` (~300 tests; core runs dependency-free)
- Score a draft: `python score.py --corpus data/sample_corpus.txt --draft data/sample_draft.txt`
- Full pipeline: `python pipeline.py --corpus ... --banned-list data/banned_list.txt --draft ... --output output/scored_draft.json`
- Callable layer: `python3 -m voice_os draft --channel email --audience boss --situation high_stakes` (JSON envelope; exit 0 pass, 1 reject, 2 error)
- Live eval harness: `python3 -m voice_os.harness run` · ingest: `python3 -m ingest status` / `run`

## Conventions

- Python 3.10+ (3.11 in use). Core scoring is stdlib-only; `anthropic` and `langgraph` are optional extras that load lazily and stay quarantined in `voice_os/product/`.
- Layout: `voice_os/` core library, `voice_os/product/` callable layer, `voice_os/harness/` eval, `ingest/` + `mine/` corpus pipeline, `docs/` design docs (15, read before touching an area).
- Ingestion is incremental and deduped by content hash; the held-in/held-out split is a seeded hash. Don't disturb either invariant.

---

<!-- BEGIN STANDING-RULES (Mitchell global, installed 2026-07-18) -->
## Standing rules (global)

These apply to any Claude instance working in this repo, including off-machine (CI, collaborators, cloud agents):

1. **Freshness re-anchor.** Before acting on the first input of a session, and again after any gap over ~3 hours, web-search to confirm the current Pacific date/time (PST/PDT-aware) and scan the task topic for anything that changed since your knowledge cutoff, before relying on training-data recall. Re-check any pending "today/tomorrow" commitment against the confirmed date.
2. **Stack-search before building.** At the start of any new build / feature / reusable tool, first research what already exists (X, Reddit, Hacker News, Discord, dev forums, package registries) for highly-rated, peer-recommended solutions. Report BUILD-vs-ADOPT with sources; bias to ADOPT over BUILD unless there is a real, audience-worthy gap. Build for an audience, not just yourself.
<!-- END STANDING-RULES -->
