# Determinism Contract

Voice OS treats offline determinism as a load-bearing property, not a
nice-to-have: the same inputs must produce byte-identical outputs on
every offline surface, so results are reproducible, regressions are
diffable, and tests need no tolerance bands. This document states what
is guaranteed today, the audit that verifies it, and the hardening
items scheduled before the evolution/drift module lands.

## What is guaranteed today

- **Offline core.** Axis scoring (`voice_os/axes.py`), tone metrics,
  register calibration, the QA gate, and the offline personas are pure
  functions of their inputs: regex heuristics and arithmetic, no RNG,
  no wall-clock in the outputs.
- **Offline graph runs.** With `VOICE_OS_OFFLINE=1`, a
  `voice_os.draft()` call is deterministic given the same input text,
  context, corpus, mined artifacts, and KB content. Only `run_id`,
  snapshot ids, and checkpoint timestamps vary; every scoring,
  decision, and text field is reproducible.
- **Content-addressed KB versioning.** KB snapshots are identified by
  a sha256 bundle hash; identical content never re-snapshots, changed
  content always does (`voice_os/product/kb.py`).
- **Checkpointed calibration.** The full calibration inputs (baseline
  mean/std, target profile, banned list, tone norms) are serialized
  into every run's state, so a checkpoint is self-describing without
  re-reading the corpus.
- **Stable iteration.** File globs are sorted, hash concatenation is
  name-sorted, exemplar selection is a bounded deterministic heap.
- **Golden regression.** `run_pipeline`'s default output is locked by
  golden tests; the test suite runs offline with synthetic fixtures.

## What live mode guarantees instead

Live persona calls (Claude API) are inherently non-deterministic. The
contract there is reproducibility-in-inputs: every live run must be
auditable to the exact inputs and engine that produced it. See
hardening item 3.

## Audit procedure (run before and after determinism-relevant PRs)

Double-run byte-equality census: execute each surface twice with
identical inputs and diff the outputs, excluding only fields documented
as run-scoped (`run_id`, snapshot ids, timestamps).

Surfaces: offline `draft()` envelope, `score_text`,
`calibrate_extended`, `gate_extended`, mined artifacts regenerated from
a fixture corpus, the eval scorecard, `load_kb` bundle hashes. Any
diff outside the documented run-scoped fields is a determinism
regression.

## Scheduled hardening (Phase 0 of the evolution-module sequence)

1. **Golden-regression lock on the offline `draft()` envelope.**
   `run_pipeline` has one; the graph path does not yet. Lock the full
   envelope minus run-scoped fields on the synthetic fixture corpus.
2. **Provenance in the envelope and checkpoints.** `QueryResult.meta`
   carries the voice_os version and mined-artifact versions, but the
   prepare node does not copy it into state. Add a serializable
   provenance field: voice_os version, mined artifact versions, KB
   bundle hash, corpus file identity.
3. **Live-run engine stamping.** `llm.complete` sends no explicit
   temperature and nothing records which model id served a live run.
   Record the resolved model id into provenance whenever any persona
   reports mode "live". Default to recording generation parameters,
   not changing them; pinning temperature is a separate decision with
   a stability-vs-quality tradeoff.
4. **Standing determinism invariant test.** Run the offline pure
   surfaces twice in-process and assert byte-identical results, so a
   future change cannot silently introduce wall-clock, dict-order, or
   RNG dependence.

## Rules for new modules

Every new module (evolution queries, drift baselines, insight
generation) inherits this contract from its first commit:

- RNG-free, or seeded with the seed recorded in outputs.
- Sorted iteration over any filesystem or dict-derived collection.
- Stored baselines and artifacts content-hashed, mirroring the KB
  snapshot scheme.
- Covered by the determinism invariant test before merge.
- Timestamps and generated ids allowed only in fields documented as
  run-scoped.
