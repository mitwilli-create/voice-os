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
- **Content-deduplicated KB versioning.** KB snapshots are
  de-duplicated by a sha256 bundle hash stored in each manifest:
  identical content never re-snapshots, changed content always does
  (`voice_os/product/kb.py`). The `snapshot_id` itself is a
  timestamped directory name and is run-scoped; the `bundle_hash` is
  the stable content identity to compare across runs.
- **Checkpointed calibration.** The full calibration inputs (baseline
  mean/std, target profile, banned list, tone norms) are serialized
  into every run's state, so a checkpoint is self-describing without
  re-reading the corpus.
- **Stable iteration.** File globs are sorted, hash concatenation is
  name-sorted, exemplar selection is a bounded deterministic heap.
- **Regression coverage today.** Two full golden snapshots
  (`tests/fixtures/golden/`) lock the complete output envelopes on the
  synthetic fixture corpus: `run_pipeline`'s default output and the
  offline `draft()` envelope, each minus the documented run-scoped
  fields. The earlier selected-field invariants in
  `tests/test_voice_os.py` and `tests/test_product.py` remain as
  behavior documentation. Regenerate after an intentional change with
  `python3 tests/regen_goldens.py` and review the golden diff in the
  PR.
- **Standing determinism invariant.** `tests/test_determinism.py` runs
  every offline pure surface twice in-process and asserts
  byte-identical canonical JSON: `score_text`, `calibrate_extended`,
  `gate_extended`, `run_pipeline`, all four miners on a synthetic
  chunk set (minus `generated_at`, the artifact envelope's run-scoped
  field), the eval scorecard, `load_kb`, and the normalized `draft()`
  envelope.
- **Provenance in every draft() envelope and checkpoint.** The prepare
  node seeds a serializable `provenance` field carried through
  checkpoints into the result envelope: voice_os version, mined
  artifact versions (`generated_at` + miner id per artifact), the KB
  bundle hash, and the corpus content identity as path + sha256 +
  byte size. Content hash rather than mtime is the documented
  identity choice: mtime differs across clones and copies of
  byte-identical content. The recorded path is machine-neutral
  (repo-relative for files under the repo, basename otherwise) so
  envelopes and checkpoints never leak absolute local paths, and the
  hash is memoized on the file's stat identity so repeated draft()
  calls do not re-read an unchanged corpus.
- **Live-run engine stamping.** The moment any persona reports mode
  "live", the resolved model id (`VOICE_OS_MODEL` /
  `llm.DEFAULT_MODEL`) is recorded into `provenance.live_model`;
  offline runs keep it `null`. `run_pipeline` and `VoiceModel.run`
  gain a `meta.model` stamp on live runs only, so their offline
  default output stays byte-identical to the golden lock. Generation
  parameters are recorded, not changed: no explicit temperature is
  pinned (stability-vs-persona-quality tradeoff, decided separately
  if ever).

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

## Hardening status (Phase 0 of the evolution-module sequence: SHIPPED)

All four scheduled items landed 2026-07-07 and are described in the
guarantees above:

1. Full golden-regression locks: `tests/fixtures/golden/` +
   `tests/test_determinism.py`, regenerated via
   `python3 tests/regen_goldens.py`.
2. Provenance in the envelope and checkpoints: seeded by the prepare
   node in `voice_os/product/graph.py`, projected by `build_result`.
3. Live-run engine stamping: `provenance.live_model` (graph) and
   `meta.model` (`run_pipeline` / `VoiceModel.run`), live runs only.
4. Standing determinism invariant test: the double-run section of
   `tests/test_determinism.py`.

The pre-hardening audit (double-run census, 2026-07-07) found all 12
surfaces byte-identical; the items above closed the coverage and
metadata gaps rather than live nondeterminism.

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
