# Voice Evolution Module (design)

Prompt 5 of the extended-model sequence: a dedicated evolution tracking
and insight surfacing module on top of the existing 4-tier temporal
corpus structure. This document is the design gate for the
implementation PR, following the design-first convention from the
extended model and the callable layer.

Capabilities (mission, restated):

1. Queryable historical voice evolution across the four tiers.
2. Automated voice-drift detection: periodic re-extraction of pattern
   profiles on new Tier 1 data, diffed against stored baselines
   (greeting distribution, sentence length, marker frequencies).
3. Insight generation: non-obvious patterns in how the voice shifts by
   audience, medium, era, or goal.
4. Integration with the banned/never-use layer: flag emerging and
   fading patterns.
5. The drift workflow as a small LangGraph graph with state
   persistence, so runs can be scheduled, reviewed, and compared.

## Composition with what exists (extend, do not rebuild)

- **Axis-level drift is done.** `voice_os/drift.py` (pure math) +
  `mine/drift.py` (batch) already produce windowed axis profiles,
  sustained-shift flags, and lexical marker crossovers into
  `corpus/mined/drift_report.json`; the callable layer surfaces those
  flags in trace notes. This module adds PATTERN-level drift
  (greetings, sign-offs, sentence-length shape, marker frequencies)
  with STORED baselines and run-over-run diffs. Nothing in the axis
  layer changes.
- **Mining conventions carry over.** New extraction runs as batch
  tooling over the chunk store, trains on the held-in split only
  (`mine/weights.py::is_train`), and writes envelope-wrapped JSON
  artifacts (`mine/weights.py::envelope`).
- **The banned merge point stays singular.** `VoiceModel.banned` merges
  the hand list with mined n-grams. Evolution flags feed a SIBLING
  artifact (`evolution_flags.json`) surfaced through `QueryResult.meta`
  exactly like `drift_flags`; nothing is ever auto-applied to the
  banned list. Proposals only, mirroring
  `voice_os/drift.py::suggest_boundaries`.
- **The graph pattern mirrors the callable layer.** Guarded langgraph
  import containment, serializable state, SqliteSaver under the
  gitignored `var/`, run-history projection from trace notes. The
  drift graph gets its OWN database file (`var/evolution.sqlite`) so
  product runs and drift runs never share a checkpoint namespace.
- **Determinism contract inherited from first commit**
  (docs/determinism.md, Rules for new modules): RNG-free, sorted
  iteration, content-hashed stored baselines, run-scoped fields
  documented, every pure surface added to the double-run section of
  `tests/test_determinism.py`.

## Package layout

    voice_os/evolution/
        __init__.py     public API; stdlib-only import
        patterns.py     pure pattern extraction math (stdlib)
        timeline.py     pure evolution queries over the chunk store (stdlib)
        baselines.py    content-hashed baseline store under var/ (stdlib)
        insights.py     pure insight generation (stdlib)
        graph.py        the ONLY module importing langgraph
        __main__.py     CLI: timeline / baseline / drift-run / runs / insights

`voice_os/__init__.py` lazily exposes the public functions via the same
PEP 562 mechanism as the product layer, so `import voice_os` stays
stdlib-only and langgraph is touched only when `drift_run` /
`drift_run_history` execute.

## Pattern profile (the unit of extraction and diffing)

`patterns.py::extract_pattern_profile(records)` consumes an iterable of
`(timestamp, text, context)` records and returns a JSON-safe dict:

- `greetings`: distribution over a FIXED greeting lexicon (hey, hi,
  hello, yo, hiya, good morning, ...) matched against the first line;
  anything else buckets to `"other"`. Matching against a fixed lexicon
  keeps names and personal content out of stored artifacts.
- `signoffs`: same treatment for a fixed sign-off lexicon (thanks,
  best, cheers, talk soon, ...) over the last line.
- `sentence_length`: mean / p50 / p90 words per sentence, sentence
  count, plus short-message rate (share of chunks under 10 words).
- `markers`: per-100-words frequencies for a configurable marker list;
  the default includes the axis-drift marker forms (yea, yeah, gonna,
  going to) plus common intensity markers (lol, haha, exclamation
  rate).
- `n`: chunk and word counts backing the profile.

All floats rounded to 4 places (repo convention), all dict iteration
sorted. Pure function; no I/O.

## Capability 1: queryable evolution timeline

`timeline.py::evolution_timeline(chunks_dir=..., group_by="window",
slice_by=None)`:

- `group_by`: `"tier"` (the four tiers), `"window"` (half-years,
  reusing `voice_os/drift.py::window_key`), or `"year"`.
- `slice_by`: optional filters on chunk context (audience, medium,
  goal), so "how did my leadership-email voice change" is one call.
- Each group returns `n_chunks`, axis means (via the existing
  `score_text`), and the pattern profile above.
- Uses ALL dated chunks including tiers 3 and 4, unweighted, matching
  the mine/drift.py rationale: temporal analysis needs the history the
  tier weights would erase. Tier 4 remains zero-weight for GENERATION;
  this surface is analysis only.
- Public as `voice_os.evolution_timeline(...)`; stdlib-only.

## Capability 2: pattern drift detection with stored baselines

**Baseline store** (`baselines.py`), mirroring the KB snapshot scheme:

- A baseline is the pattern profile of the current Tier 1 (and
  optionally Tier 2) train-split data plus the windowed profile
  series, stored under `var/evolution/baselines/<baseline_id>/` with a
  `manifest.json` carrying a `content_hash` (sha256 of the canonical
  JSON of the profile body).
- `ensure_baseline()` is content-addressed exactly like
  `kb.ensure_snapshot`: identical content never re-snapshots, changed
  content always does. `baseline_id` is a timestamped directory name
  and is run-scoped; `content_hash` is the stable identity.
- Personal data (greeting/marker distributions derived from private
  text) stays under the gitignored `var/`.
- All evolution persistence (baseline store AND the drift checkpoint
  database) resolves through the same var-dir convention as the
  callable layer: explicit `var_dir` argument, then the
  `VOICE_OS_VAR_DIR` environment override, then the repo-root-anchored
  default, so scheduled runs and non-repo-root invocations control
  where personal-data artifacts land.

**Drift check** (`patterns.py::diff_profiles(baseline, current)`), pure:

- Per-greeting / per-signoff / per-marker deltas with a
  noise floor (minimum support count and minimum relative change,
  constants documented in-module).
- `emerging`: present in current, absent or near-zero in baseline.
- `fading`: present in baseline, near-zero in current.
- Sentence-length shift when the current mean departs the baseline
  mean by more than a documented tolerance.
- Output is a JSON-safe report; deterministic given the same inputs.

## Capability 3: insight generation

`insights.py::generate_insights(timeline_groups, diffs=None)`, pure and
deterministic (no LLM):

- Contrasts each audience / medium / goal slice against the global
  profile and each era against its predecessor; ranks findings by
  normalized effect size; keeps the top K above a documented
  threshold.
- Emits human-readable strings with the numbers inline, the
  `suggest_boundaries` style: "greeting 'hey' displaced 'hi' in
  leadership email starting 2024H2 (0.8/100w vs 2.1/100w two windows
  earlier)".
- Surfaced by `voice_os.voice_insights(...)` and included in each
  drift run's report.

## Capability 4: banned/never-use layer integration

- A drift run writes `corpus/mined/evolution_flags.json` (envelope
  schema, `artifact: "evolution_flags"`), carrying the emerging /
  fading pattern lists and the run id that produced them.
- `voice_os/mined.py` gains the artifact in `ARTIFACT_FILES`;
  `VoiceModel.query` surfaces the flags in `QueryResult.meta`
  ("evolution_flags"), exactly parallel to `drift_flags`. The product
  layer's prepare node today forwards only `drift_flags` into trace
  notes; the implementation PR extends prepare() with the same
  forwarding for `evolution_flags` (a two-line addition mirroring the
  drift_flags block).
- NOTHING auto-applies: fading phrases are candidates for the hand
  banned list (`data/banned_list.txt`), emerging phrases are
  candidates for `data/never_ban.txt` protection; both remain explicit
  human edits, consistent with the tier-boundary policy in
  `voice_os/drift.py`.

## Capability 5: the drift-run graph

Small LangGraph StateGraph in `graph.py`, mirroring
`voice_os/product/graph.py` conventions:

    START -> extract -> compare -> flag -> record -> END

- `extract`: pattern profile + windowed series from the chunk store.
- `compare`: `ensure_baseline` + `diff_profiles` against the newest
  stored baseline (first run: the extraction becomes the baseline and
  the diff is empty).
- `flag`: emerging/fading classification + insight generation; writes
  `evolution_flags.json`.
- `record`: summary trace note; the envelope is projected from final
  state.
- `DriftState` TypedDict: serializable primitives only; append-only
  `trace_notes` reducer; paths (chunks_dir, var_dir, mined_dir) in
  state so checkpoints are self-describing.
- Persistence: SqliteSaver over `<var_dir>/evolution.sqlite` (own
  file, own thread namespace `drift-<timestamp>-<hex>` run ids), with
  var_dir resolved through the convention above.
- `voice_os.drift_run(...)` executes one checkpointed run;
  `voice_os.drift_run_history(run_id=None)` lists prior runs (all
  thread ids when run_id is None) for review and comparison.
- Offline-deterministic by construction: no personas, no LLM calls
  anywhere in the drift path. Run-scoped fields: run id, baseline id,
  timestamps.

## Scheduling

No launchd plists live in this repo. The periodic run stays
CLI-invoked with a documented command:

    python3 -m voice_os.evolution drift-run

If Mitchell later schedules it, the job follows the career-ops launchd
conventions (macOS Tahoe wrapper caveats documented there), pointing at
that command.

## Testing

- Synthetic fixtures only (fictional Test Person text), matching the
  existing suites; graph tests `pytest.importorskip("langgraph")`.
- Every pure surface (extract_pattern_profile, evolution_timeline,
  diff_profiles, generate_insights, baseline content hashing) added to
  the double-run section of `tests/test_determinism.py` in the same PR
  that introduces it.
- Baseline store tests mirror the KB snapshot tests: content-addressed
  dedup, changed-content re-snapshot, manifest on disk under a temp
  var dir, repo tree stays clean.

## Out of scope (unchanged parkings)

- Fusing KB pattern data into persona prompts (parked in
  docs/callable-layer.md).
- Auto-editing any banned/never-ban list or tier boundary.
- LLM-generated insights; the insight layer is deterministic math by
  design so it can run in the scheduled path at zero cost.
