# Generation Evaluation Harness + Regression Gate (design)

Prompt 6 of the sequence: move Voice OS from self-estimated alignment
numbers (the ~78-86% manual estimate in docs/architecture.md) to
reproducible evidence. This document is the design gate for the
implementation PR, following the design-first convention from the
extended model, the callable layer, and the evolution module.

Capabilities (mission, restated):

1. Hold out a set of real Tier 1 messages the current model never
   trained on.
2. For each held-out example, run the full pipeline to generate a
   reply in the same context.
3. Score generated vs real using embedding similarity plus an LLM
   judge on the core stylistic dimensions.
4. Report per-channel and per-audience fidelity metrics.
5. Make the harness the regression gate that runs automatically before
   any modeling or pipeline change is accepted.
6. Run the evaluation loop as a LangGraph graph with per-case
   checkpointing, result aggregation, and persistence.

## What this measures, honestly

The existing scorecard (`voice_os/eval.py`) asks: "how well do the
model's PREDICTED targets fit real held-out text?" It never runs the
generation pipeline. This harness asks the question the alignment
estimate was always about: "when the pipeline actually writes, how
close is the output to what Mitchell really wrote in that situation?"

Three honesty constraints shape the whole design:

- **The corpus stores only outgoing messages.** The inbound message
  that prompted each real reply was never ingested, so a literal
  "regenerate the reply from the same prompt" task is impossible
  today. Instead, each held-out real message is reduced to a
  deterministic, style-neutralized CONTENT BRIEF (what was said, with
  the voice stripped out), and the pipeline drafts that brief in the
  same context (channel, audience, medium, goal). The comparison then
  isolates exactly the quantity Voice OS claims to model: voice, not
  content recall. Capturing thread context at ingest time is a listed
  future extension, not smuggled scope.
- **Leakage is analyzed, not assumed away.** The held-out split is the
  existing content-hash split (`voice_os/holdout.py`, 20%). Guarantees
  actually enforced by code: mined artifacts train on the held-in side
  only (`mine/weights.py::is_train`), and exemplar retrieval skips
  held-out chunks (`voice_os/model.py::_exemplars`), so the pipeline
  can never see the real message it is scored against. Known residual
  leaks, stated rather than hidden: (a) the baseline axis mean/std is
  aggregate statistics over the full corpus including held-out text;
  it acts only as a normalizer, but it is not held-in-pure. (b) The
  legacy drive KB prose was written by a human who had read the whole
  corpus era; no chunk text is in it, but it is not "unseen" in the
  strict ML sense. (c) The brief is derived from the real message, so
  content similarity is partly by construction; that is why the
  composite weights style fidelity above content similarity, and why
  both are reported separately instead of blended silently.
- **Offline and live numbers are never mixed.** Offline runs (regex
  personas, lexical embeddings, heuristic judge) are deterministic and
  free; they are the regression gate. Live runs (Claude personas,
  optional live judge) measure true alignment but vary run to run;
  they are evidence snapshots, labeled with the serving model. Every
  report records which mode produced every number.

## Composition with what exists (extend, do not rebuild)

- **Holdout split**: reuse `voice_os/holdout.py::is_holdout` verbatim.
  No new split, no new hash scheme.
- **Generation**: each case runs through the real product surface,
  `voice_os.draft()`, so the harness exercises the exact graph
  (prepare -> generate -> critique -> qa_gate -> revise) that callers
  use, including KB snapshotting and provenance stamping. Inner draft
  runs are checkpointed under the harness var root with case-scoped
  run ids, so a bad case can be replayed with the standard
  `voice_os.run_history()`.
- **Stylistic dimensions**: the six canonical axes (`voice_os/axes.py`)
  plus tone metrics (`voice_os/tone.py`) are the judged dimensions.
  The paired style score reuses `AxisProfile.fidelity` with the REAL
  message's axis scores as the target, so "fidelity" means the same
  0..1 quantity everywhere in the repo.
- **LLM access**: the judge goes through `voice_os/llm.py::complete`
  with the existing offline fallback semantics; `VOICE_OS_OFFLINE=1`
  forces the deterministic path end to end.
- **Graph pattern**: mirrors `voice_os/evolution/graph.py`: like the
  product and evolution layers, the harness confines its langgraph
  import to its own graph.py (the only module in the HARNESS package
  that imports it), with serializable state, SqliteSaver in its OWN
  database file (`<var>/harness.sqlite`) and thread namespace
  (`eval-...` run ids), and run-history projection from trace notes.
- **Determinism contract inherited from first commit**
  (docs/determinism.md): RNG-free, sorted iteration, run-scoped fields
  documented, offline surfaces covered by the double-run invariant in
  `tests/test_determinism.py` and a golden lock on the fixture-corpus
  summary.

## Package layout

    voice_os/harness/
        __init__.py     public API; stdlib-only import
        cases.py        held-out case selection + content briefs (stdlib)
        scoring.py      embeddings, paired style fidelity, judge (stdlib)
        gate.py         regression gate: compare summaries, tolerances (stdlib)
        graph.py        the LangGraph eval loop (only langgraph import)
        __main__.py     CLI: run / gate / runs / report

Public API, exported lazily from `voice_os/__init__.py` via the
existing PEP 562 hook (import voice_os stays stdlib-only):

    voice_os.harness_run(...)          # run an eval, return the summary
    voice_os.harness_gate(...)         # compare a summary to a baseline
    voice_os.harness_history(run_id)   # checkpoint projection for a run
    voice_os.describe_harness_graph()  # mermaid text

## Case selection (cases.py)

Deterministic, stratified, no RNG:

1. Stream the chunk store; keep chunks that are tier 1, held out
   (`is_holdout`), well formed, 12..400 words, and whose context
   validates (same tolerant parsing as `voice_os/eval.py`).
2. Group by (channel, audience). Within each cell, sort by content
   hash and take the first `per_cell` (default 6). Cells are processed
   in sorted order; a total cap (default 72) trims the tail cells
   after per-cell quotas, never mid-cell reordering.
3. A case record carries: chunk id, hash, context (channel, audience,
   medium, goal), the real text, and the brief.

Same store, same parameters -> byte-identical case list, so runs are
comparable across weeks and machines.

## Content briefs (cases.py)

The brief is ALWAYS built by the deterministic neutralizer, in both
offline and live runs, so the pipeline input never varies by mode:

- strip a recognized greeting first line / sign-off last line (reuse
  the fixed lexicons in `voice_os/evolution/patterns.py`; nothing
  name-shaped is matched, same privacy stance as the evolution module);
- drop emojis, convert exclamation marks to periods, remove hedge and
  filler tokens (the axes hedge lexicon), collapse whitespace.

Limitation, stated: lexicon neutralization is partial; distinctive
vocabulary survives into the brief. Since the pipeline's job is to
restyle the brief INTO the target voice, residual style in the input
biases similarity UP for both weak and strong models equally; the
paired style comparison (below) is the discriminating metric, and the
composite weights reflect that. A live LLM paraphrase brief would
neutralize harder but would make case inputs nondeterministic; it is
explicitly out of scope for v1.

## Scoring (scoring.py)

Per case, generated vs real:

1. **Embedding similarity**, pluggable backend:
   - `lexical` (default, deterministic, stdlib): cosine over
     content-word term frequencies (`similarity.content`) and cosine
     over character 3-gram frequencies (`similarity.surface`). This is
     the backend the gate uses.
   - `voyage` (optional, env-gated: `VOICE_OS_EMBED_BACKEND=voyage`,
     model from `VOICE_OS_EMBED_MODEL`): true semantic embeddings via
     the voyageai SDK when installed and keyed; guarded import, hard
     error rather than silent fallback when requested but unavailable.
     Live-only evidence, never gated on.
   Every report names the backend that produced its numbers.
2. **Paired style fidelity** (deterministic, the core metric):
   `AxisProfile(mean=score_text(real), std=baseline.std)` scored
   against `score_text(generated)` gives overall + per-axis fidelity
   of the generated text TO THE REAL MESSAGE, not to a modeled target.
3. **Tone deltas** (deterministic): mean absolute error between the
   two sides' derived tone metrics.
4. **LLM judge** on the six axes: given the real message (A) and the
   generated draft (B), rate 1..5 per axis how well B matches A, plus
   `same_author` 1..5 overall, returned as strict JSON. Live mode uses
   `llm.complete`; a malformed or failed response records an error and
   falls back. Offline mode (and the fallback) uses a deterministic
   mapping from the paired per-axis fidelities to 1..5. The judge
   record always carries `mode: "live" | "offline"`; offline judge
   numbers are derived from metric 2 and are never presented as an
   independent opinion.
5. **Safety counts**: banned-phrase hits (`qa.find_banned` with the
   merged list) and em-dash occurrences in the generated text
   (Mitchell's standing outward ban; counted here so the parked scrub
   work item gets a measured baseline).
6. **Pipeline telemetry**: qa decision, revisions used, persona modes,
   inner run id.

Composites, kept separate by mode:

- `alignment_offline` = 0.60 * paired style overall
  + 0.25 * similarity.content + 0.15 * similarity.surface.
  Deterministic; THE gated number.
- `alignment_judged` = 0.50 * paired style overall
  + 0.20 * similarity.content + 0.30 * judge same_author (normalized).
  Reported only when the judge ran live; this is the number to hold
  against the manual ~78-86% estimate.

Weights live in one named constant with this rationale next to them;
changing them is a design decision, not a tweak.

## Report shape

Aggregation: overall, per channel, per audience (and per (channel,
audience) cell for drill-down): n, mean alignment (offline and, when
present, judged), mean paired style overall + per-axis means, mean
similarities, tone MAE, pass rate, banned hit rate, em-dash rate.
Cells inherit the case stratification, so per-channel and per-audience
fidelity is a first-class output, not a pivot someone runs by hand.

Persistence under the gitignored var root. `<var>` resolves by the
repo-standard convention (explicit `var_dir` argument, else the
`VOICE_OS_VAR_DIR` environment variable, else the repo-root `var/`),
exactly as in the product and evolution layers:

    <var>/eval/reports/<run_id>.json          # full: real + generated text (PERSONAL)
    <var>/eval/reports/<run_id>.summary.json  # numbers only, no message text
    <var>/eval/baseline.json                  # the accepted summary the gate compares to

Full reports contain personal text and never leave var/. Summaries
contain only metric numbers, counts, and context labels; they are what
the gate consumes and what can be quoted in PR bodies.

## The LangGraph eval loop (graph.py)

    select -> run_case -> (more cases? run_case : aggregate)
                                -> aggregate -> persist -> END

- `select`: build the case list (pure, deterministic); record case
  count and stratification in trace notes.
- `run_case`: for the cursor case, call `voice_os.draft()` with the
  case context and brief, score generated vs real, append one result
  record, advance the cursor. One checkpoint per case: an interrupted
  eval resumes from the last scored case, and per-case progress is
  visible in `harness_history()`.
- `aggregate`: fold results into the summary (pure).
- `persist`: write the two report files, refresh nothing else; the
  baseline moves only by explicit gate command.

State (`EvalState`): config paths + mode flags, `cases`, `cursor`,
`results` (operator.add), `summary`, `report_path`, `summary_path`,
`trace_notes` (operator.add). All values JSON-serializable.
Checkpoints: SqliteSaver on `<var>/harness.sqlite` (var root resolved
via `var_dir` / `VOICE_OS_VAR_DIR` / repo default, as above), thread
ids `eval-<utcstamp>-<hex8>`; recursion limit sized to
`2 * len(cases) + 12`. Inner `draft()` calls receive
`var_dir=<var>/eval` so pipeline checkpoints and KB snapshots from
eval runs live under `<var>/eval/`, fully separated from product runs.

Run-scoped fields excluded from determinism comparisons: run id,
generated_at timestamps, absolute paths, inner run ids. Everything
else in the offline summary is byte-stable across double runs.

## The regression gate (gate.py)

`gate(current_summary, baseline_summary)` returns a list of
regressions; empty list = pass. Gated metrics and default tolerances
(absolute drops, chosen tight because the offline path is
deterministic: any change is a real behavior change, not noise):

| Metric                              | Tolerance |
|-------------------------------------|-----------|
| overall alignment_offline           | 0.005     |
| overall paired style fidelity       | 0.005     |
| per-channel alignment_offline       | 0.02      |
| per-audience alignment_offline      | 0.02      |
| banned hit rate (increase)          | 0.005     |
| em-dash rate (increase)             | 0.005     |

Cells with n < 4 are reported but not gated (too noisy to block on).
Pass rate and tone MAE are reported, not gated, in v1: offline regex
personas legitimately fail the 0.80 qa gate on hard contexts, so pass
rate would gate on persona luck rather than model quality.

Enforcement layers, from automatic to manual:

1. **pytest (automatic on every PR)**: `tests/test_harness.py` runs
   the full offline loop on the synthetic fixture corpus and (a)
   byte-locks the scrubbed summary against a committed golden
   (`tests/fixtures/golden/harness_summary.json`, regenerated only via
   `tests/regen_goldens.py`), (b) asserts `gate()` semantics with unit
   fixtures (drop -> regression, within-tolerance -> pass, small-n
   skip). The repo-wide gitignore blocks `*.json` except under
   `tests/fixtures/`, which is exactly where the committed golden
   lives. This runs inside the standard suite every PR already runs.
2. **pre-push hook (automatic once hooks are enabled)**:
   `.githooks/pre-push` runs the fixture harness test, then, when
   `<var>/eval/baseline.json` exists, the real-corpus offline gate:
   `python3 -m voice_os.harness gate`. Nonzero exit blocks the push.
   Like the existing privacy pre-commit hook, this requires the
   per-clone opt-in `git config core.hooksPath .githooks` (already
   set in the primary working clone; fresh clones must run it, per
   docs/ingestion.md). It is a safeguard for enabled clones, not a
   guarantee for every contributor; layer 1 in pytest is the layer
   that runs everywhere. Escape hatch for emergencies:
   `VOICE_OS_SKIP_EVAL_GATE=1`, which prints loudly that the gate was
   skipped. Target runtime under ~3 minutes; if the real-corpus run
   exceeds that in practice, the default case cap shrinks before the
   hook gets softened.
3. **explicit baseline moves only**: `python3 -m voice_os.harness gate
   --update-baseline` copies the current summary over
   `<var>/eval/baseline.json` after a passing comparison or an explicit
   `--force`. Nothing ever moves the baseline implicitly, mirroring
   the evolution module's stored-baseline semantics.

Workflow integration (documented in the README dev section): any PR
touching `voice_os/`, `mine/`, or `ingest/` runs the suite (layer 1
automatically) and pushes through layer 2; the PR body quotes the
before/after summary numbers when the gate's baseline moved.

## CLI (__main__.py)

    python3 -m voice_os.harness run [--per-cell 6] [--cap 72] [--live]
                                    [--json] [--corpus-dir corpus]
    python3 -m voice_os.harness gate [--summary PATH] [--baseline PATH]
                                     [--update-baseline] [--force] [--json]
    python3 -m voice_os.harness runs [--json]        # list eval runs
    python3 -m voice_os.harness report <run_id> [--json]

`run` prints the rendered summary (per-channel / per-audience tables);
`--live` lifts VOICE_OS_OFFLINE-style forcing for personas and judge
(credentials still decide), and stamps the serving model into the
summary provenance. `gate` exits 0 on pass, 1 on regression, 2 on
missing baseline.

## Privacy

- Full reports, checkpoint databases, KB snapshots, and the real
  baseline all live under the gitignored `var/`; case text never
  enters git, matching the product and evolution layers.
- Summaries are numbers + context labels only; they are safe for PR
  bodies and are the only harness output ever quoted outside var/.
- Live mode sends briefs, drafts, and real held-out messages (judge
  prompts) to the Anthropic API; `VOICE_OS_OFFLINE=1` remains the
  privacy override, checked per call as today.

## Determinism

- Offline case selection, briefs, generation, scoring, aggregation:
  fully deterministic; double-run byte equality asserted in
  `tests/test_determinism.py` after scrubbing the documented
  run-scoped fields.
- The fixture-corpus summary is golden-locked; regeneration goes
  through `tests/regen_goldens.py` so a diff is always a reviewed
  decision.
- Live numbers are labeled with model id and never enter the gate.

## Out of scope for v1 (parked, deliberate)

- Live-judge calibration study (judge vs Mitchell's own ratings on the
  correction-log rubric in docs/architecture.md).
- LLM-paraphrase briefs (harder neutralization at the cost of
  deterministic case inputs).
- Thread-context ingestion so cases can become true reply tasks.
- Semantic-embedding baselines in the gate (voyage backend is
  evidence-only until determinism and cost are settled).
- CI runners: this repo gates locally (pytest + pre-push); a hosted CI
  lane can adopt the same commands unchanged if one ever lands.
