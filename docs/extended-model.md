# Voice OS: Extended Multi-Dimensional Model

Design for extending the six-axis scoring foundation with six new dimensions:
emotional calibration, situational stakes awareness, audience-specific patterning,
medium adaptation, explicit goal alignment, and voice evolution tracking with
drift detection. Approved 2026-07-07.

## Design principle

Every new dimension is expressed in currencies the existing system already
understands:

- Axis deltas: plain dicts merged onto the baseline, the `calibrate()` pattern.
- Tolerance-banded numeric targets: the `AxisProfile.fidelity` pattern
  (tolerance `max(2 * std, 0.12)`), reused for tone metrics.
- Banned phrase lists: consumed unchanged by `qa.find_banned()`.
- Composable revision-signal strings: personas already accept `list[str]`.

Nothing in the existing six-axis, dual-persona, or QA gate flow is replaced.
Existing `calibrate()`, `gate()`, and `run_pipeline()` signatures are untouched;
all extensions are new functions or keyword-only arguments.

## Dimension to mechanism map

| Dimension | Mechanism | Mined vs heuristic |
|---|---|---|
| Emotional calibration | `ToneProfile` per context: a metric space parallel to the axes. Deviations become revision signals via `gate_extended()` | Norms mined from chunk `context.tone_signals`; guidance strings heuristic |
| Stakes awareness | `STAKES_DELTAS` table plus `infer_stakes()` regex heuristic; high and critical stakes tighten tone tolerance | Heuristic v1 (no stakes labels exist in the data) |
| Audience patterning | Mined per-audience axis and tone profiles blend with hand `AUDIENCE_DELTAS`; per-recipient deltas keyed on `context.relationship_hint` are the finest grain | Data-mined with shrinkage toward hand deltas |
| Medium adaptation | New `MEDIUM_DELTAS` on chunk `context.medium` (dm, post, comment, story, email, sms, script, spoken), finer than channel | Mined where supported, hand-seeded fallback |
| Goal alignment | `GOAL_DELTAS` table plus a goal line appended to persona signals. `GOALS` extends the ingest heuristic-v1 set with negotiate, de-escalate, set-expectations | Five ingest-taggable goals mined; three new goals heuristic until a model retag pass (the chunk `context.inference` field exists for this) |
| Drift detection | Half-year window axis and tone means, plus lexical marker series (validation case: the known 2020 yea to yeah shift). Proposals only, never auto-applied | Data-mined |

## Module layout

### Runtime additions under `voice_os/` (committable, loads artifacts only, never contains personal data)

- `contexts.py`: `GOALS`, `STAKES` (low, routine, high, critical), `MEDIA`
  vocabularies; frozen `VoiceContext` dataclass
  (channel, audience, situation, goal, stakes, medium, recipient) with a
  fail-fast `validate()` mirroring `calibrate()`; `infer_stakes(text, situation)`
  as a deterministic regex heuristic in the `axes.py` style.
- `tone.py`: canonical `tone_signals(text)` implementation and `ToneProfile`
  dataclass (mean, std) with `deviations(observed)` returning revision-signal
  strings. Refactor: `ingest/enrich.py` imports `tone_signals` from here.
  The dependency direction is legal because ingest already imports
  `voice_os.calibration`; the reverse would be circular. `word_count` and
  `words_per_minute` stay on the ingest side.
- `holdout.py`: `is_holdout(chunk_hash, pct=20)` via hash-prefix mod 100.
  Deterministic and stable across re-ingestion because it keys on content hash.
  Miners use the train side; evaluation uses the holdout side.
- `mined.py`: `MinedArtifacts` dataclass and `load_artifacts(mined_dir)`.
  Missing files degrade to None or empty; a present file with the wrong
  artifact name or version raises `ValueError` (repo convention).
- `model.py`: the queryable facade (see API section).
- `drift.py`: pure math with no file I/O: `window_profiles`, `flag_shifts`
  (tolerance `max(2 * std, 0.12)` sustained 2 consecutive windows),
  `marker_series`, `suggest_boundaries`.
- `eval.py`: scorecard CLI, `python -m voice_os.eval`.

### Extensions to existing modules

- `calibration.py` gains `GOAL_DELTAS`, `STAKES_DELTAS`, `MEDIUM_DELTAS`
  and `calibrate_extended(baseline, ctx, mined) -> (target, sources)`.
  Delta application order: channel, medium, audience, recipient, situation,
  stakes, goal; clamped to [0, 1] at each step exactly like today.
  Where a mined profile has sufficient support, its delta (group mean minus
  global baseline mean) blends with the hand delta by shrinkage:
  `delta = lam * mined + (1 - lam) * hand` with `lam = n / (n + 50)`.
  One inspectable knob; degrades to the hand tables at zero data.
- `qa.py` gains `gate_extended(...)`: calls the existing `gate()` then appends
  tone deviation signals. Tone is advisory-only in v1: it adds signals the
  generative persona acts on but never flips a pass to a cycle. Promotion to
  blocking waits for evaluation evidence that the mined norms are tight.
- `__init__.py`: `run_pipeline` keeps its exact signature and adds keyword-only
  `goal="unknown"`, `stakes="routine"`, `medium=None`, `recipient=None`,
  `mined_dir=None`. With defaults every added delta is zero and the output is
  byte-identical to today, guarded by a golden regression test.

### New top-level `mine/` package (batch tooling, committable code, personal-data outputs)

Mining lives outside `voice_os/` for the same reason `ingest/` does: it is
offline batch tooling over personal JSONL data, and the runtime only ever
loads validated JSON artifacts. `mine/` may import both `voice_os` and
`ingest`; no cycles.

- `cli.py`: `python -m mine run --job recipients|tone|ngrams|drift|all`,
  `python -m mine status`, `python -m mine contrast-gen --n 300`.
- `weights.py`: tier-weighted counting and weighted mean/std using
  `voice_os.corpus.TIER_WEIGHTS`; train-split filtering via
  `voice_os.holdout.is_holdout`.
- `recipients.py`: per-recipient deltas. Support gate: at least 40 chunks and
  1,500 tier-weighted words; deltas clipped to [-0.35, +0.35]. Group key is
  the normalized `relationship_hint`, with a second aggregation level by
  recipient email domain.
- `tone_norms.py`: per audience, per medium, per goal, and per
  (audience, medium) pair profiles: axis mean/std, tone mean/std, support n.
- `ngrams.py`: 1- to 4-gram frequency diffs (see anti-pattern section).
- `contrast.py`: pluggable contrast-corpus loader (.txt passages or .jsonl).
- `drift.py`: orchestrates the `voice_os.drift` pure functions and writes the
  artifact. Uses all dated chunks unweighted, including tiers 3 and 4,
  because temporal windowing is itself the recency treatment.

## Mined artifacts

Stored at `corpus/mined/*.json`. This path is protected three ways:
`.gitignore` ignores `corpus/` wholesale, `*.json` is ignored except for
explicit exemptions, and `.githooks/pre-commit` blocks staged `corpus/`
paths. The hook only runs in clones that have enabled it with
`git config core.hooksPath .githooks` (already set in the primary working
clone); fresh clones must run that command, so the gitignore rules are the
baseline protection and the hook is defense in depth.

All artifacts share one envelope:

```json
{
  "artifact": "recipient_deltas | context_profiles | ngram_banned | drift_report",
  "version": "1.0",
  "generated_at": "2026-07-07T15:00:00",
  "miner": "mine.recipients@1.0",
  "train_split": {"method": "hash_prefix_mod100", "holdout_pct": 20},
  "tier_weights": {"1": 1.0, "2": 0.6, "3": 0.25, "4": 0.0},
  "params": {"min_chunks": 40, "min_weighted_words": 1500},
  "data": {}
}
```

Recipient names are personal data: the artifact never leaves `corpus/mined/`,
and query output includes only the deltas actually applied, not the roster.

## n-gram anti-pattern mining

Moves the banned list from hand-curated toward statistically mined evidence.

- Tokenize: lowercase, keep internal apostrophes, split on non-word characters.
- Count 1- to 4-grams over (a) train-split self chunks, tier-weighted, and
  (b) the contrast corpus; normalize both to frequency per million tokens.
- Score with smoothed log-odds: `score = log((f_contrast + a) / (f_self + a))`,
  `a = 0.5` per million.
- Ban criteria: contrast raw count of at least 5, score of at least 2.0
  (roughly 8x), and either n of at least 2 or the unigram absent from the
  `data/never_ban.txt` guard list so statistics can never ban common
  function words. The guard list is a deliverable of the n-gram PR and is
  not in the repository yet.
- Output carries per-entry evidence (frequencies and log-odds) so every mined
  ban is auditable. The merged hand plus mined list flows into the existing
  gate and offline persona replacement path unchanged.

Contrast corpus is pluggable, two sources (both are deliverables of the
n-gram PR; neither exists in the repository yet):

1. Synthetic seed `data/contrast/synthetic_llm.txt`: 150 to 250 short
   passages of characteristic LLM boilerplate. Small, synthetic, safe to
   commit, and what tests run against once the n-gram PR lands.
2. Generated set `corpus/contrast/generated.jsonl` via
   `python -m mine contrast-gen --n 300`, prompting Claude through the existing
   `voice_os.llm.complete` across the channel, audience, and goal grid.
   API spend approved 2026-07-07. The command refuses to run under
   `VOICE_OS_OFFLINE=1` and prints an estimated call count before proceeding.
   Output is gitignored.

## Drift detection

- Bucket all dated chunks into half-year windows from `provenance.timestamp`;
  merge windows with fewer than 30 chunks.
- Per window: axis means (via `score_text`) and tone means.
- Flag an axis shift when the window mean departs from the trailing four-window
  mean by more than `max(2 * trailing_std, 0.12)`, sustained for two
  consecutive windows. This deliberately reuses the fidelity tolerance formula.
- Lexical markers: configurable pairs, default `yea/yeah` and
  `gonna/going to`, with a per-window frequency series and crossover detector.
  The known 2020 yea-to-yeah transition is the validation case.
- Output includes human-readable suggestions (for example, a proposed tier
  boundary move). Suggestions are proposals printed by the eval CLI and
  surfaced in `VoiceModel.meta`; tier boundaries in `voice_os/corpus.py`
  remain the single source of truth and change only by explicit human edit.

## Queryable API

```python
@dataclass
class QueryResult:
    context: dict                     # normalized VoiceContext
    target_profile: dict[str, float]  # six axes, post all deltas
    tone: ToneProfile | None          # tone targets and tolerances for this context
    banned: list[str]                 # hand list merged with mined n-grams
    exemplars: list[dict]             # up to 5 matching held-in chunks
    guidance: list[str]               # ready-to-use persona signal strings
    sources: dict[str, str]           # per dimension: mined | heuristic | absent
    meta: dict                        # versions, artifact ages, drift summary

class VoiceModel:
    @classmethod
    def load(cls, corpus_path="corpus/voice_corpus.txt", *,
             chunks_dir="corpus/chunks", mined_dir="corpus/mined",
             banned_path="data/banned_list.txt") -> "VoiceModel": ...
    def query(self, **context_kwargs) -> QueryResult: ...
    def gate_draft(self, draft: str, q: QueryResult) -> GateResult: ...
    def run(self, draft: str, max_cycles: int = 2, **context_kwargs) -> dict: ...
```

`load` degrades gracefully: on a fresh clone with no chunks or mined
artifacts, everything falls back to the hand tables and `sources` says so.
Behavior stays offline-deterministic either way. Exemplars are selected from
tier 1 and 2 held-in chunks matching audience, medium, and goal, ranked by
fidelity to target then recency; they feed future live persona prompts as
few-shot voice samples and are never persisted.

## Evaluation

Split: `is_holdout` hash-prefix mod 100, 20 percent held out, shared by
miners (train only) and eval (holdout only).

`python -m voice_os.eval` prints a scorecard over held-out chunks:

1. Context fidelity, the headline before/after: mean fidelity of real
   held-out text against (a) baseline-only target, (b) hand-calibrated target,
   (c) extended mined target; overall and per audience, medium, and goal.
   The extended model wins when (c) beats (b) beats (a).
2. Tone calibration error: mean absolute error of predicted tone targets vs
   observed metrics, mined norms vs global norms.
3. Tagging accuracy: goal and audience inference against labels. Structural
   silver labels come free; `python -m voice_os.eval label --sample 30`
   prints a deterministic sample for hand-labeling into gitignored
   `corpus/labels/`; accuracy reported only when labels exist.
4. Banned-list efficacy: false-positive rate on held-out self chunks
   (target near zero) and recall on contrast passages.
5. Drift: report flags and suggestions printed, not scored.

`--save` writes to the gitignored `corpus/runs/` directory.

## Sequencing and decisions log

Implementation lands as green-test PRs, each merged after a clean Qodo review:

- PR A: this design doc.
- PR B: foundations (contexts, tone, holdout, extended calibration and gate,
  pipeline kwargs, golden regression test).
- PR C: mining core plus VoiceModel facade.
- PR D: n-gram anti-patterns plus contrast corpus.
- PR E: drift detection.
- PR F: eval scorecard.

Decisions (Mitchell, 2026-07-07):

- Plan approval served as the design approval gate; implementation proceeds
  without a second pause.
- Each PR runs the Qodo review loop until the verdict reads Bugs (0), then merges.
- API spend approved for the generated contrast corpus.
- Step 0 ingestion covered email and messages sources before mining work.

Tradeoffs adopted: stdlib only (no numpy or sklearn; the math is counting,
weighted means, and log-odds); shrinkage blend over hierarchical modeling;
smoothed log-odds over Dirichlet-prior z-scores; tone advisory-only in v1;
drift proposes and never applies; mining lives outside the runtime package.
