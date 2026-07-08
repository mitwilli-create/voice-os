# Voice Reflection Engine

This document is the design gate for the Voice Reflection Engine implementation PRs. It specifies a guaranteed, automatic learning loop that runs after editing sessions: record what changed between the draft Voice OS produced and the text Mitchell actually shipped, capture his stated directions, extract structured learnings, and fold those learnings back into calibration and guidance so the next draft starts closer to his current voice. It follows the pattern established by `docs/evolution.md` and `docs/extended-model.md`: extend the existing system in currencies it already understands, change nothing that works, and bound everything new.

Revision note: this design was hardened on 2026-07-08 by a six-model adversarial council review (report at `var/reviews/reflection-engine-council-review-20260708.md`, gitignored). The review's convergent findings are integrated throughout rather than appended; the load-bearing changes are the fixed reference point for residual learning, the eval-freshness hard gate, deterministic re-derivation of corroboration, locking and atomic writes on the artifact path, and calendar-time decay via an explicit `as_of_date`.

## 1. Overview and motivation

Voice OS learns through one path today: batch mining over the corpus (143,154 chunks), blended into calibration by shrinkage. That path is accurate but slow, and it only sees text after it has been ingested as corpus chunks. The highest-signal training data the system ever receives, an editing session where Mitchell corrects a draft and explains why, currently evaporates when the session ends. The draft, the final text, and the reasoning connecting them are never recorded, never measured, never learned from.

A second problem compounds the first: signal the system already mines does not all reach generation. The evolution module extracts Mitchell's current greeting, sign-off, marker, and sentence-shape patterns, but they surface only as trace notes. Drift and evolution flags are advisory strings. Recipient deltas are mined but `draft()` never threads a recipient. Doc-type profiles are anticipated in the loader but not wired. The generation target is built from less signal than the system possesses.

The Reflection Engine fixes both. It adds a fast learning path alongside the slow one, and its Phase 0 activates the dormant signal so the fast path corrects a target that is already using everything the corpus knows.

The design borrows three patterns from the agent-systems literature and adapts them to this repo's discipline:

- **Structured reflection, not transcripts** (Reflexion, Self-Refine): the durable record of a session is a set of schema-validated learning records, not conversation history. Raw texts and verbatim directives stay in gitignored `var/`, are never copied into artifacts, and exist only to make a session reproducible and retractable.
- **Actor / Reflector separation**: the Actor is the existing product graph, unchanged. The Reflector is a separate bounded pass that analyzes the delta between draft and final. It never generates prose for the user, and its output is treated as advisory claims: every field that gates application is re-derived deterministically from measurements (section 5).
- **Harness engineering over prompting**: learnings improve the system around the model (calibration tables, guidance, gate context), not a growing prompt. Every applied learning is a versioned, measurable, reversible change to the harness.

The mental model is two-speed memory. Mining is slow memory: corpus-scale, high-inertia, shrinkage-weighted at `SHRINKAGE_N = 50`. Reflection is fast memory: few curated observations, applied within one session, decayed over weeks. As shipped finals eventually re-enter the corpus and the mined profiles absorb the change, decay retires the corresponding reflection learnings. The fast path hands the signal off to the slow path instead of double-counting it.

One claim the review forced this document to state honestly: self-refinement loops that optimize against their own imperfect evaluators are known to converge to evaluator-preferred optima rather than the true objective (Reflexion, Self-Refine, and the reward-hacking literature). This design therefore does not treat any single mechanism as a stability proof. Stability rests on a stack: a fixed reference point for learning, hard support floors, clips, calendar-time decay, a per-cell stability governor, an eval-freshness gate, and three rollback paths.

## 2. Goals and non-goals

### Goals

1. Every `draft()` call opens a reflection obligation that is tracked until resolved, so no editing session is silently lost.
2. Deltas between draft and final are measured deterministically in the currencies the system already speaks: six-axis scores, tone metrics, pattern profiles, banned hits, length, structure.
3. User directives and measured deltas become schema-validated learning records with category, scope, direction, magnitude, confidence, and provenance, where every application-gating field is deterministically derived, never trusted from the LLM.
4. Learnings consolidate into a versioned mined-style artifact that calibration consumes selectively by context cell, with calendar-time decay, dual support floors, and conflict handling in both directions (opposing and concordant).
5. Application is tiered: bounded numeric axis corrections auto-apply behind an eval-freshness gate; lexical changes (banned list, never-ban) surface as human-confirmed proposals only.
6. Generation consumes every mined signal class: Phase 0 wires pattern profiles, evolution and drift flags, recipient deltas, and doc-type profiles into the generation path.
7. The whole loop is bounded (fixed node count, at most one LLM call per session, no retries), offline-capable, crash-safe, and measurable through the existing harness.

### Non-goals / out of scope (parked)

- No long-term retention of raw conversation history. Session texts live in `var/` for reproducibility and retraction, not as a corpus.
- No unbounded or self-triggering loops. Reflection runs once per session, linearly, and never spawns further generation.
- No automatic edits to `data/banned_list.txt` or `data/never_ban.txt`. The banned merge point stays singular and hand-edited.
- No writes to `corpus/chunks/` and no triggering of ingestion. Reflection is not an ingest path.
- No numeric tone application in v1. Tone learnings surface as guidance; the tone layer stays advisory, matching the extended-model decision.
- No automatic harness runs from the reflection graph. Eval remains an explicit, spend-visible command; what is automatic is the demotion of stale learnings (section 6).
- Multi-user support, streaming, and async execution.

## 3. Phase 0: activate dormant signal

Reflection converges fastest when the base target already uses all available signal, so a learning corrects a real residual rather than compensating for unwired plumbing. For that reason Phase 0 lands before the engine (decided): five small, independently measured PRs. Each is its own experiment: the exemplar-fusion null result showed that a plausible new signal can fail to move the judge, so each activation lands with a before/after harness comparison. The review added a caution that shapes Phase 0 sequencing discipline: because these signals and reflection all enter generation near each other in time, attribution confounds are real. Landing Phase 0 first, one PR at a time, each with its own A/B, is also what keeps reflection's later measurements interpretable.

1. **Pattern-profile fusion.** The Tier 1 pattern profile (greeting and sign-off distributions, sentence shape, marker rates from `evolution/patterns.py::extract_pattern_profile`) currently exists only for drift detection. Distill it into fixed-template guidance strings in the `prepare` node, exactly as KB fusion (`docs/kb-fusion.md`) distills compact-KB patterns. Template slots are validated against the fixed evolution lexicons (greeting forms, sign-off forms, markers), never free vocabulary, so the strings stay inside the privacy invariant.
2. **Evolution and drift flags become guidance.** Today `evolution_flags` and `drift_flags` reach the `prepare` node as trace notes only. Convert emerging/fading flags into guidance strings (emerging phrases encouraged, fading phrases discouraged) flowing through the existing guidance-to-persona path. This is steering, not mutation: banned-list changes stay propose-only.
3. **Recipient threading.** `VoiceContext.recipient` and the mined `recipient_deltas.json` both exist, but `draft()` never sets a recipient, so per-person calibration fires only on the lower-level `VoiceModel` paths. Thread `recipient` through `draft()` into `initial_state` and the `prepare` node.
4. **Doc-type wiring.** The chunk schema carries `doc_type` and `mined.py::group_profile` anticipates a `doc_types` kind. Mine the per-doc-type profile and wire the calibration lookup.
5. **Baseline and miner refresh.** The stored evolution baseline and several mined artifacts straddle the corpus expansion to 143,154 chunks. Re-run the miners and `drift-run --update-baseline` so both the slow path and the pattern signal reflect the full corpus before reflection starts correcting against them.

## 4. Composition with what exists (extend, do not rebuild)

The Reflection Engine is a sibling of `evolution/` and reuses its skeleton end to end. Nothing in the axis layer, calibration order, personas, QA gate, product graph, or harness is replaced.

- **Structural template**: `voice_os/evolution/graph.py` (guarded langgraph import, serializable state, own SqliteSaver, result envelope, run history projection) and `voice_os/evolution/baselines.py` (content-hashed store, manifest plus payload, timestamp ids with sha256 identity).
- **Write path precedent**: mined-style JSON artifact in `corpus/mined/`, standard envelope, validated by `voice_os/mined.py::validate_artifact`, registered in `ARTIFACT_FILES`. Like `evolution_flags.json`, the artifact is written by its own graph, not by a `mine/` job.
- **Read path precedent**: `calibration.py::calibrate_extended` already blends mined group profiles over hand tables by shrinkage (`_blended_deltas`, `lam = n / (n + SHRINKAGE_N)`, clip 0.35). Reflection adds one more layer in the same fixed-order style, with its own tighter clip and a freshness gate.
- **Signal currency**: guidance strings and revision signals. Learnings that become guidance flow through `VoiceModel.query` into persona prompts with zero graph changes.
- **Untouched**: `axes.py`, `tone.py`, `qa.py`, `personas.py`, the product graph's node set and edges, `run_pipeline` / `run_cycles`, the harness graph, and the singular banned merge in `VoiceModel.banned`.

### Package layout

```
voice_os/reflection/
    __init__.py     lazy public surface: reflect(), pending_sessions(),
                    reflect_run_history(), consolidate(), describe_reflection_graph()
                    (stdlib-only import; langgraph touched only when the graph runs)
    __main__.py     python3 -m voice_os.reflection (parity with evolution/__main__.py)
    sessions.py     pending queue + content-hashed episodic session store under
                    var/reflection/ (structural clone of evolution/baselines.py)
    deltas.py       pure measurement math over (draft, final) pairs
    schema.py       learning-record schema constants + validate_learning(), including
                    deterministic re-derivation of every application-gating field
    reflector.py    the one bounded LLM call + deterministic offline fallback
                    (the only reflection module importing voice_os.llm)
    aggregate.py    consolidation: decay, shrinkage, conflict netting, governors,
                    artifact assembly
    locks.py        advisory file lock helpers shared with the harness
    graph.py        the only module importing langgraph; five fixed nodes
```

CLI: `reflect`, `reflect-status`, `reflect-proposals`, and `retract` subcommands join `voice_os/product/cli.py`, so the documented entry point is `python3 -m voice_os reflect`, matching how the skill already invokes `draft`. Public names register in `voice_os/__init__.py` through the existing PEP 562 lazy-export machinery (a `_REFLECTION_EXPORTS` frozenset beside `_EVOLUTION_EXPORTS`).

### The reflection graph

```
START -> capture -> measure -> reflect -> consolidate -> record -> END
```

Five fixed nodes, linear, no conditional edges, no loops. Checkpoints in its own `var/reflection.sqlite` with thread ids `reflect-<UTCstamp>-<hex8>`, so product, harness, evolution, and reflection runs never share a checkpoint space. `describe_reflection_graph()` emits Mermaid, matching the other graphs.

### Data models

**Pending record**, `var/reflection/pending/<run_id>.json`, written at the end of every `draft()` call. The pending record is self-sufficient: it carries everything `capture` needs, so SQLite checkpoint rotation can never strand a session in orphan mode.

- `run_id`, `created_at`
- `context` (the envelope's context dict)
- `input_text`, `draft_text` (copied at draft time; gitignored `var/`, so no privacy change)
- `target_profile` (the run's full calibrated target) and `target_base` (the same calibration computed with the reflection layer excluded: hand tables plus mined blend only). `target_base` is the fixed reference point residual learning measures against.
- `mined_profile_hash` (content hash of the mined context profile active at draft time, for stable contested comparisons later)
- `decision`, `fidelity_overall`, `mode`, `provenance` (copied from the envelope so triage never needs the checkpoint DB)

Pending records move through a two-phase lifecycle: `pending/` until `reflect` begins, renamed to `in_progress` during a run, deleted only after the session record is durably written. A crash mid-reflect leaves an `in_progress` marker that the next run recovers rather than losing the obligation. If the pending write itself fails (disk full, permissions), `draft()` still returns, but the envelope carries `reflection.pending_write_failed = true` and a trace note counts it: the obligation contract is violated loudly, never silently.

**Session record**, `var/reflection/sessions/<session_id>/` (manifest.json plus session.json, content-hashed exactly like evolution baselines):

- Manifest: `session_id`, `created_at`, `content_hash`, `run_id`, `status`, `n_learnings`
- Body: `run_id` (nullable in orphan mode), `context`, `draft_provenance`, `draft_text`, `final_text`, `directives` (capped at 10 entries of 500 chars, each `{text, source}` where `source` is `stated` for the user's exact words or `inferred` for skill-summarized intent), `measurements`, `learnings`, `reflector` meta (`mode`, `model`, `prompt_version`, `tokens_out`), `scorer_version` (the `score_text` implementation version, so a scorer change that flips residual signs is detectable at consolidation), `status`
- `status` is one of `reflected | null-edit | lapsed | retracted`

A written session is immutable. The reflector is never re-run on a stored session, even on explicit retry, so live nondeterminism cannot produce two versions of the same edit. Retraction is the only permitted mutation, and it changes `status`, never content.

**Learning record** (validated and normalized by `schema.py::validate_learning`):

- `id`: sha256 of the canonical normalized record minus provenance (dedupe identity across sessions)
- `category`: `stylistic-axis | tone | lexical | goal | audience | medium | doc_type | recipient | process`
- `scope`: partial `VoiceContext` dict; present keys must be valid vocabulary values, absent keys are wildcards, `{}` is global. The reflector may only narrow scope to fields actually present in the session's context; a global `{}` scope on a numeric learning requires support from at least 5 distinct context cells before it can apply.
- `subject`: an axis name (must be in `AXES`), a tone metric (in `TONE_METRICS`), a phrase (lexical), or a short slug (process)
- `direction`: signed float in [-1, 1] for numeric categories; `add | remove` for lexical
- `magnitude`: axis-delta units, deterministically capped at `min(|axis_residual[subject]|, REFLECTION_DELTA_CLIP)` with `REFLECTION_DELTA_CLIP = 0.15`
- `evidence`: bounded references only, e.g. `{"kind": "measurement", "key": "axis_residual.hedging_behavior", "value": -0.21}` or `{"kind": "directive", "index": 2}`. Directive indices, never quotes: verbatim text stays in the session body.
- `confidence`: [0, 1]. The reflector's number is advisory; `validate_learning` enforces deterministic caps (not floors: a floor cannot stop a reflector that assigns 0.95 to weak evidence). Caps: stated directive plus corroborating measured delta, 1.0; stated directive only, 0.5; measured delta only, 0.4; each cap one tier lower when the backing directive is `inferred`. A record whose reflector-assigned confidence exceeds its cap is dropped with a trace note, not silently capped: a lying reflector is signal.
- `corroborated`: bool. Never trusted from the reflector. `validate_learning` recomputes it from the measurement block: the referenced measurement key must exist, `subject` must match the evidence key, and `sign(direction)` must agree with `sign(measurements[key])`. Numeric records failing any of these checks are rejected outright.
- `provenance`: `{session_id, run_id, reflector_mode, model, prompt_version}`

**Aggregated artifact**, `corpus/mined/reflection_learnings.json`, standard envelope (`artifact: "reflection_learnings"`, `version: "1.0"`, `miner: "reflection.graph@1.0"`, `generated_at`, `params`, `data`) plus `"export": "never"`: a marker that this artifact is single-user behavioral data and must not leave the machine even though it contains no prose. `params` records `sessions_hash` (sha256 of the sorted contributing (session_id, content_hash) pairs), `as_of_date`, and all tuning constants: same session store and same `as_of_date`, same artifact, byte for byte. `data` holds:

- `axis_learnings`: keyed by canonical scope key (sorted `k=v` join, e.g. `"audience=leadership|channel=email"`, `"global"` for `{}`), each `{axis: {effective_delta, n_sessions, n_corrob_sessions, weight_sum, n_eff, internal_contested, contested, frozen, last_session_at}}`. `effective_delta` is quantized to 0.01 in the shared artifact. Session ids and raw weighted means stay in a `var/`-only consolidation report (`var/reflection/artifacts/<ts>/report.json`); the shared artifact carries only what calibration needs.
- `tone_learnings`: same shape, advisory-only in v1
- `guidance`: per scope key, top 3 fixed-template strings, confidence-ordered. Template slots are restricted to axis names, tone metric names, evolution-lexicon entries, and fixed bucket labels; no free vocabulary reaches the artifact.
- `proposals`: counts only (`{banned_add: n, never_ban_add: n, vocab: n}`). The phrases themselves live in `var/reflection/proposals.json` and surface through `reflect-proposals`; verbatim user phrases never enter `corpus/mined/`.
- `stats`: `{n_sessions_total, n_null_edit, n_lapsed, pending_write_failures, override_frequency}` where `override_frequency` is the per-axis rate at which post-application residuals still exceed the noise floor (the "did the applied learning actually land" counter)

Privacy invariant, mirroring evolution's fixed-lexicon rule and hardened per the review: the artifact carries schema fields, vocabulary-validated scope values, quantized numerics, and template strings with whitelisted slots. No draft text, no directive verbatims, no free-form LLM prose, no verbatim user phrases, no session id timeline. A CI check greps the artifact fixture for email addresses, URLs, and directive substrings.

### Trigger mechanism: the guaranteed loop

Edits happen outside the system, so the guarantee cannot be a runtime invariant. It is a contract: a queue that never forgets plus interfaces that nag until it is empty. Four layers:

1. **`draft()` always enqueues.** `product/__init__.py::draft` writes the pending record after `build_result`. A failed write cannot fail the draft, but it is surfaced in the envelope and counted (see pending-record lifecycle above). The envelope gains a `reflection` key: `{"pending": true, "hint": "python3 -m voice_os reflect --run-id <id>"}`. A trace note reports the pending count so every caller sees backlog.
2. **The reflect interface.** Callable `voice_os.reflect(run_id=None, final_text=..., directives=[...], draft_text=None, context=None, var_dir=None, mined_dir=None)`. CLI `python3 -m voice_os reflect --run-id X [--directive "..."]...` with final text on stdin or `--file`, plus `--accepted-as-is` to close a pending session with final equal to draft and `--no-signal` to close one explicitly without contributing any observation. Exit codes: 0 reflected, 1 lapsed or degraded, 2 usage or validation, matching the existing CLI contract.
3. **Skill contract.** The `~/.claude/skills/voice-os` skill gains a mandatory closing step: after the drafted text is finalized in a session (edited, approved, or shipped), run `reflect` with the shipped text and the session's directives; if accepted verbatim, `--accepted-as-is`. Directive capture is hybrid (decided): the skill passes the user's exact words as `stated` directives whenever directions were given in the session, and summarizes intent shown only through edits as `inferred` directives, which carry lower confidence caps. An optional immediate redraft step verifies a correction in the same session it was made.
4. **Queue pressure and lapse.** `reflect --status` lists pending sessions and exits 1 when any exist, so a cron or loop can nag. `draft` warns on stderr above a pending threshold. Pending records older than `PENDING_TTL_DAYS = 14` move to `var/reflection/lapsed/` on the next run: recorded for stats, never learned from, and recoverable via `reflect --recover <run_id>` (recovered learnings consolidate with a staleness downweight). Stale drafts cannot poison learnings and nothing is auto-deleted.

Null-edit closures are guarded against queue-clearing abuse: after 3 consecutive `--accepted-as-is` closures in the same scope without a directive, the CLI requires either a directive or `--no-signal`. Null-edits in `stakes=low` contexts carry half the usual null-edit weight, and a null-edit contributes nothing to a scope cell that has no prior corroborated learnings (otherwise the loop would be measuring compliance, not endorsement).

## 5. Reflection process (detailed flow)

1. **capture.** Resolve inputs. With a `run_id`: read the pending record (draft text, context, `target_profile`, `target_base`, provenance are all present by construction); the product checkpoint in `var/runs.sqlite` is a diagnostic fallback, not a dependency. Mark the pending record `in_progress`. Without a `run_id` (orphan mode, for text drafted elsewhere): require `--draft-file` plus context flags. Fail fast with `ValueError` when neither a pending record nor orphan inputs resolve.

2. **measure.** Compute the full deterministic measurement block in `deltas.py`, all pure math:
   - `axis_draft`, `axis_final`, `axis_delta` via `axes.py::score_text` on both texts.
   - `axis_residual = axis_final - target_base`, per axis. The reference is `target_base`, the calibration with the reflection layer excluded, recorded in the pending record at draft time. This makes the reference point exogenous to the loop: an axis learning is an estimate of a fixed quantity (the user's offset from the base calibration), so consolidation converges to a stable estimate instead of compounding increments against a target the loop itself keeps moving. This was the review's primary stability fix; the earlier draft of this design measured against the moving target and argued residuals would self-zero, which all six reviewing models rejected as a heuristic, not an invariant.
   - `execution_residual = axis_draft - target_profile`, per axis (Gemini's catch). When `execution_residual` is persistently large while `axis_residual` is near zero, the target is right and the generator is failing to hit it. That is an execution failure, not a preference signal: it routes to process guidance and to `reflect --status` diagnostics, never to the numeric target, which would otherwise go permanently blind to it.
   - `tone_draft`, `tone_final`, `tone_delta` over `TONE_METRICS` via the shared `tone.py` computations.
   - `pattern_draft`, `pattern_final` via `evolution/patterns.py::extract_pattern_profile` per side: greeting and sign-off form changes, sentence-mean shift, marker rate changes. `diff_profiles` is deliberately not used: its `MIN_SUPPORT = 5` floors are corpus-scale and meaningless on a single document.
   - `banned_removed` (hits in the draft absent from the final: the user confirmed the ban, reinforcement) and `banned_introduced` (the user added a phrase the merged list bans: evidence the ban is wrong, a never-ban proposal candidate; the only place edits touch the banned layer, and only as proposals).
   - `length_ratio` (final words over draft words; serialized into learnings only as one of five fixed buckets: much shorter, shorter, similar, longer, much longer), `structure` (paragraph and list-marker deltas, greeting and sign-off presence flips), and `edit_magnitude` via `difflib.SequenceMatcher(autojunk=False)` over NFC-normalized text (both settings matter: the autojunk heuristic and Unicode normalization differences cause cross-platform ratio drift), reported both raw and normalized by `min(len(draft), len(final))`.

   If similarity is at least 0.98 and there are no directives, short-circuit to a `null-edit` session: no LLM call, but the session still consolidates under the null-edit rules in section 4.

3. **reflect.** Build the bounded prompt: the measurement block as compact JSON, the directives, and changed-segment excerpts capped at roughly 800 words total. The Reflector never sees the full texts. Exactly one `llm.complete` call, `max_tokens = 1500`, strict JSON out, every record through `validate_learning` (which re-derives `corroborated`, applies confidence caps, resolves evidence keys, and caps magnitude against the stored residual), invalid records dropped with a trace note. No retry loop: bounded means bounded. When `complete` returns None (no key, `VOICE_OS_OFFLINE=1`) or the output does not parse, a deterministic offline fallback derives learnings from measurements alone: axis residuals over the noise floor become stylistic-axis learnings scoped to the exact context cell, `banned_removed` becomes lexical reinforcement, `banned_introduced` becomes a never-ban proposal, an out-of-band length bucket becomes a process learning. Offline-fallback and orphan-mode sessions emit `corroborated = false` by construction: they contribute to guidance and to decay of contrary learnings, but they can never move a numeric target, because their measurement context is weaker (no directives, or no recorded `target_base`).

   Learning categories extracted: stylistic-axis (per-axis offset in a context cell), tone (advisory), lexical (banned and never-ban candidates, vocabulary), goal, audience, medium, doc_type, recipient (scoped calibration corrections), and process (workflow-level guidance such as length discipline or structure preferences).

4. **consolidate.** Write the session record (content-hash deduped), then re-aggregate the entire session store into the artifact under an exclusive lock (below). Single write path, idempotent, order-independent: sessions are explicitly iterated as `sorted(sessions, key=session_id)` before every weighted sum, so float summation order is fixed.

   The math, with every constant recorded in `params`:
   - **Decay** is calendar-time: `w = 0.5 ^ (age_days / HALF_LIFE_DAYS)` with `HALF_LIFE_DAYS = 45`, where age is measured against an explicit `as_of_date` passed into consolidation and recorded in `params`. In production `as_of_date` is the consolidation date; in offline and golden tests it is frozen. This preserves byte-stability (consolidation is a pure function of store plus `as_of_date`) while restoring the retirement story the review showed newest-session anchoring breaks: under newest-anchoring, a burst of sessions followed by silence kept months-old corrections at full weight forever.
   - **Weights and support.** Define `weight_sum = sum(w_i)` and `n_eff = sum(w_i * confidence_i)` over contributing observations, alongside raw counts `n_sessions` and `n_corrob_sessions` (distinct sessions with recomputed `corroborated = true`). Shrinkage is `lam = n_eff / (n_eff + K_REFLECT)` with `K_REFLECT = 2` (the aggressive posture, decided: two fresh confident sessions reach roughly half weight).
   - **Application floors** (all must hold before `effective_delta` is nonzero): `n_corrob_sessions >= MIN_SESSIONS` with `MIN_SESSIONS = 2`; `n_eff >= MIN_NEFF` with `MIN_NEFF = 1.5`, so one fresh session plus one decayed ghost cannot pass the count floor while carrying the weight of one (the exact defeat two models worked the arithmetic on); and at least 2 contributing sessions with `|axis_residual[subject]| >= NOISE_FLOOR`, provisionally `NOISE_FLOOR = 0.04`, to be set empirically by measuring `score_text` repeat variance during implementation (the council's 0.04-0.07 estimate is model-derived and unverified).
   - **Netting and dispersion.** Within a scope, opposite-signed learnings net arithmetically as weighted observations, but the entry also carries `internal_contested = true` when weighted positive and negative support each exceed 0.7: a cell where two strong sessions disagree must be distinguishable from a cell with no signal, and guidance emission is suppressed for internally contested cells (including the n = 1 fast path).
   - **Conflict with the slow path, both directions.** Opposing signs: when the reflection delta opposes the mined group delta for the scope by more than 0.10, the entry is `contested` and its clip halves to 0.075. The comparison uses the mined profile identified by the `mined_profile_hash` recorded at each session's draft time, not whatever mined state exists at consolidation, so a transient miner refresh cannot throttle good learnings. Concordant signs: when reflection and mined agree in sign and their combined pre-clip magnitude exceeds 0.25, the reflection delta is reduced by the overlap (the mined-absorption subtractor). `contested` only firing on disagreement would let the fast and slow paths double-count the same shipped-final signal during the mining lag, which the review flagged as the silent variant of the double-count risk.
   - **Stability governor.** Per (axis, scope): if the sign of the session-level residual flips 3 or more times in the last 5 contributing sessions, or the applied learning's post-application `override_frequency` fails to decrease over a window, the entry is marked `frozen`: it demotes to guidance-only until a harness run (or explicit human clear) lifts it. A rolling cumulative cap also holds: total applied reflection movement per (axis, scope) within any 90-day window stays within `CUMULATIVE_CAP = 0.25`.
   - **Clip.** `effective_delta` clips to `REFLECTION_DELTA_CLIP = 0.15` (0.075 when contested).

   **Write safety.** The whole consolidate, snapshot, overwrite sequence runs under an exclusive advisory lock (`var/locks/reflection_artifact.lock`, `fcntl`); the harness and `VoiceModel` artifact loads take the same lock shared, which turns the previously unenforced "no reflection during live harness runs" rule into a mechanism. Before writing, the session directory is re-scanned; if it changed since the read, consolidation restarts under the lock. The artifact is written as tmp file, `fsync`, `os.replace`, directory fsync, then reloaded and re-validated; on validation failure the snapshot is restored. The previous artifact is snapshotted to `var/reflection/artifacts/<ts>/` (with the `var/`-only consolidation report) before replacement; the last 20 snapshots are kept.

5. **record.** Summary trace note and result envelope: `{run_id, session, measurements_summary, learnings, proposals, artifact_path, active_learnings, pending_remaining, trace}`. `active_learnings` names which entries are now applied, frozen, stale, or guidance-only, so the skill can report exactly what the session changed.

### Versioning and rollback

`params.sessions_hash` plus `params.as_of_date` make every consolidation reproducible from the store. Three rollback paths, cheapest first: `python3 -m voice_os retract <session_id>` marks a session retracted and re-consolidates, surgically removing a bad learning at its source; restore any snapshot from `var/reflection/artifacts/<ts>/`; delete the artifact file entirely, upon which the read path degrades to exactly today's behavior, the same absent-artifact contract every mined artifact honors. Two kill switches short of deletion: `VOICE_OS_DISABLE_REFLECTION=1` disables the read-path layer without touching data, and `var/reflection/disabled_scopes.json` disables named (axis, scope) entries. Schema evolution follows the repo's fail-fast convention: `validate_artifact` rejects version mismatches, session records carry their schema version, and a version bump ships with a migration script for the session store.

## 6. Integration points

### Application (tiered, decided)

- **Axis deltas auto-apply behind a freshness gate.** `calibration.py::calibrate_extended` gains one final layer after the goal step: look up `axis_learnings` from the most specific matching scope down to `"global"` (a scope matches when every present key equals the context value; among matches the most keys wins, ties broken by sorted key, fully deterministic), apply `effective_delta` per axis with the existing round-and-clamp, and record `sources["reflection"]`. When the artifact is absent the output is byte-identical to today, asserted by a golden test.
- **The eval tripwire is load-bearing, not advisory** (the review's top-ranked fix, adopted). `var/eval/baseline.json` gains `last_eval_sessions_hash`. When `artifact.params.sessions_hash != last_eval_sessions_hash`, the reflection layer demotes to guidance-only for that draft and records `sources["reflection"] = "stale"`. Auto-apply resumes when an offline harness run refreshes the baseline. This preserves the no-shadow-period decision (the first eval after the first consolidation activates the layer immediately) while making unbounded auto-apply stacking between evals structurally impossible: with `MIN_SESSIONS = 2` and `K_REFLECT = 2`, two sessions in one morning can legitimately move an axis 0.075, and nothing else in the design forces a judge to look before a third session compounds it.
- **Auto-apply runs from day one** within that gate. The floors, caps, governor, decay, and rollback paths carry the rest of the safety load.
- **Guidance strings** append through the existing `VoiceModel.query` guidance path into persona prompts. Zero graph changes.
- **Lexical stays propose-only.** `banned_add` and `never_ban_add` phrases live in `var/reflection/proposals.json`, surface via `QueryResult.meta["reflection_proposals"]` (sibling to `drift_flags` and `evolution_flags`) and `python3 -m voice_os reflect-proposals`. The human edits `data/banned_list.txt` and `data/never_ban.txt` as today; `VoiceModel.banned` remains the single merge point.
- **Recipient and doc_type scopes** are schema-ready immediately and become active the moment Phase 0 items 3 and 4 land.

### Reuse map

| Need | Reuse |
|---|---|
| Axis scoring of both texts | `axes.py::score_text`, `AXES` |
| Tone metrics on both texts | `tone.py::tone_signals` / `derive_metrics`, `TONE_METRICS` |
| Banned hit sets | `qa.py::find_banned` over `VoiceModel.banned` |
| Per-side pattern profiles | `evolution/patterns.py::extract_pattern_profile` (not `diff_profiles`) |
| Draft recovery fallback by run_id | the `SqliteSaver` retrieval in `product/graph.py::run_history` |
| Content-hashed session store | structural clone of `evolution/baselines.py` (`content_hash`, `ensure_baseline`) |
| The one LLM call | `llm.py::complete` (honors `VOICE_OS_OFFLINE`, returns None to trigger fallback) |
| Artifact envelope and load | `mined.py::validate_artifact`, `ARTIFACT_FILES`, a `MinedArtifacts` field |
| Application site | `calibration.py::calibrate_extended`, new final layer |
| Surfacing | `QueryResult.meta` and `guidance` in `model.py::VoiceModel.query`; prepare-node trace notes |
| Graph skeleton | `evolution/graph.py` end to end |
| CLI conventions | `voice_os/product/cli.py` (stdin/file input, exit codes, `_emit`) |

### Evaluation harness

Reflections are measurable, and fidelity claims are held to the honest number:

- **A/B with zero harness changes**: the harness already accepts `--mined-dir`, so runs with and without `reflection_learnings.json` isolate the artifact's effect. `alignment_judged` is the number that decides whether reflection helps; `alignment_offline` is the regression floor, never the evidence.
- **The freshness gate** (above) is the enforcement mechanism; `reflect --status` reports "learnings changed since last eval, axis learnings currently demoted" until the harness runs.
- **No auto-runs**: the reflection graph never invokes the harness. Eval spend stays an explicit decision; the cost of skipping it is demotion, not corruption.
- **Provenance is automatic**: the artifact lands in `MinedArtifacts.meta`, which `prepare` already copies into every envelope's `provenance.artifacts`, so every draft records which learnings version shaped it.
- **Locking replaces the mutation comment**: the harness takes the artifact lock shared, so a live run and a consolidation physically cannot interleave.

## 7. Implementation considerations

- **LangGraph**: the five-node linear graph above, compiled with its own `SqliteSaver` over `var/reflection.sqlite`. `graph.py` is the only langgraph import in the package; `import voice_os` stays stdlib-only through the lazy-export machinery. Fixed node count means no recursion-limit exposure.
- **Persistence and versioning**: JSON everywhere, standard artifact envelope, content-hash identity, timestamp ids that are run-scoped while sha256 is the cross-run identity. `generated_at` and `as_of_date` are excluded from content-hash computation (asserted by test: `generated_at` inside a hashed envelope would otherwise make every re-consolidation a "new" artifact). Var-dir resolution follows the uniform precedence: explicit argument, then `VOICE_OS_VAR_DIR`, then repo-anchored `var/`.
- **Privacy and data minimization**: draft text, final text, directive verbatims, lexical proposal phrases, session id lists, and raw weighted means live only under gitignored `var/reflection/`. The artifact under `corpus/mined/` carries schema fields, vocabulary-validated scope values, quantized deltas, and whitelisted-slot template strings, and is marked `"export": "never"`: the review's judgment, adopted here, is that even prose-free behavioral deltas are a high-entropy fingerprint of the user, so the structured-fields-only invariant is necessary but not sufficient the moment a file is synced or shared. The Reflector prompt sees measurements, directives, and capped excerpts, never full documents. Nothing reflection produces is committed. Session records are kept indefinitely (decided): the privacy boundary is location, not lifespan, and full history keeps every consolidation reproducible and every learning retractable.
- **Determinism** (per `docs/determinism.md` rules for new modules): all measurement and consolidation math is RNG-free with explicitly sorted iteration; decay takes an explicit `as_of_date` recorded in `params`, frozen in offline tests; `SequenceMatcher(autojunk=False)` over NFC-normalized input pins `edit_magnitude` across platforms; the offline path is byte-stable and covered by the determinism invariant test before merge; live Reflector output is inherently non-deterministic and is contained by schema validation, deterministic re-derivation of gating fields, session immutability, and run-scoped `reflector.mode/model/prompt_version` fields, the same treatment as `provenance.live_model`.
- **Testing** (beyond the standard suite): a golden asserting artifact-absent byte-identity of `calibrate_extended`; a golden permuting session input order and asserting artifact-hash stability; a property/fuzz test feeding adversarial-but-schema-valid records through `validate_learning` and asserting clean accept-or-reject, never a mid-consolidation crash; a crash-recovery test (kill between session write and artifact replace, assert the lock plus `in_progress` recovery restores consistency); an OSError-path test for the pending write; and a retraction-equivalence test (retract then re-consolidate equals consolidating without the session).
- **Bounding and cost**: per session, exactly five node executions, at most one LLM call at `max_tokens = 1500` (roughly 600 tokens of measurement JSON plus up to about 1,300 tokens of directives and excerpts, on the order of $0.05 to $0.10 live, $0 offline), O(len(draft) + len(final)) regex passes, and O(total learnings) consolidation math. Storage grows by two texts and a small JSON per session, deduped by content hash; artifact snapshots prune to the last 20.

### Departures from repo precedent (flagged deliberately)

1. `draft()` gains a side effect (the pending-record write, now carrying draft text and both targets) and an envelope key. It is the smallest change that makes the loop start automatically and crash-proof, and it cannot fail a draft.
2. Numeric auto-application from a non-corpus source. Evolution is propose-only; reflection applies axis deltas. Justification: clipped tighter than the mined layer (0.15 versus 0.35), corroboration re-derived deterministically, dual support floors, governor-frozen on instability, gated on eval freshness, decayed on calendar time, and reversible five ways. Lexical stays propose-only, preserving evolution's invariant where a single bad learning would be most visible.
3. `K_REFLECT = 2` versus `SHRINKAGE_N = 50`: different observation unit, documented constant.
4. Artifact history snapshots under `var/reflection/artifacts/`: mined artifacts are otherwise overwritten in place; auto-application demands cheap rollback.
5. A file lock shared with the harness: the first cross-subsystem runtime coordination mechanism in the repo, replacing a documentation-only rule the review showed was the sole protection on the artifact path.

## 8. Risks and mitigations

1. **Overfit to the last session.** Mitigations: `MIN_SESSIONS = 2` counted over corroborated sessions, `MIN_NEFF = 1.5` so decayed ghost sessions cannot satisfy the floor, the noise floor on contributing residuals, `K_REFLECT` shrinkage, 45-day calendar decay, the 0.15 clip, and null-edit reinforcement pulling toward "current calibration is right". The aggressive posture (`K_REFLECT = 2`, half-life 45 days) trades noise immunity for convergence speed; the freshness gate is the backstop when a noisy fortnight steers too hard.
2. **Reflection-to-generation feedback loop.** Primary defense: residuals are measured against `target_base`, a reference the loop's own artifact cannot move, so axis learnings are convergent estimates of a fixed offset rather than compounding increments. The review rejected the earlier moving-target formulation as unstable under `score_text` noise and axis coupling, and the literature on self-refinement against imperfect evaluators backs that rejection. Secondary: the stability governor (sign-flip freeze, override-frequency monitor), the cumulative per-cell cap, the clip, decay, and the freshness gate.
3. **Execution failure masked as convergence.** When the generator persistently under-hits a correct target, preference residuals read zero while drafts stay wrong. `execution_residual` is measured separately and routed to diagnostics and process guidance, so the loop cannot go blind to it.
4. **Double-counting with the slow path.** Shipped finals eventually re-enter the corpus. Mitigations: the contested check (against the draft-time mined snapshot) for opposing signals, the concordant-absorption subtractor for agreeing ones, calendar decay retiring fast-path learnings, and the invariant that reflection never writes to `corpus/chunks/` or triggers ingestion.
5. **LLM-invented learnings.** The reflector's output is advisory: `corroborated`, confidence caps, magnitude caps, and evidence validity are all re-derived deterministically from the measurement block, and records that overclaim are dropped, not capped. An uncorroborated learning can never become an applied numeric delta; offline-fallback and orphan sessions are barred from numeric application by construction.
6. **Data corruption under concurrency or crashes.** Exclusive lock across consolidate-snapshot-write, shared lock on all readers including the harness, atomic tmp-fsync-replace writes with post-write validation and snapshot fallback, two-phase pending lifecycle, and immutable session records.
7. **Privacy leakage past the invariant.** Quantized deltas, hashed-out session ids, phrases confined to `var/`, whitelisted guidance slots, bucketed length ratios, the `"export": "never"` marker, and a CI scan for prose leakage into the artifact.
8. **Queue rot and skill non-compliance.** TTL lapse to a recoverable directory, `--status` exit-code nagging, per-draft pending-count trace notes, surfaced pending-write failures, and rate limits on `--accepted-as-is`. Accepted residual risk: a draft consumed entirely outside the skill can lapse silently; that is the honest boundary of what software can guarantee here.
9. **Noisy or contradictory directives.** Directives are capped, recorded verbatim in the session only, and enter learnings by index reference with deterministic confidence caps; contradictions net out at consolidation and mark the cell `internal_contested`, which also suppresses its guidance.
10. **Cost creep.** The loop is structurally bounded (one call, no retries, no sub-generation) and fully functional offline.

## 9. Success metrics

The review's sharpest meta-finding: the obvious dashboard is blind to this subsystem's worst failure modes. Every headline metric below therefore pairs with the instrumentation that distinguishes its good reading from its pathological one.

- **Voice fidelity (the honest number)**: `alignment_judged` trend per context cell rises over successive harness runs with reflection active; A/B runs via `--mined-dir` show the artifact's isolated contribution. `alignment_offline` never regresses past the gate tolerances. Guard against judge-preferred collapse (a rising judge score for a narrowing voice): track the variance of axis positions across recent drafts per scope, compare periodically against a frozen pre-reflection holdout slice, and schedule occasional blinded human preference checks.
- **Convergence indicators**: null-edit rate rises, `edit_magnitude` (length-normalized) falls, and repeated directives fall toward zero. Guards: correlate null-edit rate with session-close latency (rising null-edits with falling latency is convergence; with rising latency it is abandonment), and read `edit_magnitude` only in its normalized form so shorter drafts cannot fake improvement.
- **Applied-learning efficacy**: the `override_frequency` stat (post-application residuals still exceeding the noise floor) falls for each applied entry; entries where it does not fall get frozen by the governor. This is the direct "did the learning land" measurement the headline metrics cannot see.
- **Scorecard**: the three-way fidelity comparison in `eval.py` gains a fourth arm (reflection-applied) with the expected ordering `baseline_only < hand_calibrated < extended_mined <= reflection_applied` on held-out text.
- **Two-path health**: contested and concordant-absorption events logged with raw pre-clip magnitudes of both layers, so "low contested rate" is distinguishable from "reflection has drowned out mined" or "both wrong in the same direction".
- **Operational health**: pending queue near zero with low lapse rate and zero unexplained `pending_write_failed` counts; `frozen` entries rare and investigated; retractions rare and effective; every consolidation that changes an active delta followed by its harness run (enforced by the gate, observed in `--status`).

## 10. Decisions and open questions

### Resolved (2026-07-08, design review with Mitchell; hardening adopted same day from the six-model council review)

1. **Application tier**: tiered. Numeric axis deltas auto-apply; lexical changes stay human-confirmed proposals.
2. **Rollout**: auto-apply from day one, no shadow period, behind the eval-freshness hard gate (stale `sessions_hash` demotes axis learnings to guidance-only until an offline harness run refreshes the baseline).
3. **Directive capture**: hybrid. Exact words pass through as `stated` directives; intent shown only through edits is summarized as `inferred` directives with lower confidence caps.
4. **Sequencing**: Phase 0 lands before the engine, one measured PR at a time, so early learnings correct a target already using all signal and attribution stays clean.
5. **Constants posture**: aggressive. `K_REFLECT = 2`, `HALF_LIFE_DAYS = 45`, clip 0.15, `PENDING_TTL_DAYS = 14`, plus the hardening floors `MIN_NEFF = 1.5`, `NOISE_FLOOR = 0.04` (provisional, to be measured), `CUMULATIVE_CAP = 0.25`. Tune only against harness A/B evidence.
6. **Advisory layers**: tone and pattern-profile conformance both stay advisory in v1; neither shifts numeric targets nor gates.
7. **Retention**: session records are kept indefinitely under `var/reflection/`. The privacy boundary is location, not lifespan; full history keeps every consolidation reproducible and every learning retractable.
8. **Stability architecture** (from the council review): residuals measured against the reflection-excluded `target_base`; `execution_residual` tracked separately; all application-gating fields re-derived deterministically; locking plus atomic writes on the artifact path; calendar-time decay via explicit `as_of_date`.

### Still open

1. **Tone numeric application.** Revisit once harness `tone_mae` can measure a numeric tone layer's effect in isolation.
2. **Pattern-profile fusion gating.** Phase 0 item 1 starts advisory, like tone. Whether pattern conformance should ever gate is a separate, later decision.
3. **`NOISE_FLOOR` calibration.** The 0.04 value is provisional; implementation includes a repeat-variance measurement of `score_text` on held-out singles to set it empirically.
