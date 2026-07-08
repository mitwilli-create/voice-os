# Pattern Fusion: Mined Tier 1 Pattern Profile in Live Persona Prompts

Status: implementation record for `feat/pattern-fusion`. Phase 0 item 1
of docs/voice-reflection-engine.md (activate dormant signal), modeled
end to end on the KB fusion PR (docs/kb-fusion.md).

## Problem

The evolution module extracts a Tier 1 pattern profile on every drift
run: greeting and sign-off distributions matched against fixed
lexicons, casual-marker rates per 100 words, sentence-length shape
(`voice_os/evolution/patterns.py::extract_pattern_profile`). Until now
that profile existed only for drift detection: it was diffed against
the stored baseline, reduced to emerging/fading flags, and surfaced as
trace notes. The distributions themselves, the most direct measurement
the system has of how the author actually opens, closes, and paces
messages today, never reached generation. The live persona drafted from
axis numbers, a banned list, exemplars, and KB guidance while the
corpus-mined pattern statistics sat unread in the drift layer.

## What changed

1. **The drift graph persists the profile.** The `flag` node in
   `voice_os/evolution/graph.py` now writes the run's current Tier 1
   profile into the `evolution_flags` artifact as
   `data.pattern_profile`, beside the existing flags. The artifact is
   written by its own graph, loaded by the standard
   `mined.py::load_artifacts` path, and the schema change is additive:
   version stays 1.0, and artifacts that predate the field simply
   yield no pattern guidance.
2. **A distiller renders it as guidance.**
   `voice_os/product/fusion.py::distill_pattern_guidance` turns the
   profile into at most 6 fixed-template plain-English lines (at most
   120 words total): top greeting forms with their message shares, top
   sign-off forms, casual-marker rates per 100 words, sentence shape
   (mean, median, p90), short-message rate, and exclamation rate.
3. **The prepare node fuses it.** `prepare` in
   `voice_os/product/graph.py` distills
   `model.mined.evolution_flags["pattern_profile"]` into a new
   `pattern_guidance` state field and records a trace note with the
   line and word counts when non-empty.
4. **Personas render it additively.** `GenerativePersona.revise` takes
   an additive `pattern_guidance=None` keyword; `_profile_block` emits
   an "Observed voice patterns mined from the author's recent writing:"
   section only when provided, with every line nested under the header
   (data, not instructions, the same delimiting as the KB and exemplar
   blocks). With the kwarg absent the block is byte-identical to
   before, locked by the profile-block stability test, so `run_cycles`,
   `run_pipeline`, and their goldens are untouched. The offline regex
   persona and `AdversarialPersona` ignore it entirely.

## Privacy: the lexicon is the whitelist

Template slots are validated against the fixed evolution lexicons, the
same boundary that keeps names out of stored baselines: the distiller
iterates `GREETING_LEXICON`, `SIGNOFF_LEXICON`, and `DEFAULT_MARKERS`
rather than the artifact's keys, so free vocabulary, including the
"other" bucket, can never render, even from a hand-edited or corrupted
artifact. Everything else in a rendered line is a template literal or
a number. This satisfies the invariant stated in
docs/voice-reflection-engine.md section 3 item 1. The distilled lines
enter live persona prompts and run checkpoints under gitignored `var/`,
the same destinations KB guidance already reaches.

## Bounds and floors

- `PATTERN_GUIDANCE_MAX_ITEMS = 6`, `PATTERN_GUIDANCE_MAX_WORDS = 120`:
  the template set is closed, every line is single-line by
  construction, and the word budget counts exactly the text that
  renders (same bounding idea as KB fusion).
- `PATTERN_MIN_CHUNKS = 50`: a profile mined from fewer chunks is too
  thin to state percentages as the author's habits and distills to
  nothing.
- Per-form floors reuse the evolution diff constants (`MIN_SUPPORT = 5`
  raw occurrences, `NEAR_ZERO = 0.01` share or per-100w rate), so
  fusion and drift agree on what "present" means.
- Non-finite, boolean, negative, or mistyped values are skipped, never
  rendered: NaN and Infinity survive `json.load` and would otherwise
  propagate through `round()` into prompts.

## Activation

The signal turns on when the artifact carries a profile: run
`python3 -m voice_os.evolution drift-run` to regenerate
`corpus/mined/evolution_flags.json` with the `pattern_profile` field.
Phase 0 item 5 (baseline and miner refresh over the full 143k-chunk
corpus) supersedes this with a `--update-baseline` run. Until then,
absent-field behavior is byte-identical to today, asserted by the
golden locks.

## Measurement (offline A/B, 2026-07-08)

Same instrument as the repo's other fusion changes: the offline harness
via `--mined-dir` isolation, 38 held-out cases.

- Arm A: `corpus/mined` as of this PR (no `pattern_profile` field).
- Arm B: a copy of the same directory with `pattern_profile` added,
  computed by `extract_pattern_profile` over the current Tier 1 texts
  (6,948 chunks; distills to 5 lines, 67 words).
- Result: the two summaries are byte-identical
  (`alignment_offline 0.9399`, `style_overall 0.903`, `pass_rate
  0.5789`, `em_dash_rate 0.0` in both arms). This is the expected
  regression-floor reading, not the efficacy reading: offline personas
  ignore prompt guidance by design, so the offline A/B proves the
  change cannot regress the gate while a per-draft trace check confirms
  the B arm actually fuses (`prepare: pattern guidance fused
  (5 patterns, 67 words)`).

The decisive number is live: per the kb-fusion measurement discipline,
the live harness (same 18-case judged instrument, `alignment_judged`
and `same_author`) runs before and after this lands in live use, and
that pair is quoted when the fusion is evaluated. Expected observables:
greeting and sign-off choices in failing drafts stop diverging from the
author's measured distributions; judged alignment moves up or holds. If
judged alignment regresses, the fusion is feature-flagged off by
reverting the two `pattern_guidance=` pass-throughs in
`product/graph.py` (the distiller, state field, and artifact field can
stay).

Pattern conformance stays advisory in v1 (decided,
docs/voice-reflection-engine.md section 10): these lines steer the
persona; nothing here gates or shifts numeric targets.
