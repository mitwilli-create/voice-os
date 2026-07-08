# Live Alignment: Closing the Offline-vs-Judged Gap

Status: design + implementation record for the two-PR alignment track
(exemplar fusion + length budget, then gate calibration). Each PR is
measured by a live harness run before the next lands.

## Problem

The two live harness runs to date (pre and post em-dash scrub, same 18
held-out cases, claude-opus-4-8) measured `alignment_judged` 0.560
against `alignment_offline` 0.706, with a pass rate of 0.333: 12 of 18
drafts failed the 0.80 QA gate after exhausting all revisions. The
offline gate was guarding a number that did not track judged quality
(gate fidelity vs style-vs-real correlation: 0.18).

## Root causes (verified against the persisted report and the code)

1. **The live persona had never seen a single example of the author's
   writing.** `VoiceModel.query` computes ranked held-in exemplars
   (`voice_os/model.py`, documented as feeding live prompts) but nothing
   consumed them: not the product graph's `prepare` node, not
   `personas.py`. The live model wrote from six axis numbers plus a
   banned list.
2. **The gate demands more target-conformity than the author's own
   writing has.** Real held-out text scores mean 0.7909 against the same
   calibrated targets the gate uses (peer audience: 0.6543), yet
   `PASS_THRESHOLD` is 0.80. All 12 failures hit max revisions; several
   failing drafts scored 0.86-0.89 against the actual real message while
   the modeled target failed them.
3. **Live drafts are inflated.** Failing cases ran 1.2x-3.1x the real
   message length; `length_ratio` correlates -0.60 with the judge's
   `same_author` score. Nothing in the persona prompts or gate signals
   controlled length.

A fourth finding is deliberately parked: the judge sees exactly one
short real message as its authorship anchor. Anchoring it with held-in
exemplars would be fairer, but changing the measuring instrument
mid-experiment would break before/after comparability. It stays parked
until both PRs below are measured with the current judge.

## PR 1: exemplar fusion + length budget (`fix/live-exemplar-fusion`)

Give the live persona the voice, and a length target, without touching
any locked legacy surface.

- **State**: `VoiceState` gains `exemplars: list[dict]`
  (id/text/tier/fit, JSON-safe), seeded empty by `initial_state`.
- **Prepare**: copies the top 3 of `QueryResult.exemplars` into state.
  These are held-in tier 1/2 chunks, holdout-filtered upstream in
  `VoiceModel._exemplars`, so evaluation never scores text the persona
  saw as an example.
- **Personas**: `GenerativePersona.revise` takes additive
  `exemplars=None` and `length_target_words=None` keywords.
  `_profile_block` gains an "Examples of this author's real messages in
  this context" section and a length instruction when provided, and is
  byte-identical to the pre-change output when neither is passed
  (locked by a stability test), so `run_cycles` / `run_pipeline` and
  their goldens are untouched.
- **Length budget**: `generate` and `revise` pass the input's word
  count (the brief derives from the real message, so its length is the
  author's length for the situation). The `qa_gate` node additionally
  appends an advisory revision signal when the draft exceeds ~1.4x the
  input word count; it never changes the gate decision.
- **Offline paths ignore both**: the regex persona and the fixture
  golden path (`chunks_dir=None`) see no exemplars, keeping offline
  determinism and goldens byte-stable.

Privacy: exemplar text is personal data. It already feeds live prompts
by documented intent (`model.py`); with this change it also lands in
product checkpoints under the gitignored `var/` directory, the same
data class as the draft text those checkpoints already contain.

Expected live observables: `length_ratio` mean toward 1.0-1.3,
`alignment_judged` up from 0.560, pass rate up from 0.333, em-dash rate
stays 0.0.

**Measured (live run eval-20260708T053843Z-9143b1ad, 18 cases,
claude-opus-4-8, all 18 confirmed receiving 3 exemplars via
checkpoints):** length_ratio 0.9591 (inflation eliminated; was
1.2x-3.1x on failing cases), em_dash 0.0, banned 0.0. alignment_judged
0.5589 (flat vs 0.560), judge same_author 1.67 (flat vs 1.72), pass
rate 0.2222 (down from 0.333; drafts changed shape, the 0.80 gate did
not). Honest reading: the length root cause is closed, exemplar fusion
did not move the judge, and the flat judged score with a mean
same_author of 1.67 strengthens the parked judge-anchoring hypothesis
(one short real message is a weak authorship anchor). The gate
overdemand (root cause 2) remains the live blocker on pass rate and is
addressed by PR 2 below.

## PR 2: gate calibration (`fix/gate-calibration`)

A threshold above what real text scores in a cell makes the gate reject
drafts for being no more target-conformant than the author himself.

- Emit per-(channel, audience) real-text fidelity percentiles
  (p25/p40/p50, n) into `corpus/mined/gate_calibration.json`,
  envelope-wrapped like the other mined artifacts and loaded tolerantly.
- Split choice: the miner measures the held-in split, not the held-out
  scorecard pass the gap analysis quoted. Mining on holdout would break
  the repo invariant that miners see only the train side (mine/weights
  docstring) and would let the gate learn from the exact chunks the
  scorecard measures. Held-in gives roughly 4x the sample per cell; the
  generation run that produces the artifact quotes both splits'
  percentiles so the (small) difference is on record.
- The product graph's `qa_gate` uses a per-cell threshold of
  `clamp(cell p40, 0.65, 0.80)` only when the artifact exists and the
  cell has n >= 50; otherwise the 0.80 default holds everywhere.
  Fixtures ship no artifact, so goldens are untouched.
- The real-corpus offline baseline moves (earlier passes mean fewer
  revision cycles); the move is made explicitly via
  `python3 -m voice_os.harness gate --update-baseline` with before and
  after quoted in the PR body.

Expected live observables: pass rate approaches the share of drafts the
judge rates 3+, gate-vs-judge disagreement shrinks, judged does not
regress.

**Measured (live run eval-20260708T060442Z-8b0af212, same 18 cases):**
pass rate 0.2222 to 0.6667, alignment_judged 0.5589 to 0.6338, judge
same_author 1.67 to 2.22, length_ratio 0.9523 holds, em_dash 0.0.
Judged improved alongside the gate change: calibrated thresholds mean
fewer forced revision cycles, so drafts keep more of their natural
voice instead of being over-revised toward the modeled target.

## Measurement 3: KB fusion (feat/kb-fusion, PR #21)

The compact KB's distilled voice patterns joined the live prompts
(docs/kb-fusion.md). **Measured (live run
eval-20260708T062409Z-290f9e36, same 18 cases):** alignment_judged
0.6338 to 0.6664, pass rate 0.6667 to 0.7222, chat judged 0.7725,
em_dash and banned 0.0.

Campaign trajectory across the three measured changes (18 held-out
cases, claude-opus-4-8 personas and judge): alignment_judged 0.560 to
0.5589 to 0.6338 to 0.6664; pass rate 0.278 to 0.2222 to 0.6667 to
0.7222; length_ratio from 1.2x-3.1x inflation on failures to ~0.95
throughout. The remaining judged headroom most plausibly sits in the
parked judge-anchoring weakness and in corpus breadth (the peer cell
scores lowest and had the thinnest evidence at measurement time).

## Out of scope

- Judge exemplar-anchoring (parked, above).
- Any change to the six axes or the judged composite weights.
- Thread-context ingestion (true reply tasks).
