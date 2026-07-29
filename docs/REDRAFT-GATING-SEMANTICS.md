# Redraft gating semantics: what `decision: pass` actually means

Established 2026-07-28. Source-verified against `voice_os/product/graph.py`,
`voice_os/product/cli.py`, `voice_os/product/state.py`, `voice_os/model.py`,
`voice_os/mined.py`, and `voice_os/qa.py` at that date.

## If you are an agent told to "gate on decision: pass", read this first

`python3.11 -m voice_os draft --redraft --file <doc>` does **not** validate your
document. It rewrites it and grades its own rewrite.

The envelope's top-level `decision`, `fidelity`, and `banned_hits` describe
Voice OS's generated `output_text`, never the file you passed in. A handover
that says "loop until Voice OS returns `decision: pass`" is a category error:
you are not measuring whether the human-authored letter is good, you are
measuring whether the machine's replacement for it is good enough to adopt.

Following that instruction literally gets you one of two bad outcomes:

1. an unbounded loop, because the input is never what is being scored, so
   editing the input does not reliably move the number; or
2. adopting a rewrite that silently dropped content the original had. An
   earlier run on this same letter dropped a verbatim JD quote, caught only by
   the `conservation.quote_violations` field.

Skip to [Correct usage](#correct-usage) for what to run instead.

## The semantics

### The graph always rewrites

`voice_os/product/graph.py:638` builds the pipeline as:

```
START -> prepare -> generate -> critique -> qa_gate -> {pass|reject: END, revise: revise -> critique -> ...}
```

`generate` (graph.py:409) unconditionally calls `GenerativePersona.revise()` on
`state["input_text"]` and writes the result to `current_draft`. There is no
conditional edge that skips it. Every run produces a rewrite before anything is
scored.

The one exception is the short-input guard, `_short_input_guard` (graph.py:371):
for inputs under `_SHORT_INPUT_WORDS = 25` words the original can be retained
verbatim (and `conservation.input_retained` goes true). A cover letter is
hundreds of words, so this never fires for application materials.

### What `qa_gate` scores

`qa_gate` (graph.py:451) opens with `draft_text = state["current_draft"]` and
uses that value, and only that value, for:

- `score_text(draft_text)` -> the six axis scores -> `fidelity`
- `find_banned(draft_text, state["banned"])` -> `banned_hits`
- `moves.detect(draft_text)` -> `signature_moves.detected`

`state["input_text"]` appears in exactly one place in that function:
`conservation.check(state["input_text"], draft_text)`.

So in the envelope built by `state.py:163 build_result`:

| field | describes |
|---|---|
| `decision` | the rewrite |
| `fidelity` | the rewrite |
| `banned_hits` | the rewrite |
| `signature_moves` | the rewrite |
| `output_text` | the rewrite |
| `conservation` | the relationship between input and rewrite |

`conservation` is the only field carrying information about your input.

### What `--redraft` changes (and does not)

`--redraft` does **not** switch the tool into a validate-only mode. It only
changes which conservation findings are blocking:

- `redraft=True`: `conservation.unsupported_sentences` block a pass (graph.py:539-543).
  Also, inside the sub-25-word guard, an unentailed or insufficiently improved
  rewrite is discarded in favor of the input (graph.py:394-406).
- `redraft=False` (compose): conservation is still measured and reported, never
  blocking on unsupported sentences.

In both modes `conservation.quote_violations` always block a pass, and an
`--avoid`ed signature move always blocks a pass.

The rewrite happens either way. The scoring target is the rewrite either way.

### There is no validate-only CLI path

Checked: `voice_os/product/cli.py` exposes exactly three subcommands, `draft`,
`history`, `graph`. No `--check`, `--validate`, `--score-only`, or dry-run flag
exists. `draft` is the only path into the graph, and the graph always generates.

Validate-only capability exists in the **library**, not the CLI:
`VoiceModel.gate_draft(draft, query)` (`voice_os/model.py`) scores an arbitrary
string against a context with no persona, no LLM call, and no cost. The
root-level `score.py` also scores a file without rewriting, but only against the
raw corpus baseline with no register calibration and no banned-phrase check.

## Evidence

Run `gate2-3513-2026-07-28` on
`~/Documents/career-ops/apply-pack/3513-elevenlabs-writer-and-editor/cover-letter.md`
returned `banned_hits: ["i would welcome the"]`.

That phrase does not occur in the input file (`grep` returns no match, exit 1).
It occurs only in the run's own `output_text`, in a closing sentence Voice OS
generated: "This is that work, and I would welcome the chance to make the case
in person."

Running the validate-only path on the same file, same context (`doc|external`),
offline and free:

```
banned list size: 916
INPUT banned hits: []
INPUT fidelity: 0.741
per-axis: rhetorical_pace 0.406 · risk_tolerance 0.920 · sentence_rhythm 0.418
          escalation_pattern 0.912 · hedging_behavior 0.843 · editorial_register 0.950
```

The human-authored input is clean on banned phrases and scores 0.741, comfortably
above the calibrated `doc|external` threshold of 0.65. The failing artifact was
the machine's rewrite, not the letter.

`"i would welcome the"` is a **mined** banned phrase, not a hand-curated one. It
is absent from `data/banned_list.txt` (40 lines) and present in
`corpus/mined/ngram_banned.json` (888 entries, `mine.ngrams@1.0`, generated
2026-07-10) with `n: 4, contrast_count: 7, log_odds: 6.13`. The effective banned
list at runtime is the union of both, 916 phrases
(`VoiceModel.banned`, model.py: hand list first, mined n-grams appended).

## Threshold calibration (correcting a common misreading)

`PASS_THRESHOLD = 0.80` in `voice_os/qa.py:14` is the **default and the
ceiling**, not the threshold that actually runs in the graph.

`prepare` calls `_cell_threshold` (graph.py:189), which reads the p40 percentile
for the `channel|audience` cell from `corpus/mined/gate_calibration.json`,
requires `n >= 50`, and clamps the result into
`[_GATE_THRESHOLD_FLOOR 0.65, _GATE_THRESHOLD_CEILING 0.80]`. That value
overrides `PASS_THRESHOLD` for the run. `PASS_THRESHOLD` applies only when the
cell is unknown, thin, or malformed.

Note the artifact stores **p40**, and the threshold uses p40, not the mean.
Both are shown here because the means are often quoted:

| cell | n | corpus mean | p40 | effective threshold |
|---|---|---|---|---|
| chat\|friend-family | 88187 | 0.8123 | 0.816 | 0.80 (clamped to ceiling) |
| doc\|peer | 247 | 0.6983 | 0.683 | 0.683 |
| text\|friend-family | 900 | 0.6631 | 0.664 | 0.664 |
| email\|job-seeking | 332 | 0.6723 | 0.659 | 0.659 |
| social\|external | 9650 | 0.6464 | 0.630 | 0.65 (clamped to floor) |
| email\|friend-family | 6499 | 0.6285 | 0.584 | 0.65 (clamped to floor) |
| doc\|external | 503 | 0.6345 | 0.617 | 0.65 (clamped to floor) |
| email\|peer | 7771 | 0.6028 | 0.575 | 0.65 (clamped to floor) |

The takeaway stands either way: Mitchell's own corpus, in document and email
registers, sits well below 0.80. A flat 0.80 gate would reject most of his real
writing. That is precisely why the calibrated per-cell threshold exists, and why
quoting 0.80 as "the bar" overstates it by 0.15 for document work.

One trap: `VoiceModel.gate_draft()` does **not** apply the calibrated cell
threshold. It defaults to `PASS_THRESHOLD` 0.80. In the run above it therefore
reported `decision: cycle` on a 0.741 input that the graph's calibrated 0.65
would have passed. If you use `gate_draft` for validation, compare the fidelity
number against `_cell_threshold` yourself, or pass the threshold explicitly.

## Correct usage

### Gate a finished, human-authored document

Do not use the top-level `decision`. Do this instead, in three parts. It runs
offline, calls no model, and costs nothing.

**1. Score the input and check it against the full 916-phrase banned list.**

```bash
cd /Users/mitchellwilliams/Documents/voice-os
python3.11 - "$FILE" <<'PY'
import sys
from voice_os.model import VoiceModel
from voice_os.qa import find_banned, gate_extended
from voice_os.axes import AxisProfile, score_text
from voice_os.product.graph import _cell_threshold, _LOAD_DEFAULTS

CHANNEL, AUDIENCE, SITUATION = "doc", "external", "standard"

m = VoiceModel.load(
    _LOAD_DEFAULTS["corpus_path"],
    chunks_dir=_LOAD_DEFAULTS["chunks_dir"],
    mined_dir=_LOAD_DEFAULTS["mined_dir"],
    banned_path=_LOAD_DEFAULTS["banned_path"],
)
text = open(sys.argv[1], encoding="utf-8").read()
q = m.query(channel=CHANNEL, audience=AUDIENCE, situation=SITUATION)
threshold = _cell_threshold(m.mined.gate_calibration, CHANNEL, AUDIENCE) or 0.80
hits = find_banned(text, q.banned)
result = gate_extended(
    score_text(text), m.baseline, q.target_profile, hits, threshold=threshold
)
print("threshold      :", threshold)
print("input fidelity :", round(result.fidelity, 4))
print("banned hits    :", hits)
print("per axis       :", {k: round(v, 3) for k, v in result.per_axis.items()})
print("VERDICT        :", "pass" if result.fidelity >= threshold and not hits else "below bar")
PY
```

Set `CHANNEL`/`AUDIENCE` to the cell that matches the artifact: cover letters and
other application documents are `doc|external`; job-search email is
`email|job-seeking`.

**2. Only if you also want a rewrite**, run the CLI, and read `conservation`,
not `decision`:

```bash
cd /Users/mitchellwilliams/Documents/voice-os
VOICE_OS_PROVIDER_POLICY_ENABLED=true \
VOICE_OS_ALLOW_DEGRADED=true \
VOICE_OS_ALLOWED_PROVIDERS=anthropic,openai,google,xai \
python3.11 -m voice_os draft --redraft --channel doc --audience external \
  --file "$FILE" > /tmp/vos-run.json
```

Then treat the output as a *candidate* to accept or discard:

- `conservation.quote_violations` non-empty -> discard the rewrite. It edited a
  quoted span (a verbatim JD quote, a named person's words).
- `conservation.unsupported_sentences` non-empty -> discard or hand-edit. The
  rewrite invented claims the input does not support.
- `conservation.dropped_modifiers` -> the rewrite lost qualifiers; check each.
- `banned_hits` non-empty -> a defect **in the rewrite**. It is a reason not to
  adopt the rewrite. It is never evidence about your input.

Live runs cost real money and take minutes. If all you need is a verdict on
finished writing, part 1 alone is the whole job.

**3. Never** loop "edit the input, re-run `draft`, wait for `decision: pass`".
The number you are chasing belongs to a different document each iteration.

### Check any file against the banned list on its own

The runtime list is the union of two sources:

- `data/banned_list.txt`: 40 hand-curated lines, committed, comments allowed
  (`#`). Literal phrases only: `find_banned` applies `re.escape`, so regex here
  would not work. This file is also the source of truth for the generated Vale
  rule in stack-ops.
- `corpus/mined/ngram_banned.json`: 888 mined n-grams, gitignored, regenerated
  by `mine.ngrams`. Loaded by `voice_os/mined.py`, merged in `VoiceModel.banned`
  (hand list first, deduped).

Total at time of writing: 916 phrases.

```bash
cd /Users/mitchellwilliams/Documents/voice-os
python3.11 - "$FILE" <<'PY'
import sys
from voice_os.model import VoiceModel
from voice_os.qa import find_banned
from voice_os.product.graph import _LOAD_DEFAULTS
m = VoiceModel.load(
    _LOAD_DEFAULTS["corpus_path"],
    chunks_dir=_LOAD_DEFAULTS["chunks_dir"],
    mined_dir=_LOAD_DEFAULTS["mined_dir"],
    banned_path=_LOAD_DEFAULTS["banned_path"],
)
hits = find_banned(open(sys.argv[1], encoding="utf-8").read(), m.banned)
print(f"{len(m.banned)} phrases checked; hits: {hits or 'none'}")
PY
```

Matching is case-insensitive and word-boundary anchored
(`qa.py:44 find_banned`). Em dashes are handled separately and deterministically
by `qa.scrub_em_dashes`, not by the phrase list.

## Summary for handover authors

Write "Voice OS `conservation` must be clean and the input must clear its
calibrated cell threshold with zero banned hits."

Do not write "Voice OS must return `decision: pass`." That sentence gates the
wrong document.
