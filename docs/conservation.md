# Content Conservation

Status: shipped. Motivated by the 2026-07-08 field report
(feedback/2026-07-08-storytellermitch-site-pass.md), whose headline
finding was that the QA gate scores voice alignment, not content
conservation: a redraft that invented four opinions the author never
wrote passed at fidelity 0.718 with zero warnings.

Module: `voice_os/conservation.py` (stdlib-only, deterministic).
Wiring: the product graph's `qa_gate` node runs every check on every
cycle; results land in the envelope's additive `conservation` field.

## The redraft contract

`voice_os.draft(text, redraft=True)` / `python3 -m voice_os draft
--redraft` declares the input finished writing being re-voiced rather
than a brief being composed from. Under the contract, every output
sentence must be entailed by the input; unentailed sentences block a
pass (revision signals name them; exhausted revisions reject).

With `redraft=False` (the default, and compose semantics) the same
checks run and report in the envelope, but only quote violations
block. Existing callers see identical decisions unless a quote span
was damaged, which was always a defect.

## Checks

| Check | Receipt it closes | Blocking |
|---|---|---|
| `unsupported_sentences` — lexical claims diff: an output sentence is flagged when fewer than half its content words trace to the input (stem or 6-char-prefix match; sentences under 3 content words are cadence, not claims) | builder-turn's four invented opinions; stream-launch-night's fabricated closer | redraft mode only |
| `quote_violations` — input text inside double quotes must survive verbatim, marks included (curly/straight glyphs normalized) | mong-kok pull returned unquoted | always |
| `dropped_modifiers` — precision hedges within two tokens before a numeral ("roughly fifty"), the modifier slot directly after it ("four-month **electrical** blackout"), framing labels ("on air", "design targets", "internal figure") | hurricane-maria, builder-turn, jazz-jennings qualifier drops | advisory + revision signal |
| `format_flags` — markdown lists/headings introduced into prose input | trans-navy-panel bullet leak | advisory + revision signal |
| `diction_flags` — charged terms in the output the input never used | "pursuing its critics" → "hunting its critics" | advisory + revision signal |
| `diction_escalations` — a `diction_flags` term sharing a sentence with a heuristic named entity (stdlib sentence-case NER: mid-sentence capitalized runs, acronyms, and sentence-initial names anchored elsewhere in the text; no model, so lowercase particles like "bin" split a name) | the same receipt, which mattered precisely because Scientology was in range | advisory + stronger revision signal naming the party |

Calibration: against the site pass's 17 original/human-approved body
pairs the full check set produces zero false flags, while every
receipt in the report is caught. The claims diff whitelists the house
banned-phrase replacement vocabulary (`qa.REPLACEMENTS`), so the
pipeline's own substitutions are never flagged as invented.

## Conservative mode below 25 words

Micro-copy at the calibration floor degraded in 14/17 pull quotes in
the report. For inputs under 25 words, quote spans are protected in
every mode; under the redraft contract the generate and revise nodes
additionally keep a rewrite only when it adds no unentailed content
and beats the input's own fidelity by at least 0.05. Otherwise the
input is returned unchanged, and an unchanged short input passes
rather than rejecting (the author's own words are never "below the
bar"). Compose briefs are exempt from the entailment and margin
retentions: a brief is meant to be expanded, not handed back. The
guard is suspended while the input carries banned phrases so the
scrub machinery can still work.

## Envelope field (additive)

```json
"conservation": {
    "redraft": false,
    "unsupported_sentences": [{"sentence": "...", "support": 0.25}],
    "quote_violations": ["\"...\""],
    "dropped_modifiers": [{"modifier": "roughly", "anchor": "fifty",
                           "kind": "numeral-adjacent"}],
    "format_flags": ["..."],
    "diction_flags": ["hunt"],
    "input_retained": false
}
```

Existing envelope keys are untouched; callers parsing
mode/decision/fidelity/banned_hits/output_text are unaffected.
Redraft callers should treat non-empty `unsupported_sentences` or
`quote_violations` on a pass (possible only in compose mode) as an
audit signal.

## Em-dash scrub convention

`qa.scrub_em_dashes` rewrites rather than glyph-swaps (report class
4b): digit ranges become an unspaced hyphen, line-opening dashes stay
list markers, decorative dashes drop, and every other run becomes a
comma pause. The scrub never emits the spaced-hyphen tell `" - "`.

## Batch callers: signature moves and the avoid list

Single-call architecture cannot see corpus-level convergence. Across
the 17 independent runs of the site pass the same signature moves
recurred until they read as a house tic (report class 6). The
catalog (`voice_os/moves.py`):

| Key | Move |
|---|---|
| `fragment-date-opener` | "December 2007." / "October 2014." bare-date openings |
| `x-not-y` | "X. Not Y." corrective-fragment constructions |
| `no-x-run` | staccato "No X. No Y. No Z." fragment runs |
| `punch-tag-closer` | "That's the whole point." / "Every time." / "On purpose." emphatic tags |
| `fragment-closer` | any ≤4-word unquoted closing fragment |

Detection always runs; every envelope reports what the output used in
the additive `signature_moves.detected` field, so a batch caller can
tally moves across items and rotate any move converging past ~3 of
17. To opt a run out of specific moves, pass
`draft(avoid=["fragment-date-opener", ...])` or CLI
`--avoid fragment-date-opener,punch-tag-closer` (repeatable): the
persona is steered away, and a detected avoided move blocks a pass
(revise, then reject) the same way a redraft entailment failure does.
Unknown keys fail fast with the catalog listed. One precedence rule:
if a sub-25-word input is retained unchanged, conservation outranks
the avoid list; the violation is still reported for the caller.

```json
"signature_moves": {
    "detected": {"fragment-date-opener": ["October 2014."]},
    "avoid": ["punch-tag-closer"],
    "violations": []
}
```
