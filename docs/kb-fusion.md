# KB Fusion: Compact KB Patterns in Live Persona Prompts

Status: implementation record for `feat/kb-fusion`. Closes the item
parked in docs/callable-layer.md ("KB pattern data fused into persona
prompts") and gap 4 of the architecture review.

## Problem

Every product-graph run already loads, content-hashes, and snapshots
the compact KB (voice_os/product/kb.py): `prepare` records `kb_meta`
and the bundle hash in provenance, so runs are auditable to the exact
KB they saw. But none of the KB's PATTERN CONTENT ever reached the
personas. The hand-built voice intelligence from the legacy claude.ai
Voice OS (greeting and closing distributions, structure and formality
rates, social voice notes) sat in state metadata while the live persona
drafted from six axis numbers, a banned list, and (since PR #18) three
exemplars and a length budget.

## What is fused, and why those fields

`distill_kb_guidance(bundle)` in voice_os/product/kb.py turns the
loaded compact KB into a short list of plain-English voice-pattern
statements. It reads exactly these sections:

- `pattern_analysis_by_tier.tier_1_current.email.greetings` and
  `.closings`: the top 3 patterns by observed count. These are direct
  phrasing preferences (how the author actually opens and closes
  email) that no axis score or exemplar reliably conveys.
- `...tier_1_current.email.structure` (bullet, bold, TLDR usage
  percentages) and `...email.formality.contraction_usage_pct`: style
  rules stated as observed rates, so the model calibrates rather than
  overfits to a single example.
- `...tier_1_current.linkedin` punctuation and emoji rates, plus
  `linkedin_voice_notes.social_media_patterns.post_style` and
  `linkedin_voice_notes.networking_message_patterns.greeting`: the
  social-surface voice patterns the corpus exemplars are thinnest on.

Tier 1 only, by the KB's own temporal model: tier 1 is the
current-voice tier with primary weight; tiers 2 and 3 exist to track
evolution, not to define today's voice.

Deliberately excluded:

- `corpus_samples` and `linkedin_voice_notes...typical_responses`: raw
  personal messages. The exemplar mechanism from PR #18 already covers
  the "show, don't tell" role with holdout-filtered chunks; duplicating
  raw messages here would add personal text to prompts without adding
  signal.
- `voice_evolution.*.insight`: phrased as analyst instructions ("Track
  shift from..."), not generation guidance; the current-tier
  distributions already carry the endpoint of each trend.
- All word-count averages: draft length is governed by the
  input-derived `length_target_words` (PR #18); KB length statistics
  would conflict with it.
- `documents` and the system prompt: long-form content; fusing it is a
  future step and needs its own size and relevance strategy.

The distiller is schema-tolerant and never invents content: missing or
mistyped sections are skipped, and an absent KB yields an empty list
(the run proceeds exactly as before).

## Size bounds

Same bounding idea as the per-exemplar 120-word cap in PR #18, applied
to the whole KB section:

- Each item is first normalized to a single rendered line: internal
  whitespace (including newlines) collapses to single spaces, then the
  line is capped at `KB_GUIDANCE_LINE_MAX_WORDS = 40` words and
  `KB_GUIDANCE_LINE_MAX_CHARS = 400` characters (the character cap
  stops pathological no-space tokens a word count alone would miss).
- `KB_GUIDANCE_MAX_WORDS = 220` total words, counted on the normalized
  text; accumulation stops before the line that would overflow.
- `KB_GUIDANCE_MAX_ITEMS = 12` lines.

Budgeting happens on exactly the text that renders: one item is one
prompt line through `_profile_block`, so KB strings with embedded
newlines or giant tokens cannot defeat the bound (Qodo round 1
finding). The real compact KB currently distills to 7 lines and 92
words, well inside the caps; they exist so future KB growth or a
malformed KB file cannot bloat prompts or checkpoints.

## Plumbing (extension pattern from PR #18)

- `VoiceState` gains `kb_guidance: list[str]` (JSON-safe), seeded empty
  by `initial_state`.
- `prepare` calls `distill_kb_guidance` on the already-loaded bundle
  and records a trace note with the pattern and word counts when
  non-empty.
- `GenerativePersona.revise` takes an additive `kb_guidance=None`
  keyword. `_profile_block` emits an "Observed voice patterns from the
  author's knowledge base:" section only when provided, with every
  guidance line nested under the header (data, not instructions: the
  same delimiting as the exemplar block, so embedded newlines or
  prompt-like markers cannot alter the block structure). With the
  kwarg absent the block is byte-identical to before, locked by the
  profile-block stability test, so `run_cycles` / `run_pipeline` and
  their goldens are untouched.
- `generate` and `revise` pass `kb_guidance` from state. The offline
  regex persona and `AdversarialPersona` ignore it entirely, keeping
  offline runs and legacy surfaces byte-stable.

## Privacy

KB pattern text is personal data. Until now it stayed in gitignored
`sources/` and gitignored `var/kb_snapshots/`. With this change the
distilled statements additionally:

- enter live persona prompts (sent to the model API), and
- land in run checkpoints (`var/runs.sqlite`) via `VoiceState`.

Both are the same data class and destinations as the draft text and
exemplar text those surfaces already carry (see docs/live-alignment.md
privacy note). Everything stays under the gitignored `var/` directory;
nothing personal enters git, and tests use only the synthetic Test
Person fake KB.

## Measurement plan

Same instrument discipline as the live-alignment track: the live
harness (18 held-out cases, same judge) runs before and after this
change lands in live use, and the before/after pair is quoted when the
result is evaluated. Expected observables: `alignment_judged` and the
judge's `same_author` component move up or hold; pass rate does not
regress; em-dash rate stays 0.0; greeting/closing choices in failing
drafts stop diverging from the author's observed patterns. If judged
alignment regresses, the fusion is feature-flagged off by reverting the
two `kb_guidance=` pass-throughs in graph.py (the distiller and state
field can stay).
