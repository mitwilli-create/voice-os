# Voice OS

A personal communication system built on Claude that generates output in my actual voice -- not AI-polished, not generic professional writing, mine.

**TL;DR:** Voice OS drafts emails, LinkedIn posts, and messages calibrated to how I actually write. It routes between two personas, scores on six voice dimensions, and catches corporate filler before anything goes out. Built on 6.9M+ words of my own writing. [Full architecture docs](docs/architecture.md).

---

## In 60 seconds

- **What it does:** Claude Projects-based communication system that writes in my actual voice.
- **How:** Six-axis scoring (Directness, Structure, Precision, Assertiveness, Warmth, Formality) + dual persona (Architect/Teammate) + temporal corpus weighting.
- **Scale:** Calibrated on 6.9M+ words across email, LinkedIn, iMessage, and social.
- **Status:** Active, ~78–86% alignment; target 95%+.
- **Read next:** [architecture](docs/architecture.md).

## What it does

- Drafts emails, LinkedIn posts, networking messages, and cover letters calibrated to my actual voice
- Routes between two personas: **The Architect** (professional) and **The Teammate** (casual)
- Applies audience-specific adjustments -- formality, directness, warmth -- based on who I'm writing to
- Runs every draft through a pre/post checklist before anything goes out
- Catches my failure modes: buried leads, hedge-stacking, corporate filler bleeding in

## How it works

**Knowledge base layers:**
- **Corpus** -- 6.9M+ words of my own writing, stratified by recency (recent writing gets more weight)
- **Psychological profile** -- behavioral patterns that affect how I communicate under stress, excitement, time pressure
- **Anti-patterns library** -- explicit banned phrases and failure modes
- **Recipient profiles** -- calibration tables per audience type

**Generation pipeline:**
1. Classify context and audience
2. Select persona (Architect or Teammate)
3. Apply audience calibration
4. Generate 2-3 variants
5. Run QA checklist

**Dimensional scoring tracks six axes:** Directness, Structure Density, Precision, Assertiveness, Warmth, Formality.

## Quality gates

Every draft runs through two checklists:

**Pre-draft (input):**
- Audience identified and profile loaded?
- Persona selected?
- Emotional state modifiers active?

**Post-draft (output):**
- Anti-patterns scan passed?
- Lead in the first two sentences?
- Passive voice eliminated?
- Length appropriate to channel?

## Example: drafting in action

Mode 1 produces three variants (Short / Standard / Warm) with an explicit classification block, applied dimension adjustments, and a confidence score.

![Mode 1 drafting — classification and adjustments](docs/images/02-mode1-drafting-1.png)

![Mode 1 drafting — short and standard variants](docs/images/02-mode1-drafting-2.png)

![Mode 1 drafting — warm variant and confidence](docs/images/02-mode1-drafting-3.png)

## Example: slop detection in action

**Input (draft submitted for review):**

> Hi team — I hope this email finds you well. I wanted to quickly reach out to leverage our synergy and circle back on the deliverables from last week's ideation session. As a quick FYI, please don't hesitate to reach out if you have any questions or bandwidth issues!

**Voice OS output:**

*Banned phrases flagged:*
- "hope this email finds you well" — performative opener; 0 matches in Tier 1 corpus
- "leverage our synergy" — corporate filler; banned
- "circle back" — banned
- "please don't hesitate to reach out" — stock closer; banned

*Tone issues:*
- Generic opener buries the actual point (main-point-first rule violated)
- Performative warmth reads anxious
- Closes with a stock phrase, not a real next step

*Rewrite (Standard variant):*

> Quick update on last week's deliverables:
> - [specific item 1]: shipped Monday
> - [specific item 2]: pending input from [name]
>
> Anything blocking on your end?

*Scoring vs. baseline:* Directness +0.31, Warmth +0.12, Formality −0.15. Confidence: 0.94.

Running live in Claude:

![Mode 2 slop detection — banned phrases flagged](docs/images/03-mode2-slop-detection-1.png)

![Mode 2 slop detection — tone issues and rewrite](docs/images/03-mode2-slop-detection-2.png)

![Mode 2 slop detection — scoring vs. baseline](docs/images/03-mode2-slop-detection-3.png)

## Example: Quality Transparency Report

Every substantive Voice OS output ends with a 10-metric self-assessment. Sample from an email-drafting run:

| Metric | Score | Note |
|---|---|---|
| Drift | 9/10 | Output aligns with Tier 1 patterns |
| Sycophancy | 10/10 | No performative warmth |
| Answer Relevancy | 9/10 | Addressed the actual ask |
| Task Completion | 10/10 | Three variants delivered |
| Correctness | 9/10 | No fabricated specifics |
| Hallucination Risk | 10/10 | All claims source-grounded |
| Tool Correctness | — | No tools used this turn |
| Context Relevancy | 9/10 | Channel + audience detected correctly |
| Responsibility | 10/10 | Assumptions stated |
| Voice Alignment (custom) | 8/10 | Warmth slightly below baseline; adjusted in Warm variant |

Alert thresholds: < 5 = concern, ≥ 8 = meets standard.

## Why I built this

My communication quality varies across contexts. Too formal in some, too casual in others. Corporate-speak bleeds in when I'm tired. I built this to solve that permanently.

## What's not in this repo

The corpus itself -- 20+ years of personal writing -- isn't here. Neither are the calibrated recipient profiles. This documents the system design, not the personal data that powers it.

## Status

Active, production. Personal use only. Current alignment: ~78-86% (target: 95%+).

This was fun to build. Questions? [Open an issue](https://github.com/mitwilli-create/voice-os/issues).
