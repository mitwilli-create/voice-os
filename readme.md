# Voice OS

A personal communication system built on Claude that generates output in my actual voice -- not AI-polished, not generic professional writing, mine.

**TL;DR:** Voice OS drafts emails, LinkedIn posts, and messages calibrated to how I actually write. It routes between two personas, scores on six voice dimensions, and catches corporate filler before anything goes out. Built on 6.9M+ words of my own writing. [Full architecture docs](docs/architecture.md).

---

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

## Why I built this

My communication quality varies across contexts. Too formal in some, too casual in others. Corporate-speak bleeds in when I'm tired. I built this to solve that permanently.

## What's not in this repo

The corpus itself -- 20+ years of personal writing -- isn't here. Neither are the calibrated recipient profiles. This documents the system design, not the personal data that powers it.

## Status

Active, production. Personal use only. Current alignment: ~78-86% (target: 95%+).

This was fun to build. Questions? [Open an issue](https://github.com/mitchellwilliams/voice-os/issues).
