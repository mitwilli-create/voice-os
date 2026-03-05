# Voice OS

A personal communication system built on Claude. It generates written output that sounds like me — not like AI, not like generic professional writing, like me.

## Why it exists

My communication quality varies across contexts. Too formal in some, too casual in others. Corporate-speak bleeds in when I'm tired. I built this to solve that permanently.

## What it does

- Drafts emails, LinkedIn posts, networking messages, and cover letters calibrated to my actual voice
- Routes between two personas: **The Architect** (professional) and **The Teammate** (casual)
- Applies audience-specific adjustments — formality, directness, warmth — based on who I'm writing to
- Runs every draft through a pre/post checklist before anything goes out
- Catches my failure modes: buried leads, hedge-stacking, corporate filler bleeding in

## Architecture

**Knowledge base layers:**
- **Corpus** — years of my own writing, stratified by recency (recent writing gets more weight)
- **Psychological profile** — behavioral patterns that affect how I communicate under stress, excitement, time pressure
- **Anti-patterns library** — explicit banned phrases and failure modes
- **Recipient profiles** — calibration tables per audience type

**Generation pipeline:**
1. Classify context and audience
2. Select persona (Architect or Teammate)
3. Apply audience calibration
4. Generate 2–3 variants
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

## What's not here

The corpus itself — 20+ years of personal writing — isn't in this repo. Neither are the calibrated recipient profiles. This documents the system design, not the personal data that powers it.

## Status

Active, production. Personal use only.
