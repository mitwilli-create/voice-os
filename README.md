# Voice OS — Personal AI Communication System

A personal communication operating system built on Claude, designed to generate written communications that are consistent, authentic, and calibrated across professional and personal contexts.

## Problem

Communication output quality varies across contexts — too casual in some, too formal in others, inconsistent register across channels, corporate-speak bleeding into authentic writing. At scale, this inconsistency erodes trust and requires constant manual correction.

## Architecture

Voice OS uses a multi-layer knowledge base approach inside Claude Projects:

### Knowledge Base Structure
- **Corpus layer** — longitudinal writing samples stratified by time period (Tier 1: recent, highest weight → Tier 4: oldest, lowest weight)
- **Psychological layer** — behavioral and cognitive profile integration (communication style, decision patterns, stress responses)
- **Anti-patterns library** — explicit banned phrases, failure modes, and slop detection rules
- **Recipient profiles** — audience-specific calibration tables (formality, directness, warmth adjustments)
- **QA checklists** — pre-draft (input) and post-draft (output) quality gates

### Dual Persona Model
| Persona | Context | Characteristics |
|---------|---------|-----------------|
| The Architect | Professional communications | High directness, high structure, low formality |
| The Teammate | Casual/personal communications | Moderate directness, low structure, high warmth |

### Generation Pipeline
1. **Classify** — determine context, audience, urgency, register
2. **Register Select** — route to Architect or Teammate persona
3. **Calibrate** — apply audience-specific adjustments from recipient profile
4. **Generate** — produce 2-3 variants
5. **QA Check** — run against anti-patterns library and output checklist

### Dimensional Scoring
Voice calibration tracked across 6 axes:
- Directness
- Structure Density
- Precision
- Assertiveness
- Warmth
- Formality

### Temporal Weighting
Corpus ingestion uses a 4-tier recency model — recent communications receive higher calibration weight to account for voice evolution over time.

## Quality Gates

**Pre-draft checklist (input):**
- Audience identified and profile loaded?
- Register selected (Architect / Teammate)?
- Urgency and context flagged?
- Any emotional state modifiers active?

**Post-draft checklist (output):**
- Anti-patterns scan complete?
- Main point surfaced in first two sentences?
- Passive voice eliminated?
- Length appropriate to channel?
- Tone matches recipient calibration?

## Anti-Patterns Library (sample)
Banned constructs include:
- Corporate filler: "leverage" (verb), "utilize," "synergy," "deep dive," "circle back"
- Empty openers: "Great question," "Hope this finds you well," "As per my last email"
- Buried leads: main point in paragraph 3+
- Hedge stacking: "I think it might potentially be worth considering"

## Use Cases
- Professional email drafting (peer, leadership, external)
- LinkedIn content and networking messages
- Job application materials
- Personal communications
- Self-reflection documents

## Key Design Decisions

**Why Claude Projects over a custom app?**
Claude Projects provides persistent KB, stateful memory, and native document ingestion without requiring infrastructure management. The tradeoff (context window limits, no API access) is acceptable for personal use.

**Why temporal stratification?**
Voice evolves. A corpus weighted equally across 20 years would anchor calibration to outdated patterns. Tier weighting ensures the system reflects current communication style.

**Why explicit anti-patterns vs. just good examples?**
Positive examples teach the model what to do. Anti-patterns explicitly prevent the model's default tendencies (corporate-speak, buried leads, hedge stacking) from bleeding into outputs.

## Current Status
Active, production. Personal use only — not a product.
