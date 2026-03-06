# Voice OS

**A personal communication operating system grounded in 6.9M+ words of corpus data, six psychological frameworks, and two decades of voice evolution.**

> Current alignment: **~78–86%** | Target: **95%+**

---

## Table of Contents

1. [What Is Voice OS?](#what-is-voice-os)
2. [Architecture Overview](#architecture-overview)
3. [System Prompt Design](#system-prompt-design)
4. [Knowledge Base](#knowledge-base)
5. [Operational Protocol](#operational-protocol)
6. [Use Cases](#use-cases)
7. [Deployment](#deployment)
8. [Roadmap](#roadmap)
9. [Feedback & Iteration](#feedback--iteration)
10. [Contributing](#contributing)

---

## What Is Voice OS?

Voice OS is a Claude-powered personal communication system built to generate outputs that are indistinguishable from what its owner would write themselves. Not generic professional writing. Not AI-polished corporate prose. The specific voice of one person — with all its patterns, preferences, evolution, and personality intact.

The core premise: every person has a unique communication fingerprint. That fingerprint lives in the patterns of thousands of emails, messages, posts, and conversations accumulated over years. Voice OS extracts that fingerprint, structures it into a retrieval-optimized knowledge base, and pairs it with a psychological model that can reason about *why* the person communicates the way they do — enabling authentic generation even in novel situations the corpus never covered.

**This is not a prompt template. It's an operating system.**

---

## Architecture Overview

Voice OS is a layered system with five distinct components:

```
┌─────────────────────────────────────────────────┐
│                 System Prompt v4.0               │
│     (Identity + Rules + Output Modes + QA)      │
└────────────────────────┬────────────────────────┘
                         │
┌────────────────────────▼────────────────────────┐
│              Psychological Foundation            │
│   Enneagram 4w3 · INTJ-T · Big Five · VIA       │
│        DISC · CliftonStrengths · Astro           │
└────────────────────────┬────────────────────────┘
                         │
┌────────────────────────▼────────────────────────┐
│              Knowledge Base (16 docs)            │
│  Corpus metadata · Pattern analysis · Samples   │
│  Domain vocab · Anti-patterns · Psych ops       │
└────────────────────────┬────────────────────────┘
                         │
┌────────────────────────▼────────────────────────┐
│           Temporal Weighting Model               │
│  Tier 1 (2024–26) → Tier 2 (2021–23) →         │
│  Tier 3 (2015–20) → Tier 4 (pre-2015)           │
└────────────────────────┬────────────────────────┘
                         │
┌────────────────────────▼────────────────────────┐
│          Register Calibration Engine             │
│  Channel × Audience × Situation → Adjustments   │
└─────────────────────────────────────────────────┘
```

### Layer Descriptions

**Layer 1 — System Prompt**
The instruction surface. Contains the identity model, hard rules, output mode definitions, confidence scoring framework, and quality self-assessment protocol. The system prompt doesn't contain voice data — it contains the *logic* for applying voice data.

**Layer 2 — Psychological Foundation**
Six integrated frameworks (Enneagram, MBTI, Big Five, VIA, DISC, CliftonStrengths) converted into concrete generation rules via the Psychological Operations document. This layer answers the question: *why* does the owner communicate this way, and how should that inform generation in situations the corpus doesn't cover?

**Layer 3 — Knowledge Base**
The empirical ground truth. 16 documents covering corpus metadata (6.9M+ words across email, iMessage, LinkedIn, Facebook, Instagram), extracted patterns, temporal analysis, voice evolution tracking, domain vocabulary, anti-patterns, and calibration data. The KB is the authoritative source — it supersedes any general assumptions.

**Layer 4 — Temporal Weighting Model**
A four-tier system that weights corpus data by recency. Tier 1 (most recent 2 years) is the primary voice source for generation. Tier 4 (pre-2015) is context only — never replicated. This ensures the system reflects who the owner is *now*, not who they were five years ago.

**Layer 5 — Register Calibration Engine**
A matrix of adjustments applied at generation time based on three variables: channel (email vs. chat vs. LinkedIn), audience (leadership vs. peers vs. external), and situation (error acknowledgment vs. follow-up vs. bad news). Adjustments are expressed as deltas on baseline voice dimensions (directness, warmth, formality, structure, assertiveness, precision).

---

## System Prompt Design

The system prompt (v4.0) is organized into eight logical sections:

### 1. Core Identity Block
Establishes the psychological foundation table — six frameworks, each with a direct implication for voice generation. This isn't biographical background; it's a generation constraint. The identity block answers: "When extrapolating to novel situations, reason from these traits."

### 2. Knowledge Base Access Table
Maps each document in the KB to its specific use case. The prompt instructs the system to *always* consult the knowledge base and explicitly states that KB patterns supersede general assumptions. Document types:
- Primary reference (`voice_os_complete_summary.json`)
- Quick lookup (`voice_os_compact.json`)
- Deep calibration (`Multidimensional_Communication...docx`)
- Psychological reasoning (`Unified_Psychological_Narrative.md`)
- Social register calibration (`Social_Media_Voice_Analysis_Complete.md`)

### 3. Temporal Weighting Model
A four-tier table with explicit weights and usage rules. Tier 1 = primary generation source. Tier 4 = never replicate. The prompt's critical rule: "If a pattern appears in Tier 1, use it. If it only appears in Tier 3/4, do NOT use it — that's how [the owner] used to write."

### 4. Voice Calibration Tables
Six baseline dimensions with numerical scores (0.0–1.0), current pattern data drawn directly from corpus analysis, and a signature phrase library (greetings, acknowledgments, follow-ups, closings). Also includes the banned phrase list — words and phrases that are either corporate-slop or inconsistent with current voice.

### 5. Dual-Persona Model
Two named modes:
- **The Architect** — professional written voice (high structure, high precision, no "lol," no profanity)
- **The Teammate** — personal spoken/social voice (low structure, high warmth, "lol" expected, mild profanity permitted)

### 6. Register Calibration Tables
Three-dimensional adjustment matrix:
- Audience adjustments (7 audience types × 4 dimensions)
- Channel adjustments (6 channels × 4 dimensions)
- Situational adjustments (6 situations × 4 dimensions)

### 7. Output Modes
Three distinct modes triggered by request type:
- **Mode 1: Communication Drafting** — generates 2-3 variants (Short/Standard/Warm) with context classification, confidence score, and stated assumptions
- **Mode 2: Slop Detection & Revision** — audits existing drafts, identifies issues with quoted text and authentic replacements, provides complete rewrite
- **Mode 3: Register Analysis** — analyzes how the owner would approach a given communication without generating it

### 8. Quality Self-Assessment Protocol
A 10-metric Quality Transparency Report appended to all substantive responses. Metrics include Drift, Sycophancy, Answer Relevancy, Task Completion, Correctness, Hallucination Risk, Tool Correctness, Context Relevancy, Responsibility, and a Task-Specific custom metric. Scores are 1–10 with alert thresholds at 5 (🔴 concern) and 8 (🟢 meets standard).

---

## Knowledge Base

### Corpus Statistics

| Source | Volume | Timespan | Tier |
|--------|--------|----------|------|
| Email (sent) | 17,718 messages / 5.8M words | 2007–2026 | 1–3 |
| iMessage | 852 messages | 2021–2026 | 1–2 |
| LinkedIn Posts | 627 posts | 2012–2026 | 1–3 |
| LinkedIn Messages | 1,341 messages | 2012–2026 | 1–3 |
| Facebook/Messenger | 134,042 items / 758K words | 2005–2020 | 3–4 |
| Instagram DMs | 76,517 items / 326K words | 2014–2026 | 1–2 |
| **Total** | **~6.9M words** | **2005–2026** | — |

### Tier 1 Corpus (Primary Generation Source)

| Channel | Tier 1 Volume | Key Patterns |
|---------|--------------|--------------|
| Email | 843 messages | "Hi [Name]," dominant (126 vs 114 "Hey"); "Thanks," 36%; TL;DR 2.8%; bullets 67% |
| iMessage | 231 messages | Very low structure, high warmth, "yeah" 96%+ |
| LinkedIn Messages | 102 messages | 24-word avg; 39% exclamation; 0% emoji |
| Instagram DMs | 12,099 items | Tier 1 social voice — most current casual register |

### Document Inventory (16 Files)

| File | Type | Purpose |
|------|------|---------|
| `voice_os_complete_summary.json` | Corpus metadata + samples | Primary reference |
| `voice_os_compact.json` | Condensed patterns | Quick lookup |
| `Voice_OS_Profile.md` | Original voice profile | Legacy reference |
| `Unified_Psychological_Narrative.md` | Integrated psych profile | Reasoning foundation |
| `Multidimensional_Communication.docx` | Deep dimension analysis | Calibration |
| `Social_Media_Voice_Analysis_Complete.md` | FB + IG analysis | Social register |
| `Voice_OS_Psychological_Operations.md` | Psych → generation rules | Novel situation handling |
| `Voice_OS_Gap_Analysis_v2.md` | System audit | Known gaps + resolution plan |
| `Voice_OS_Anti_Patterns.md` | What NOT to generate | Slop detection |
| `Voice_OS_Emotional_Calibration.md` | State-based adjustments | Edge cases |
| `Voice_OS_Domain_Vocabulary.md` | Google/xGE/AI terminology | Context accuracy |
| `Voice_OS_QA_Checklist.md` | Pre-send verification | Quality gate |
| `Voice_OS_Correction_Log.md` | Feedback capture | Continuous improvement |
| `Voice_OS_Recipient_Profiles_Template.md` | Person-specific calibration | Relationship nuance |
| `Voice_OS_Email_Samples_Template.md` | Curated examples | Pattern matching |
| `Voice_OS_This_Week_Action_Plan.md` | Gap closure roadmap | Project management |

### Knowledge Base Retrieval Logic

The system prompt instructs explicit consultation of KB documents at generation time. The retrieval hierarchy:

1. Check `voice_os_compact.json` for quick pattern lookup
2. Check `voice_os_complete_summary.json` for corpus-grounded data
3. Apply temporal weighting (Tier 1 patterns override older patterns)
4. Consult `Psychological_Operations.md` for novel situations not covered by corpus
5. Apply register adjustments from calibration tables
6. Run against `Anti_Patterns.md` for slop detection

---

## Operational Protocol

### Request Classification

Every incoming request is classified on three axes before generation begins:

```
CHANNEL: email | chat | linkedin | text | doc | social
AUDIENCE: leadership | peer | direct-report | external | friend/family
SITUATION: standard | follow-up | error-ack | bad-news | request | edge-case
```

This classification drives which register adjustments are applied. The system states its classification and any assumptions before generating output — no silent decisions.

### Dimension Scoring

After classification, the system computes adjusted dimension scores by applying deltas to the owner's baseline:

```
Baseline:
  Directness:  0.87
  Structure:   0.81
  Warmth:      0.62
  Formality:   0.44
  Precision:   0.82
  Assertiveness: 0.76

Example: Email to senior leadership (L7+) about a missed deadline:
  Formality:     +0.20 → 0.64
  Assertiveness: -0.10 → 0.66
  Warmth:        +0.15 → 0.77 (error acknowledgment)
  Formality:     +0.25 → 0.89 (error acknowledgment stacks)
```

### Output Generation

**Mode 1 (Drafting)** generates three variants:

- **Short** — Minimum viable words. For quick replies, simple confirmations, low-stakes asks.
- **Standard** — Balanced. Typical voice. Used when context doesn't push strongly toward either end.
- **Warm** — More relational. Appropriate when the relationship or situation calls for extra care.

Each variant includes:
- Context classification (channel/audience/situation)
- Applied dimension adjustments (with rationale)
- Confidence score (0.0–1.0)
- Stated assumptions (flagged with brackets for missing specifics)

### Confidence Scoring

| Type | Confidence |
|------|------------|
| Professional emails | 0.95 |
| LinkedIn messages | 0.92 |
| Text messages / DMs | 0.90 |
| Social media posts | 0.88 |
| Resumes/cover letters | 0.88 |
| Professional documents | 0.85 |
| Blog posts | 0.82 |
| Scripts/talking points | 0.70 |

Scores below 0.70 trigger a clarifying question. Scores 0.70–0.89 include noted uncertainties. Scores 0.90+ are ready to use with minimal review.

### Hard Rules (Non-Negotiable)

1. Main point first. Always. No warming up to the topic.
2. One clarifying question maximum — make reasonable assumptions and state them.
3. Contractions are default in all but the most formal contexts.
4. Questions create dialogue — most emails end with a question or clear next step.
5. Format serves consumption — use structure when it helps; don't impose it on simple content.
6. Warmth is genuine, not performative — include it because the owner actually cares.
7. Never invent specifics — use brackets for unknown names/dates/details.
8. Respect the aesthetic — outputs should be elegant, not just functional.
9. Tier 1 patterns override — always check the KB when in doubt.
10. "yeah" not "yea" — the spelling shifted ~2020 and is locked.

---

## Use Cases

### Professional Email Drafting

**Scenario:** Need to send a complex project update to senior leadership with multiple stakeholders.

**Input:**
```
Draft a project update to my VP about the RAG agent launch. 
Hits: 10K interactions, 170 engineers using it, 2 months in.
Context: Email, leadership audience, positive news.
```

**What Voice OS does:**
- Classifies: Email / Leadership / Positive announcement
- Applies: Formality +0.20, Structure +0.10
- Generates three variants (Short/Standard/Warm)
- Includes TL;DR (2.8% and rising — appropriate for leadership)
- Uses metric-forward structure (achievement orientation, 4w3 wing 3)
- Leads with the result, then the breakdown
- Closes with a question or next step

---

### Slop Detection & Revision

**Scenario:** Draft was written but feels off — too corporate, too AI.

**Input:**
```
Check this: "I hope this email finds you well. I wanted to reach out 
to leverage our synergy and deep dive into the opportunities..."
```

**What Voice OS does:**
- Flags banned phrases: "I hope this email finds you well," "leverage," "synergy," "deep dive"
- Identifies tone issues: generic opener, performative warmth
- Provides issue-by-issue breakdown with authentic replacements
- Delivers complete rewrite in owner's actual voice
- Assesses: Too corporate / Slightly off / Authentic

---

### Register Analysis

**Scenario:** Unsure how to approach a sensitive communication.

**Input:**
```
How would I write a message to my manager explaining I missed 
a deadline because I was dealing with a personal situation?
```

**What Voice OS does:**
- Classifies the situation (error acknowledgment + personal context)
- Shows dimension settings with rationale
- Lists voice markers to include (directness, genuine warmth, forward path)
- Lists anti-patterns to avoid (over-apologizing, excessive hedging, burying the point)
- Optionally generates the communication in Mode 1

---

### Social Voice Generation

**Scenario:** DM to a close friend about catching up.

**What Voice OS does:**
- Switches to Teammate persona
- Applies: Structure -0.60, Formality -0.40, Warmth +0.30
- Uses Instagram DM corpus (Tier 1 social voice) as reference
- Permits: "lol," "yeah," "gonna/wanna," mild profanity
- Removes: All structure, formal closings, work language
- Generates: Short, warm, authentic personal message

---

### LinkedIn Content

**Scenario:** Thought leadership post about a recent AI project.

**Voice OS handles:**
- ~24-word avg sentence target
- 39.2% exclamation usage calibration
- Hook-middle-callback structure
- No emoji (confirmed 0% in Tier 1 LinkedIn data)
- Personal stakes and credibility markers
- Engagement question at close
- Target polish: ~65–70% (elevated, not hyper-corporate)

---

### Edge Case: Emotionally Charged Communication

**Scenario:** Delivering bad news. Responding to criticism. Writing when frustrated.

**What Voice OS does:**
- Consults `Emotional_Calibration.md` for state-based adjustments
- Applies psychological reasoning (Enneagram 4 vulnerability handling, INTJ-T directness vs. DISC efficiency)
- For stressed states: increases structure (+0.10), reduces warmth slightly (-0.10), shortens length (-20%)
- For frustrated states: maintains directness, removes performative softening, keeps warmth genuine
- For bad news: directness maintained, warmth +0.15, explicit ownership

---

## Deployment

### Requirements

- **Platform:** Claude.ai (Projects feature required)
- **Model:** Claude Sonnet 4.5+ recommended (larger context window for full KB)
- **Claude Plan:** Pro or higher (Projects not available on Free)

### Setup Steps

**Step 1: Create a Claude Project**

Navigate to Claude.ai → Projects → New Project. Name it "Voice OS" or similar.

**Step 2: Upload Knowledge Base Documents**

Upload the following to Project Knowledge (order matters for retrieval priority):

```
Priority 1 (upload first):
  voice_os_complete_summary.json
  voice_os_compact.json

Priority 2 (core calibration):
  Unified_Psychological_Narrative.md
  Social_Media_Voice_Analysis_Complete.md
  Voice_OS_Psychological_Operations.md
  Voice_OS_Anti_Patterns.md
  Voice_OS_Emotional_Calibration.md
  Voice_OS_Domain_Vocabulary.md

Priority 3 (operational tools):
  Voice_OS_QA_Checklist.md
  Voice_OS_Correction_Log.md
  Voice_OS_Recipient_Profiles.md
  Voice_OS_Email_Samples.md
  Voice_OS_Gap_Analysis_v2.md
```

**Step 3: Configure System Prompt**

Paste `VOICE_OS_SYSTEM_PROMPT_v4.0.md` into the Project's custom instructions field. Do not truncate — the full prompt is required for proper operation.

**Step 4: Verify Configuration**

Run the following test prompts to verify each layer is working:

```
Test 1 (Knowledge Base): 
"What is my current Tier 1 greeting distribution?"
Expected: Should return data from voice_os_compact.json

Test 2 (Register Calibration):
"Write a quick Slack message to my manager confirming I got his note."
Expected: Should apply chat channel adjustments, leadership audience

Test 3 (Slop Detection):
"Check this: 'Please don't hesitate to reach out if you need anything.'"
Expected: Should flag "please don't hesitate to reach out" as banned phrase

Test 4 (Persona Switch):
"Text to a close friend: just say hi and check in"
Expected: Should switch to Teammate persona, no structure, warm, casual
```

**Step 5: Calibrate With Your Corpus**

Voice OS performs significantly better with actual corpus samples. Before going live for high-stakes communications:

1. Add 50+ curated email samples to `Voice_OS_Email_Samples.md`
2. Add your current resume and recent cover letters
3. Complete `Voice_OS_Recipient_Profiles.md` for 5-7 key contacts
4. Run 10 test communications across different contexts
5. Log any corrections in `Voice_OS_Correction_Log.md`

### Environment Notes

- Voice OS does not require any API keys or external services in its base configuration
- All data lives in Claude Projects knowledge base — no external database
- Session data is not persisted between conversations (stateless per session)
- Project knowledge updates require manual re-upload when documents change

---

## Roadmap

### Phase 1: Foundation Complete ✅ (~78% alignment)

- Comprehensive psychological profile (6 frameworks integrated)
- 6.9M-word corpus metadata extracted and structured
- Temporal tiering model (4 tiers with clear weighting)
- Dual-persona model (Architect / Teammate)
- Register calibration matrix (channel × audience × situation)
- System prompt v4.0 with quality self-assessment protocol
- Psychological Operations document (psych → generation rules)
- Anti-patterns library
- Emotional state calibration
- Domain vocabulary (Google/xGE/AI terminology)

### Phase 2: Sample Integration (~86% alignment, in progress)

- [ ] 50+ curated Tier 1 email samples across 10 context types
- [ ] Job application materials (cover letters, resume, brag doc)
- [ ] Recipient profiles for 5-7 key contacts (manager, peers, family, friends)
- [ ] Chat/Slack sample library (30+ examples)
- [ ] LinkedIn post samples (15-20 recent, high-performing)

### Phase 3: Feedback Loop (~92% alignment)

- [ ] Correction log populated with 20+ real corrections
- [ ] Monthly review process established
- [ ] Pattern tracking for recurring issues
- [ ] First systematic rule updates based on correction data
- [ ] QA checklist integrated into pre-send workflow

### Phase 4: Edge Case Mastery (~95%+ alignment)

- [ ] Conflict communication guidelines
- [ ] Difficult conversation templates (sensitive, emotional, high-stakes)
- [ ] Voice evolution monitoring (track drift every 6 months)
- [ ] Spoken voice / talking points calibration improvement
- [ ] Second-language recipient handling (international contacts)

### Long-Term Vision

**MCP Integration:** Voice OS as an MCP-native architecture — directly connected to Gmail, Google Chat, and LinkedIn for real-time draft generation without copy-paste workflows.

**Evaluation Framework:** Automated alignment scoring against real sent communications. Current manual alignment estimate (~78-86%) → objective measurement via embedding similarity.

**Voice Evolution Tracking:** Periodic corpus updates that automatically detect when patterns shift (like the "yea" → "yeah" transition in 2020) and update generation rules accordingly.

**Multi-Context Voice Modeling:** Separate fine-tuned models for high-volume, high-stakes context types (job applications, executive communications, media/external).

---

## Feedback & Iteration

### Current Alignment Status

**Overall: ~78–86%** (January–March 2026 estimate)

| Component | Score | Status |
|-----------|-------|--------|
| Psychological Foundation | 14/15 | Strong — not fully operationalized |
| Corpus Coverage | 12/15 | Good — retrievable samples needed |
| Pattern Analysis | 14/15 | Strong |
| Temporal Calibration | 10/10 | Complete |
| Register Differentiation | 8/10 | Good — specific recipient calibration missing |
| Edge Case Handling | 3/10 | Major gap — in progress |
| Feedback Systems | 0/10 | New — correction log just initialized |
| Documentation Quality | 8/10 | Good |

### Known Issues

**Gap 1 — No retrievable current-era samples (Critical)**
Patterns are extracted but actual 2024-2026 email samples aren't stored in the KB. System reasons from rules only, not examples. Significant improvement expected when 50+ samples are loaded.

**Gap 2 — Job application materials missing (Critical)**
Cover letters and resume are referenced in the KB but not stored. Voice OS cannot reliably generate application materials without them.

**Gap 3 — Psychological insights partially operationalized (High)**
The Psychological Operations document was added to address this. Ongoing refinement needed as edge cases surface.

**Gap 4 — No feedback mechanism until now (High)**
Correction Log is initialized. First month of data collection underway.

**Gap 5 — Generic audience tiers only (High)**
Recipient profiles need to be populated for key contacts to capture relationship-specific nuance.

### How to Log Corrections

When Voice OS generates output that doesn't sound authentic, use the Correction Log template:

```markdown
### [DATE] — [Brief Description]

**Context:** [Channel, recipient, purpose]

**Voice OS Output:**
> [Paste problematic output]

**Problem:**
- [ ] Wrong tone
- [ ] Wrong structure  
- [ ] Wrong greeting/closing
- [ ] Banned phrase
- [ ] Too formal / Too casual
- [ ] Missing warmth
- [ ] Not direct enough
- [ ] Other: ___

**How I Would Actually Say It:**
> [Your corrected version]

**Pattern to Update:** [What rule should change]

**Severity:** Minor tweak | Significant rewrite | Complete miss
```

### Monthly Review Process

1. Review all correction log entries from the past month
2. Identify recurring issues (any pattern appearing 3+ times)
3. Update relevant KB documents with new rules or examples
4. Re-run the 10 verification test prompts
5. Update alignment estimate
6. Tag entries as "resolved" or "needs more data"

---

## Contributing

Voice OS is a personal system — its knowledge base is specific to one person and not generalizable. However, the architecture, system prompt structure, and methodological approach are designed to be adaptable.

If you're building a similar system for yourself:

1. **Start with psychological frameworks** — they provide the reasoning backbone when corpus data doesn't cover a situation
2. **Corpus size matters** — the more data, the more reliable the pattern extraction
3. **Temporal weighting is critical** — voice evolves; old patterns actively hurt generation quality if used uncritically
4. **Build the feedback loop early** — it's the only path to 95%+ alignment
5. **Separate what-from-why** — knowing *what* patterns exist without knowing *why* limits extrapolation to novel situations

---

## Technical Reference

This section consolidates internal technical details for quick reference — system prompt section mapping, token budgets, generation pipeline, infrastructure dependencies, and generation constraints.

### System Prompt Sections (Quick Reference)

| Section | Purpose |
|---------|---------|
| Core Identity (Psychological Foundation) | Establishes the owner's Enneagram, MBTI, Big Five, VIA, and DISC profile as generation priors |
| Knowledge Base Access Table | Maps each KB document to its purpose so Claude knows when to reference it |
| Temporal Weighting Model | Defines the 4-tier corpus system; enforces Tier 1 override rule |
| Current Voice Calibration | Baseline dimension scores, signature phrases, banned phrases, and Tier 1 patterns |
| Dual-Persona Model | Defines The Architect (professional) and The Teammate (personal) modes |
| Register Calibration by Context | Audience and channel adjustment tables |
| Output Modes | Specifies Mode 1 (Drafting), Mode 2 (Slop Detection), Mode 3 (Register Analysis) |
| Hard Rules | 10 non-negotiable generation constraints |
| Confidence Scoring | Score definitions by use case; alert thresholds |
| Quality Transparency Report | Self-assessment protocol appended to all substantive outputs |

### Knowledge Base Token Estimates

| Document | Estimated Tokens | When Retrieved |
|----------|-----------------|----------------|
| `voice_os_compact.json` | ~30K | Every generation |
| `voice_os_complete_summary.json` | ~60K | Deep calibration only |
| `Unified_Psychological_Narrative.md` | ~15K | Novel situations |
| `Social_Media_Voice_Analysis_Complete.md` | ~12K | Personal register requests |
| `Voice_OS_Domain_Vocabulary.md` | ~5K | Google/work contexts |
| All other documents | ~5K avg | Contextually |

Voice OS's full knowledge base, when retrieved simultaneously, approaches the 200K token context limit. The `compact.json` optimization reduces token usage by ~40% vs. the full summary, making it the primary reference for standard generation tasks.

### Input Classification to Mode Mapping

| Input Signal | Mode Triggered | KB Documents Consulted |
|-------------|---------------|----------------------|
| "Draft / write / compose" | Mode 1: Communication Drafting | compact.json, email samples, recipient profiles, domain vocab |
| "This doesn't sound like me" / "fix this" / "clean this up" | Mode 2: Slop Detection & Revision | anti_patterns, compact.json, QA checklist |
| "How would I write..." / "what register..." | Mode 3: Register Analysis | compact.json, psychological_operations |
| Edge case signals (conflict, bad news, criticism) | Mode 1 + Emotional Calibration | emotional_calibration, psychological_operations |

### 8-Step Generation Pipeline

For every request, Voice OS follows this sequence:

| Step | Action | Source |
|------|--------|--------|
| 1 | Classify input (mode, channel, audience, emotional context) | System prompt |
| 2 | Retrieve relevant KB documents based on classification | Knowledge base |
| 3 | Apply dimension adjustments for audience + channel | Calibration tables |
| 4 | Apply temporal override (Tier 1 patterns take precedence) | Temporal model |
| 5 | Generate 2-3 variants (Short / Standard / Warm) | Generation rules |
| 6 | Apply hard rules pass (banned phrases, main-point-first, etc.) | Hard rules layer |
| 7 | Compute confidence score | Corpus match data |
| 8 | Append Quality Transparency Report | Self-assessment protocol |

### Dependencies & Infrastructure

| Dependency | Type | Role | Replacement Risk |
|-----------|------|------|-----------------|
| Anthropic Claude | Foundation model | All language generation | High — system prompt is Claude-specific |
| claude.ai | Interface | Project management, KB storage | Medium — could migrate to API with effort |
| Claude Project system | Feature | Persistent KB attachment | Low — core Claude feature |
| 16-document Knowledge Base | Data | Voice calibration | None — built and maintained by the owner |

### Status Dysregulation Pattern (Generation Constraint)

The owner has documented specific composure anti-patterns that Voice OS must not replicate. These include accommodation behaviors that temporarily reduce anxiety but erode authority over time. Anti-patterns are documented in `Voice_OS_Anti_Patterns.md` and `Voice_OS_Psychological_Operations.md`.

---

*Voice OS v4.0 | Architecture: Claude Projects + Knowledge Base | Alignment target: 95%+*
*Last updated: March 2026*
