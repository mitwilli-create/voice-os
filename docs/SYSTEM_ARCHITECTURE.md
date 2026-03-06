Voice OS
System Architecture
Technical Reference for Mitchell Williams’ Communication OS

v4.0 | March 2026 | Internal

System Overview
Voice OS is a prompt-engineered AI communication system built on Anthropic’s Claude (claude-sonnet-4-20250514). It is not a standalone app or API — it is a configured Claude Project with a curated knowledge base that enables voice-calibrated content generation.

The system has three primary components:
A System Prompt (v4.0) that defines generation rules, persona, and output modes
A 16-document Knowledge Base that provides corpus data, psychological calibration, and real-world examples
An Operational Protocol that governs how inputs are classified and outputs are generated

Property
Value
Foundation Model
Anthropic Claude (claude-sonnet-4-20250514)
Deployment
Claude.ai Project
System Prompt Version
v4.0
Knowledge Base Documents
16
Corpus Size
5.8M+ words (17,718 emails + LinkedIn + iMessage + social)
Current Voice Alignment
~78%
Target Alignment
95%+
Primary Use Cases
Email, LinkedIn, Slack/Chat, Text/DM, Documents

Architecture Layers
Voice OS operates across four distinct layers. Each is described below.

Layer 1: Foundation Model
All generation runs through Claude, Anthropic’s large language model. Voice OS does not call any external APIs, run custom model fine-tuning, or execute code at inference time. It is a pure prompt + knowledge engineering solution.

Attribute
Detail
Model
claude-sonnet-4-20250514
Interface
claude.ai web/mobile/desktop
Context Window
200K tokens (enables full knowledge base retrieval)
Fine-tuning
None — behavior is entirely prompt-governed
Deployment type
Hosted (Anthropic infrastructure)

Layer 2: System Prompt
The System Prompt is the behavioral specification of Voice OS. It is pasted at the start of every Claude Project session and defines all generation rules.

Prompt Sections
Section
Purpose
Core Identity (Psychological Foundation)
Establishes Mitchell’s Enneagram, MBTI, Big Five, VIA, and DISC profile as generation priors
Knowledge Base Access Table
Maps each KB document to its purpose so Claude knows when to reference it
Temporal Weighting Model
Defines the 3-tier corpus system; enforces Tier 1 override rule
Current Voice Calibration
Baseline dimension scores, signature phrases, banned phrases, and Tier 1 patterns
Dual-Persona Model
Defines The Architect (professional) and The Teammate (personal) modes
Register Calibration by Context
Audience and channel adjustment tables
Output Modes
Specifies Mode 1 (Drafting), Mode 2 (Slop Detection), Mode 3 (Register Analysis)
Hard Rules
10 non-negotiable generation constraints
Confidence Scoring
Score definitions by use case; alert thresholds
Quality Transparency Report
Self-assessment protocol appended to all substantive outputs

Layer 3: Knowledge Base
The Knowledge Base is the corpus + calibration layer. It lives as a set of documents attached to the Claude Project, making them retrievable at generation time. The knowledge base is the primary differentiator between Voice OS and generic AI output.

Document Inventory
Document
Type
Size / Scope
Function
voice_os_compact.json
JSON
~500K tokens
Primary pattern reference — condensed corpus stats
voice_os_complete_summary.json
JSON
~1M tokens
Full metadata and calibration data
Mitchell_Williams_Unified_Psychological_Narrative.md
Markdown
~30K tokens
Integrated psychological profile for novel situation extrapolation
Social_Media_Voice_Analysis_Complete.md
Markdown
~20K tokens
FB + IG voice analysis for personal register
Mitchell_Williams_Voice_OS_Profile.md
Markdown
~15K tokens
Legacy voice profile (reference only)
Mitchell_Williams__Multidimensional_Communication.docx
DOCX
~25K tokens
Deep communication dimensions analysis
Voice_OS_Anti_Patterns.md
Markdown
~10K tokens
What not to generate — used for slop detection
Voice_OS_Correction_Log.md
Markdown
~5K tokens
Logged corrections for pattern improvement
Voice_OS_Domain_Vocabulary.md
Markdown
~8K tokens
Google/xGE-specific terminology
Voice_OS_Email_Samples_Template.md
Markdown
~15K tokens
Curated real email examples by context
Voice_OS_Emotional_Calibration.md
Markdown
~8K tokens
Emotional state handling rules
Voice_OS_Gap_Analysis_v2.md
Markdown
~12K tokens
Known gaps and resolution roadmap
Voice_OS_Psychological_Operations.md
Markdown
~10K tokens
Psych frameworks as concrete generation rules
Voice_OS_QA_Checklist.md
Markdown
~5K tokens
Pre-send verification protocol
Voice_OS_Recipient_Profiles_Template.md
Markdown
~5K tokens
Key contact calibration template
Voice_OS_This_Week_Action_Plan.md
Markdown
~8K tokens
Development roadmap and action items

Layer 4: Operational Protocol
The Operational Protocol is the request processing layer — how Voice OS classifies inputs and selects generation strategies.

Input Classification
Input Signal
Mode Triggered
KB Documents Consulted
"Draft / write / compose"
Mode 1: Communication Drafting
compact.json, email samples, recipient profiles, domain vocab
"This doesn’t sound like me" / "fix this" / "clean this up"
Mode 2: Slop Detection & Revision
anti_patterns, compact.json, QA checklist
"How would I write..." / "what register..."
Mode 3: Register Analysis
compact.json, psychological_operations
Edge case signals (conflict, bad news, criticism)
Mode 1 + Emotional Calibration
emotional_calibration, psychological_operations

Generation Pipeline
For every request, Voice OS follows this sequence:

Step
Action
Source
1
Classify input (mode, channel, audience, emotional context)
System prompt
2
Retrieve relevant KB documents based on classification
Knowledge base
3
Apply dimension adjustments for audience + channel
Calibration tables
4
Apply temporal override (Tier 1 patterns take precedence)
Temporal model
5
Generate 2–3 variants (Short / Standard / Warm)
Generation rules
6
Apply hard rules pass (banned phrases, main-point-first, etc.)
Hard rules layer
7
Compute confidence score
Corpus match data
8
Append Quality Transparency Report
Self-assessment protocol

Corpus Architecture
Data Sources
Source
Volume
Years
Tier Assignment
Gmail (sent)
17,718 emails
2007–2026
T1: 843 | T2: 1,348 | T3: 15,527
iMessage
852 messages
2022
T1: 231 | T2: 621
LinkedIn Posts
627 posts
2012–2026
T1: 0 | T2: 12 | T3: 615
LinkedIn Messages
1,341 messages
2012–2026
T1: 102 | T2: 662 | T3: 577
Facebook
134,042 posts/comments
2005–2023
T3/T4
Instagram
76,517 posts/DMs
2014–2026
T1: 12,099 | T2: 173K | T3: rest
TOTAL
~5.8M words processed
2005–2026


Temporal Tier Model
The temporal tier system ensures Voice OS generates content that sounds like Mitchell in 2025–2026, not like Mitchell in 2015. Older patterns are treated as evolution context, not generation targets.

Tier
Years
Generation Weight
Override Rule
Tier 1 (Current)
2024–2026
Highest — primary source
If pattern exists in T1, use it. Full stop.
Tier 2 (Validation)
2021–2023
Moderate — confirms stability
Use to validate that T1 patterns are stable, not just temporary
Tier 3 (Evolution)
Pre-2021
Low — context only
Never replicate. Understand how voice has changed.

Voice Dimension Calibration
Six baseline dimensions define Mitchell’s voice. These are derived from corpus analysis and validated against psychological frameworks.

Dimension
Score
Corpus Basis
Adjustment Range
Directness
0.87
Inverted pyramid structure in 80%+ of emails; main point in sentence 1
±0.15 based on audience
Structure Density
0.81
67.1% bullet usage in Tier 1 emails; 18.4% bold usage
±0.30 based on channel
Warmth
0.62
Kindness VIA #2; "No worries", "Hope you have a great weekend"
±0.30 based on relationship
Formality
0.44
50.4% contraction rate in Tier 1; 4.3% full signature
±0.25 based on context
Precision
0.82
Specific dates, links, data in near-all professional emails
±0.10
Assertiveness
0.76
Direct requests; confident framing without aggression
±0.15 based on audience

Psychological Architecture
Voice OS is the only voice AI system grounded in six validated psychological frameworks simultaneously. These aren’t decorative — they’re active generation priors.

Framework Integration
Framework
Result
Generation Implication
Enneagram
4w3 — The Aristocrat
Authenticity imperative: voice must feel genuinely his, not generic. Recognition of self, not just work, matters.
MBTI
INTJ-T (88% Turbulent)
Strategic framing; structure as emotional regulation; self-critical quality checking
Big Five
Openness 96th %ile
Novel framings preferred; elegant over conventional solutions
Big Five
Extraversion 90th %ile
Exclamation points are authentic, not performative
DISC
Results-Oriented
Direct, dislikes filler, leads with the point
VIA Top 5
Beauty/Excellence, Kindness, Creativity, Learning, Curiosity
Outputs should be elegant AND genuinely caring; warmth is real, not strategic

The Core Tension
4w3 Dynamic (The Aristocrat)
Fours want to be valued for who they ARE
Threes want to be valued for what they ACHIEVE
Mitchell lives in the gap: recognition that validates only the work feels hollow
Generation implication: outputs that read as purely strategic or performative fail. Warmth and authenticity must be real.

Status Dysregulation Pattern (Generation Constraint)
Mitchell’s composure challenges stem from treating professional interactions as status threats
This triggers accommodation behaviors: pre-emptive surrender, excessive apologizing, thinking out loud, credibility citation
Voice OS must NOT generate these patterns — they temporarily reduce anxiety but erode authority
Anti-patterns documented in Voice_OS_Anti_Patterns.md and Voice_OS_Psychological_Operations.md

Known Gaps & Development Roadmap
Voice OS tracks its own gaps. Current alignment is ~78%. Here’s the gap inventory and projected resolution.

Current Gap Inventory
Gap
Severity
Status
Alignment Impact
No retrievable Tier 1 email samples
Critical
In progress
+4% when resolved
No job application materials
Critical
Not started
+3% when resolved
Psychological insights not operationalized
High
Partially resolved
+2% when resolved
No feedback/correction mechanism
High
Template exists
+3% when resolved
No recipient-specific calibration
High
Template exists
+3% when resolved
No edge case handling
High
Partially resolved
+3% when resolved
No QA checklist
Medium
Resolved
+1% when resolved
No anti-pattern examples
Medium
Resolved
+1% when resolved
No domain vocabulary
Medium
Resolved
+1% when resolved
No Slack/Chat samples
Medium
Not started
+1% when resolved

Alignment Projection
Phase
Timeline
Alignment
Current state
March 2026
~78%
After critical gap closure
+3 weeks
~86%
After full build-out
+6 weeks
~92%
Steady-state refinement
Ongoing
95%+

Dependencies & Infrastructure

Dependency
Type
Role
Replacement Risk
Anthropic Claude
Foundation model
All language generation
High — system prompt is Claude-specific
claude.ai
Interface
Project management, KB storage
Medium — could migrate to API with effort
Claude Project system
Feature
Persistent KB attachment
Low — core Claude feature
16-document Knowledge Base
Data
Voice calibration
None — built and maintained by Mitchell

Context Window Management
Voice OS’s full knowledge base, when retrieved simultaneously, approaches the 200K token context limit. The compact.json optimization reduces token usage by ~40% vs. the full summary, making it the primary reference for standard generation tasks.

Document
Estimated Tokens
When Retrieved
voice_os_compact.json
~30K
Every generation
voice_os_complete_summary.json
~60K
Deep calibration only
Psychological Narrative
~15K
Novel situations
Social Media Analysis
~12K
Personal register requests
Domain Vocabulary
~5K
Google/work contexts
All other documents
~5K avg
Contextually

Voice OS System Architecture v4.0 — Internal Reference — March 2026