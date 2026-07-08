# Callable Layer: LangGraph Product API Design

Status: design (PR 1 of the callable-layer sequence). Implementation
follows in a separate PR once this design is reviewed.

## Goal

Give other agents and systems a reliable, high-level way to invoke
Voice OS without knowing anything about its internals:

```python
import voice_os

result = voice_os.draft(
    text="...",
    channel="email",
    audience="boss",
    situation="high_stakes",
    goal="set_expectations",
)
```

Under the hood the call runs a LangGraph `StateGraph` that orchestrates
draft generation, adversarial critique, QA gating, and a bounded
revision loop, with SQLite-checkpointed runs for inspection and replay.

## Design principles

1. **Compose, do not replace.** The tested core stays authoritative:
   `VoiceModel.query()` resolves context to targets, `GenerativePersona`
   / `AdversarialPersona` do the writing work, `gate_extended()` makes
   the quality decision. Graph nodes are thin orchestration over these.
2. **LangGraph stays contained.** The core package remains stdlib-only
   and offline-deterministic. All LangGraph code lives in a new
   `voice_os/product/` submodule behind guarded imports; `import
   voice_os` and the existing test suite pass with langgraph absent.
3. **State is plain data.** Everything in `VoiceState` is
   JSON/msgpack-serializable (str, int, float, bool, list, dict, None)
   so `SqliteSaver` can checkpoint every step. No dataclasses or model
   handles in state.
4. **Locked surfaces untouched.** `run_pipeline` (golden-regression
   locked), `run_cycles`, `calibrate()`, and `gate()` keep their exact
   signatures and behavior.
5. **Privacy first.** Drafts, KB snapshots, and run checkpoints are
   personal data. All of it lives under a new gitignored `var/`
   directory. Nothing personal enters git.

## Package layout

```
voice_os/product/
    __init__.py     public wrappers: draft(), run_history(), describe_graph(),
                    kb helpers. Lazy-imports graph.py so stdlib-only callers
                    never touch langgraph.
    state.py        VoiceState TypedDict + result envelope (stdlib only)
    aliases.py      friendly-alias normalization (stdlib only)
    kb.py           KB loader + snapshot versioning (stdlib only)
    graph.py        LangGraph StateGraph build + node functions
                    (imports langgraph; the only module that does)
    cli.py          argparse CLI behind python3 -m voice_os
                    (stdlib-only at import; graph loads on command run)
voice_os/__main__.py    package entry point delegating to product/cli.py
```

Top-level exposure via PEP 562 module `__getattr__` in
`voice_os/__init__.py`: attribute access to `draft`, `run_history`,
`describe_graph`, `load_kb`, `snapshot_kb`, or `list_kb_snapshots`
lazily imports `voice_os.product`. With langgraph not installed,
`import voice_os` still succeeds; calling `voice_os.draft()` raises an
`ImportError` naming the missing dependency and the install command.

## VoiceState

Follows the agreed starter structure, extended with the calibration
data the nodes need (seeded once by the prepare node) and the
observability fields the mission requires. All values serializable.

```python
class VoiceState(TypedDict):
    # request (set once at invoke; all post-normalization canonical values)
    input_text: str
    channel: str
    audience: str
    situation: str
    goal: str
    stakes: str        # always canonical here; None never reaches state
    medium: str | None
    max_revisions: int

    # calibration (seeded by the prepare node; primitives only)
    target_profile: dict        # axis -> float
    baseline_mean: dict
    baseline_std: dict
    tone_mean: dict | None      # mined tone norms when available
    tone_std: dict | None
    banned: list[str]           # hand list + mined n-grams, merged
    guidance: list[str]         # goal/stakes guidance strings
    kb_meta: dict               # snapshot id, files, hashes, status

    # working
    current_draft: str
    critique_feedback: str      # newline-joined adversarial findings
    qa_decision: Literal["pass", "revise", "reject"]
    revision_count: int
    banned_hits: list[str]
    persona_modes: list[str]    # "live" / "offline" observed across nodes

    # observability (append-only reducers)
    revision_history: Annotated[list[str], operator.add]
    fidelity_scores: dict       # {"overall": float, "per_axis": {...}}
    trace_notes: Annotated[list[str], operator.add]
```

## Graph topology

```
START -> prepare -> generate -> critique -> qa_gate
                                   ^            |
                                   |       (conditional)
                                 revise <-- "revise"
                                            "pass"   -> END
                                            "reject" -> END
```

Node responsibilities (each returns a partial state update):

- **prepare**: loads `VoiceModel` (module-level cached), runs
  `model.query(**context)`, copies the serializable parts of the
  `QueryResult` into state (target profile, baseline mean/std, tone
  mean/std, merged banned list, guidance). Loads the KB bundle, ensures
  a content snapshot exists (see KB section), records `kb_meta`, and
  appends a trace note with the KB status and drift flags if any.
- **generate**: first-pass voice transformation.
  `GenerativePersona.revise(input_text, target, banned, guidance)`
  produces `current_draft`. Records the persona mode and a trace note.
- **critique**: `AdversarialPersona.critique(current_draft, target,
  banned)`. Findings become `critique_feedback` (newline-joined) and a
  trace note. An empty finding list means the adversary passed it.
- **qa_gate**: scores the draft (`score_text`), runs `gate_extended`
  against the target with tone norms when present, and maps the gate
  decision to the product-layer vocabulary:
  - gate `pass` -> `qa_decision = "pass"`
  - gate `cycle` and `revision_count < max_revisions` -> `"revise"`
  - gate `cycle` and revisions exhausted -> `"reject"` (new product
    semantics: best-effort draft returned, flagged as below threshold)
  Writes `fidelity_scores`, `banned_hits`, and a trace note either way.
- **revise**: `GenerativePersona.revise(current_draft, target, banned,
  signals)` where signals = the gate's `revision_signals` + guidance +
  carried adversarial findings (prefixed, same convention as
  `run_cycles`). Appends the pre-revision draft to `revision_history`,
  increments `revision_count`, then loops back to critique.

Conditional edge on `qa_decision` from qa_gate: `"revise"` routes to
revise; `"pass"` and `"reject"` route to END. The loop is bounded by
`max_revisions` (default 2, matching `run_cycles`), so the graph always
terminates; `reject` is the terminal state for revisions exhausted
while fidelity is still below `PASS_THRESHOLD` (0.80).

The existing `run_cycles` gates first and revises after; the graph
follows the mission's generate -> critique -> gate -> revise ordering
instead. Both use identical persona and gate calls; `run_cycles` and
its callers are unchanged.

## Public API

```python
voice_os.draft(
    text: str,
    *,
    channel: str = "email",
    audience: str = "peer",
    situation: str = "standard",
    goal: str = "unknown",
    stakes: str | None = None,     # None = "not specified"; see contract below
    medium: str | None = None,
    max_revisions: int = 2,
    run_id: str | None = None,     # supply to make the run resumable/inspectable by name
) -> dict
```

`stakes` contract: the public parameter accepts None only so alias rule
3 can reroute a stakes-shaped situation value. Normalization always
resolves it to a canonical string before anything else runs: an
explicit value is alias-normalized ("high_stakes" -> "high"), a
rerouted situation value supplies it, and otherwise it defaults to
"routine" (the `VoiceContext` default). `VoiceContext` is then
constructed and validated with the canonical string, so
`VoiceState.stakes` is always a required, validated `str`, exactly as
the existing context model demands. None never reaches the graph or
the persisted state.

Returns a JSON-safe result envelope:

```python
{
    "run_id": "...",               # thread id in the checkpoint store
    "decision": "pass" | "reject",
    "output_text": "...",
    "fidelity": {"overall": 0.87, "per_axis": {...}},
    "revisions": 1,
    "revision_history": ["..."],
    "banned_hits": [],
    "mode": "live" | "offline",
    "context": {...},              # normalized VoiceContext dict
    "kb": {...},                   # snapshot id + file hashes or {"status": "absent"}
    "trace": ["..."],
}
```

Companion functions:

- `voice_os.run_history(run_id)`: returns the checkpoint sequence for a
  run (step index, node, qa_decision, fidelity, revision_count) read
  back from the SqliteSaver store. Powers inspection and debugging.
- `voice_os.describe_graph()`: returns the compiled graph's structure
  (nodes + edges, mermaid text) for documentation and demos.
- `voice_os.load_kb()`, `voice_os.snapshot_kb()`,
  `voice_os.list_kb_snapshots()`: see KB section.

## CLI entry point (cli.py, cross-instance calling pattern)

The same API is reachable from any shell, which makes Voice OS callable
from other agents and Claude Code sessions running in other working
directories without importing the package:

```bash
cd ~/Documents/voice-os
python3 -m voice_os draft --channel email --audience boss \
  --situation high_stakes --goal set_expectations <<'EOF'
quick note to say the launch plan looks right, one timing question
EOF
```

- Draft text arrives on stdin (heredoc-friendly) or via `--file PATH`.
- The full JSON envelope prints to stdout; `--text-only` prints just
  the drafted text.
- Exit codes: 0 decision pass, 1 decision reject (envelope still
  printed with the best-effort draft), 2 usage, validation, or
  dependency error.
- All `draft()` keyword arguments have flag equivalents, including the
  fixture-path overrides (`--corpus`, `--mined-dir`, `--banned-path`,
  `--kb-dir`, `--var-dir`), so tests and sandboxed callers never touch
  the personal corpus.
- `python3 -m voice_os history <run_id>` and
  `python3 -m voice_os graph` wrap `run_history()` and
  `describe_graph()`.
- Live mode engages exactly as in the Python API: when
  `ANTHROPIC_API_KEY` resolves in the environment and
  `VOICE_OS_OFFLINE` is unset. Callers that need live drafting should
  check `"mode": "live"` in the envelope rather than assuming.

The recommended cross-instance pattern is a thin wrapper (for Claude
Code, a user-level skill) that exports the key, cd's to the repo, runs
the command above, and interprets the envelope. The CLI itself stays
dumb on purpose: no key discovery, no config files, no network logic
beyond what the pipeline already does.

## Alias normalization (aliases.py)

The calibrator fails fast on unknown vocabulary, so the wrapper
normalizes friendly forms before constructing `VoiceContext`. Rules:

1. Lowercase, strip, convert underscores to hyphens.
2. Apply the alias table.
3. If a situation value names a stakes level (for example
   `high_stakes`), reroute it: `stakes` takes the level, `situation`
   falls back to `standard` (only when the caller did not pass an
   explicit situation and stakes of their own).
4. Anything still unknown raises `ValueError` listing the canonical
   vocabulary (same failure the calibrator gives today).

Alias table (initial; extend as callers surface new forms):

| Dimension | Alias | Canonical |
|---|---|---|
| audience | boss, manager, exec, executive | leadership |
| audience | coworker, colleague, teammate | peer |
| audience | report, direct-report (underscored form) | direct-report |
| audience | client, customer, vendor, partner | external |
| audience | friend, family | friend-family |
| audience | recruiter, hiring-manager | job-seeking |
| audience | connection, network | networking |
| channel | slack, teams, dm, im | chat |
| channel | sms, imessage | text |
| situation | followup | follow-up |
| situation | apology, error, mistake | error-ack |
| situation | badnews | bad-news |
| situation/stakes | high-stakes, critical-stakes, low-stakes, routine | stakes=high/critical/low/routine |
| goal | set-expectations (underscored form) | set-expectations |
| goal | deescalate | de-escalate |

Mitchell's example call maps to: channel=email, audience=leadership,
situation=standard, stakes=high, goal=set-expectations.

## Persistence (SqliteSaver)

- Database: `var/runs.sqlite` (override with `VOICE_OS_VAR_DIR`). The
  `var/` directory is gitignored (entry added in this design PR, ahead
  of any code writing there); checkpoints
  contain draft text, which is personal data.
- One `thread_id` per `draft()` call: caller-supplied `run_id` or a
  generated UTC-timestamp + random-suffix id.
- The saver is constructed over a `sqlite3` connection
  (`check_same_thread=False`) held by the product module; the graph is
  compiled once per process with the checkpointer attached.
- `run_history(run_id)` reads checkpoints via the saver's `list()` API
  and projects them to a compact, JSON-safe summary. Full state replay
  stays possible with LangGraph tooling against the same database.

## KB loading and snapshot versioning (kb.py)

Source of truth: the legacy claude.ai Projects KB under
`sources/drive-voice-os/` (gitignored symlink), overridable with
`VOICE_OS_KB_DIR`. Two files are loaded:

- **Compact KB**: newest file matching `*voice-os-compact*.json`.
- **System prompt**: files matching `*System-Instructions*.md`; the
  loader parses the `System Instructions v<major.minor>` header and
  picks the highest version (v5.0 wins over the v4.0 variant today),
  tie-breaking on filename.

`load_kb()` returns a bundle: parsed compact KB dict, system prompt
text, file list with sha256 content hashes, and a combined bundle hash.
If the KB directory or files are missing, the bundle reports
`status: "absent"`; `draft()` proceeds without KB and says so in
`trace_notes` and the result envelope. The loader never invents
content.

Versioning: `snapshot_kb()` copies the KB files to
`var/kb_snapshots/<UTC-timestamp>/` with a `manifest.json` (file names,
sizes, sha256s, bundle hash, source dir). The prepare node calls an
`ensure_snapshot()` helper: if no existing snapshot manifest matches
the current bundle hash, a new snapshot is taken. Every run's `kb_meta`
records the snapshot id and bundle hash, so any historical run can be
traced to the exact KB content it saw. `list_kb_snapshots()` lists
manifests. Snapshots live under `var/` and never enter git.

The v1 request path records and versions the KB; it does not yet parse
KB pattern data into generation prompts. Deeper fusion (feeding
tier-1 pattern data or the system prompt into the live personas) needs
a persona API extension and is deliberately out of scope here; the
snapshot plumbing this PR designs is what makes that step auditable
later.

## Dependency containment

- New optional runtime deps for the product layer: `langgraph` +
  `langgraph-checkpoint-sqlite` (verified installed and importing as
  langgraph 1.2.8 / checkpoint-sqlite 3.1.0 on the dev machine, Python
  3.11). They join the already-optional `anthropic` SDK in
  `requirements.txt`; the core pipeline keeps running on the stdlib
  alone, with the anthropic and langgraph integrations both degrading
  gracefully when absent.
- `requirements.txt` gains a clearly-marked optional product-layer
  section: the core needs none of it; `voice_os.draft()` requires the
  langgraph pair.
- `voice_os/product/graph.py` is the only module importing langgraph.
  `state.py`, `aliases.py`, `kb.py` are stdlib-only and importable
  everywhere.
- Tests for the graph use `pytest.importorskip("langgraph")`; alias,
  state, and KB tests run in all environments. The existing 127 tests
  must stay green with or without langgraph installed.

## Testing plan (offline-deterministic, synthetic fixtures only)

- `tests/test_product.py`:
  - alias normalization: every table row + underscore handling +
    stakes rerouting + ValueError on garbage.
  - KB loader against a temp fake KB dir (synthetic "Test Person"
    content): version-header selection, hashing, absent-dir status.
  - snapshot versioning: first call snapshots, unchanged content does
    not re-snapshot, changed content does; manifest hashes verify.
  - graph run (importorskip): full `draft()` under `VOICE_OS_OFFLINE=1`
    with the synthetic corpus fixture; asserts decision in
    {"pass", "reject"}, bounded revisions, serializable envelope,
    `run_history()` returns steps, checkpoint db lands under a tmp
    `VOICE_OS_VAR_DIR`, and `git status` stays clean (no personal-data
    paths created inside the repo tree beyond gitignored var/).
  - reject path: unreachable target profile forces revisions to
    exhaust; asserts `decision == "reject"` and best-effort text.
- Environment note carried from the core: `VOICE_OS_OFFLINE` is a
  truthy check, so tests set `VOICE_OS_OFFLINE=1` and never `=0`.

## Determinism

The callable layer inherits the project-wide determinism contract:
offline runs are byte-reproducible given the same inputs, live runs
are reproducible-in-inputs via provenance stamping. Guarantees, audit
procedure, scheduled hardening, and the rules every new module must
follow are in docs/determinism.md.

## Out of scope (parked)

- KB pattern data fused into persona prompts (needs persona API change).
- Async / streaming graph execution.
- Multi-draft variants (architecture doc "Mode 1" 2-3 variants).
- Replacing `run_pipeline` / `run_cycles`; they remain the legacy and
  library entry points.
