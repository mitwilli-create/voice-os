# Ingestion Layer

The ingestion layer turns local personal-data exports into a deduplicated,
provenance-tagged chunk store, then renders that store into the corpus
format the scoring layer (`score.py`, `pipeline.py`) consumes.

It consolidates the three legacy Google Drive extractors (preserved
verbatim in `ingest/legacy/`) and extends them to email, text messages,
documents, and video transcripts.

## Data flow

```
raw exports (never copied into the repo)
      |  adapter.iter_records()          ingest/adapters/*
      v
normalize: mojibake fix, self-filter,   ingest/normalize.py
timestamps, whitespace
      |
      v
dedup: sha256 of normalized text;       ingest/dedupe.py
seen hashes skipped (incremental)
      |
      v
tier + enrich: temporal tier,           ingest/tiering.py, ingest/enrich.py
channel/audience/goal/tone tags
      |
      v
corpus/chunks/<source>.jsonl            provenance-rich store (gitignored)
corpus/manifest.json                    hash index + run ledger
      |  python -m ingest export        ingest/export.py
      v
corpus/voice_corpus.txt                 consumed by score.py / pipeline.py
```

## Usage

```bash
cp ingest.example.json ingest.local.json   # then fill in your local paths
git config core.hooksPath .githooks        # enable the privacy pre-commit hook

python -m ingest run                       # all configured sources, incremental
python -m ingest run --source instagram    # one source
python -m ingest run --source email --full # rebuild one source from scratch
python -m ingest export                    # render corpus/voice_corpus.txt
python -m ingest status                    # manifest summary
```

A new export drop only appends: content whose normalized-text hash is
already in the manifest is skipped, so re-running over old exports plus a
new archive is cheap and safe. Every run appends a ledger entry to the
manifest and writes a report under `corpus/runs/`.

## Sources

| Adapter | Reads | Notes |
|---|---|---|
| `instagram` | Instagram data exports | captions, comments, self DMs; mojibake fix |
| `facebook` | Meta/Facebook exports | Messenger (inbox + archived), posts, comments, stories |
| `email` | Gmail `.mbox` | sent mail only; quoted replies and signatures stripped |
| `messages` | combined-sent and iMessage text exports | iMessage needs `identity.phone_numbers` to attribute lines |
| `documents` | directories of `.txt` / `.md` / `.docx` | paragraph-chunked; date from filename or mtime |
| `video` | `.srt` / `.vtt` / `.txt` transcripts | words-per-minute pacing recorded; transcription via a manual whisper hook (see `ingest/adapters/video.py`) |

## Chunk schema (the scoring contract)

Each line of `corpus/chunks/<source>.jsonl` is one chunk:

- `id`, `hash`: sha256 of the whitespace-normalized text
- `tier`: temporal tier 1 to 4, computed with `voice_os.corpus.tier_for_year`
  (undated content is tier 4 and carries zero generation weight)
- `provenance`: `source_type`, `origin_file`, `export_id`, `timestamp`,
  `extractor` (adapter@version), giving full traceability to the original file
- `context`: `channel` and `audience` from the `voice_os.calibration`
  vocabulary (so exported headers feed the register calibration matrix
  directly), plus raw `medium`, `relationship_hint`, heuristic `goal`, and
  `tone_signals` (exclamation density, question ratio, emoji count,
  sentence length, caps ratio); `inference` records how tags were derived
  so a later model-based pass can retag without re-ingesting

## Privacy

Local-first: adapters read raw exports in place; nothing personal is
committed. Three layers keep corpus data out of git:

1. `.gitignore` blocks `corpus/`, `sources/`, `*.jsonl`, `*.mbox`,
   `ingest.local.json`
2. `.githooks/pre-commit` hard-fails on those paths and on any staged file
   over 1MB (enable with `git config core.hooksPath .githooks`)
3. Test fixtures are synthetic (a fictional "Test Person"); no real data
   is used in tests

`ingest.local.json` holds machine-specific paths and identity aliases and
never enters git; `ingest.example.json` is the committed template.

A gitignored `sources/` directory inside the working copy is the suggested
home for raw export copies (social exports, mbox, message exports, scripts,
transcripts): the whole system lives in one folder while the ignore rules
and hook keep every byte of it out of git.
