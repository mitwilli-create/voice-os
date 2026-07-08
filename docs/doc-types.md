# Document Subtypes (doc_type)

The corpus is about to gain long-form professional material: broadcast
scripts, segment intros, interview questions, program plans, CVs, cover
letters, impact docs, writing samples, and on-camera transcripts. Before
this change, every one of those landed in the chunk store as an
undifferentiated `document` or `video_transcript` with channel `doc`, so
the mining layer could not tell a CV apart from a show script. A cover
letter and a segment intro are different registers of the same author;
collapsing them into one cell would blur exactly the distinctions the
per-context profiles exist to capture.

This change adds a `doc_type` subtype dimension end to end: schema,
adapters, and mining.

## Design

### Schema

`Context` in `ingest/schema.py` gains one field:

```python
doc_type: str = ""
```

Empty string means "no subtype": every chat, email, and social chunk, and
every chunk ingested before this field existed. The default keeps old
chunk stores loadable through `Chunk.from_dict`, which constructs
`Context(**data)`; a stored context dict without the key simply gets the
default. Every reader of stored context dicts in the runtime and the
mining layer uses tolerant `dict.get` access, so no reader treats the
missing key or the empty value as an error.

`RawRecord` in `ingest/adapters/base.py` gains the matching field, and
`build_context` in `ingest/enrich.py` passes it through, following the
same pattern as `relationship_hint`.

### Documents adapter: folder name is the subtype

The first-level folder name under each configured dir becomes the
`doc_type`, verbatim. The expected layout is:

```
sources/documents/
  scripts/
  segment-intros/
  interview-questions/
  program-plans/
  cv/
  cover-letters/
  impact-docs/
  writing-samples/
```

Any other folder name passes through unchanged as the `doc_type`, so new
subtypes need no code change, only a new folder. Files placed directly in
the configured dir get `doc_type` "". Only the first path component
counts: `cover-letters/2025/letter.md` is still `cover-letters`.

Filenames stay the date carrier (`YYYY-MM-DD` prefix, else mtime), exactly
as before; the folder carries the type, the filename carries the date.

### Video adapter: on-camera transcripts

Every transcript chunk is stamped `doc_type: "on-camera"`. Two further
adjustments make the adapter fit the real transcript inventory (plain
whisper `.txt` output plus some `.srt`):

- Transcripts are now paragraph-chunked with the same `max_chunk_words`
  bound (default 400) as the documents adapter, instead of one chunk per
  file. A full episode transcript is far too long to be a single chunk.
- Words-per-minute pacing is computed only for timed formats (`.srt` /
  `.vtt`), where cue timestamps make it honest, and is carried on every
  chunk of the file as before (it is a file-level pacing signal). Plain
  `.txt` has no timing info and therefore no WPM.

Provenance semantics are unchanged: `source_type` stays
`video_transcript`, medium stays `spoken`, audience stays `external`.

### Messages provenance fix: message_sent

The combined-sent text-message export (blocks of `FILE:` / `SUBJECT:` /
`DATE:` headers over a body) was stamped `source_type: "email_sent"`,
which made its 679 chunks indistinguishable from Gmail mbox chunks. The
adapter's own docstring describes it as a text-message export, so those
chunks now get:

- `source_type: "message_sent"`
- channel `text`, medium `sms` (the same cell as iMessage, which is the
  honest register for sent texts; `source_type` still distinguishes the
  two exports)
- audience `friend-family` via the existing text-channel heuristic

### Mining: doc_type as a grouping dimension

`mine_context_profiles` in `mine/tone_norms.py` gains a `doc_types` group
alongside `audiences`, `media`, `goals`, and `pairs`, following the
existing pattern exactly: chunks with an empty or absent `doc_type`
contribute no group, the `min_chunks` support gate applies, and
`group_profile` in `voice_os/mined.py` looks the groups up with the same
tolerant access as every other kind.

## Compatibility and migration

**Old chunk stores** load unchanged: the missing `doc_type` key defaults
to "". No re-ingest is required for existing sources.

**Old mined artifacts** (no `doc_types` key) keep loading and serving:
`group_profile` returns None for absent kinds and the runtime falls back
to the hand tables, its normal degraded path. New artifacts with a
`doc_types` key are ignored by runtimes that do not ask for it.

**The 679 combined-sent chunks need a one-shot rebuild.** Incremental
re-ingestion dedupes by content hash (`corpus/manifest.json` records the
sha256 of every chunk's normalized text; `Manifest.seen` skips known
hashes regardless of source_type). Re-running the messages adapter after
this change therefore skips all 679 existing chunks as duplicates and the
old `email_sent` stamp survives on disk. The migration is a full rebuild
of that one source:

```bash
python -m ingest run --source messages --full
```

`--full` drops the messages source's hashes from the manifest, deletes
`corpus/chunks/messages.jsonl`, and re-ingests from the raw export, so
the combined-sent chunks come back as `message_sent` (and iMessage chunks,
if configured, are rebuilt in the same pass). No other source is touched.
Until that rebuild runs, existing chunks keep the old stamp and mining
keeps grouping them under email; nothing breaks, the split just is not
visible yet. The same applies to `doc_type` on any documents or video
chunks ingested before this change, but both of those chunk stores are
currently empty, so in practice only the messages rebuild matters.

**Hash note:** the content hash covers normalized text only, not context
tags, so a `--full` rebuild reassigns the same ids and hashes; downstream
consumers keyed on chunk id are unaffected.
