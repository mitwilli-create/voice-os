# Silent partial sources: diagnosis and fixes

Three sources that the corpus counted as "integrated" were silently
partial or absent. This doc records what was actually broken for each
one, the root cause found by reading the real exports, the fix, and the
exact re-ingest commands to run once the measurement freeze on
`corpus/` lifts. All diagnosis was done by reading export files and
running adapter iterators in memory; nothing under `corpus/` or
`sources/` was touched.

## 1. iMessage line export: 0 chunks

**Symptom.** `Export_Messages_iMessages_2026-01-19.txt` contributed
nothing; the run warning reported 65,208 lines skipped as "other
senders". `ingest.local.json` had `"phone_numbers": []`.

**What the export actually is.** A flat, chronological log of every
chat, one line per message:
`[YYYY-MM-DD HH:MM:SS] sender: text`, with bare continuation lines.
65,208 sender-labeled lines, 601 distinct senders. Reading real
conversations end to end shows what the sender field actually carries:

- In one-to-one chats, BOTH directions are stamped with the
  conversation partner's handle. The same number asks a question at
  11:02 and answers it at 11:04; tapback lines appear under the same
  label as the message they react to. Direction is simply not encoded
  for these lines, so they can never be safely attributed.
- In group chats, each line carries the true sender, and the account
  owner's lines are labeled with the literal string `Me` (877 lines,
  of which 200 are tapback reactions and 677 are authored prose).
- The owner's own phone number (verified against the phone number in
  the Facebook and Instagram profile exports) appears on zero lines.

**Sender frequency distribution (numbers masked to last 4).** In a
correctly sender-labeled personal archive the owner would dominate.
Here no sender does, which is what proves the one-to-one labels are
conversation handles rather than authors:

| Sender | Lines | Share |
|---|---|---|
| ****8643 | 4,922 | 7.5% |
| ****5507 | 3,692 | 5.7% |
| ****9550 | 2,872 | 4.4% |
| ****5194 | 2,825 | 4.3% |
| ****7010 | 2,615 | 4.0% |
| ****1830 | 2,561 | 3.9% |
| ****2600 | 2,422 | 3.7% |
| ****5432 | 1,774 | 2.7% |
| `Me` | 877 | 1.3% |

**Root causes in the adapter.**

1. `is_self` in `ingest/normalize.py` matched phone numbers by raw
   substring, so a configured `+1 (555) 000-1111` could never match the
   export's `+15550001111`, and an empty `phone_numbers` list meant
   nothing could ever match.
2. `parse_imessage` did not know the exporter's `Me` label, so the only
   provably self-authored lines in the file were counted as "other
   senders".
3. `MessagesAdapter.iter_records` skipped the whole file when
   `identity.phone_numbers` was empty, so even a fixed parser would
   never have run.

**Fixes** (`ingest/normalize.py`, `ingest/adapters/messages_txt.py`):

- Phone matching is now digit-normalized: `+1` prefixes, dashes,
  spaces, dots, and parentheses are ignored, and a trailing-10-digit
  match handles country-code asymmetry. It only applies when the sender
  string itself is phone-shaped, so digits embedded in spam email
  handles never match.
- `sender == "Me"` is treated as self-authored by definition of the
  format, independent of identity config.
- Tapback lines (`Loved "..."`, `Liked "..."`, `Laughed at "..."`,
  `Emphasized`, `Disliked`, `Questioned`, and the attachment forms such
  as `Loved an image`) are dropped: they quote other people's words.
  Prose that merely starts with a reaction verb ("Loved seeing you
  boys") is kept.
- The file is no longer skipped when `phone_numbers` is empty; a
  warning explains that only `Me` lines were attributed.
- The owner's mobile number (from the Meta profile exports) was added
  to the gitignored `ingest.local.json` so future, correctly labeled
  exports attribute by number. Real identity values stay out of git;
  `ingest.example.json` already documents the field shape.

**Expected yield after re-ingest:** 678 self-authored iMessage records
(677 group-chat `Me` prose lines plus one reaction-verb-prefixed real
message), before content-hash dedup. The 64,331 one-to-one lines remain
skipped on purpose: ingesting them would put other people's words in
the voice corpus. Recovering the owner's side of one-to-one chats needs
a direction-aware export (for example imessage-exporter against
chat.db), which is a separate task.

## 2. Instagram posts, stories, and comments: 0 chunks

**Symptom.** `ig_post`, `ig_story`, and `ig_comment` produced nothing
across all three configured export drops, while `instagram_dm` worked
(152,890 records).

**Root causes.** Meta moved and reshaped the content files; the
adapter was looking for the older layouts:

1. Discovery: posts and stories now live at
   `your_instagram_activity/media/posts_1.json` and
   `your_instagram_activity/media/stories.json`. The adapter searched
   `content/`, `your_instagram_activity/content/`, and `media/posts/`
   directories, and its recursive fallback only accepted files with a
   literal `content` or `posts` directory in the path, so everything
   under `your_instagram_activity/media/` was filtered out.
2. Comment payloads: 2026 exports carry each comment in
   `string_map_data` keyed `Comment` (value) and `Time` (timestamp).
   The adapter only read the older `string_list_data` list. Reels
   comments also moved under a new top-level key,
   `comments_reels_comments`, which the adapter did not know.
3. Album captions: multi-photo posts store empty per-media `title`
   fields with the real caption on the item; the adapter returned the
   first media `title` even when empty, losing the item-level caption
   (10 of 461 posts in the primary export).

**Fixes** (`ingest/adapters/instagram.py`):

- Discovery accepts `your_instagram_activity/media/` and
  `your_instagram_activity/posts/`, and the recursive fallback accepts
  a `media` directory alongside `content` and `posts`. Lookalike files
  outside content directories (for example
  `logged_information/past_instagram_insights/posts.json`) are still
  excluded.
- Comment extraction reads `string_map_data` (`Comment` value, `Time`
  timestamp) as a fallback after `string_list_data`, and
  `comments_reels_comments` joins the known list keys.
- `_caption_and_ts` only takes a non-empty media title and otherwise
  falls back to the item-level title or caption.

**Expected yield after re-ingest** (raw records before dedup across
the overlapping export drops): 910 `ig_post`, 1,964 `ig_story`, 1,936
`ig_comment`. DM counts are unchanged.

## 3. Facebook: never ingested

**Symptom.** `sources/social/facebook-export` is configured and the
`FacebookAdapter` is registered, but the manifest has no facebook
chunks.

**Root cause.** Operational, not structural: all seven recorded runs in
`corpus/runs/` used `--source instagram`, `--source email`, or
`--source messages`. No run ever selected facebook, and no full
`python -m ingest run` (which would include every available source) was
ever executed. Tested in memory against the real export, the adapter
already yields 110,327 `messenger`, 380 `messenger_archived`, 17,003
`fb_post`, and 6,332 `fb_comment` records.

**One structural mismatch found and fixed** while verifying: the
stories file keys its list `archived_stories_v2`, which the adapter did
not know. The 865 items in the real file contain no authored text at
all; every `title` is the auto-generated archive notice "A photo from
Mitchell Williams's story was added to his archive. Shared from
Instagram." The fix (`ingest/adapters/meta_facebook.py`) reads the
`archived_stories_v2` key and filters archive-notice boilerplate, so
this export correctly yields zero `fb_story` records while a future
export with real story text will ingest.

## Tests

`tests/test_ingest.py` gained six regression tests, each written to
fail against the pre-fix adapters using synthetic fixtures that
reproduce the real-format mismatches:

- `test_is_self_matches_phone_number_formats`
- `test_imessage_me_lines_attributed_and_reactions_dropped`
- `test_imessage_phone_identity_attributes_sender_lines`
- `test_imessage_file_not_skipped_without_phone_identity`
- `test_instagram_2026_activity_layout`
- `test_facebook_archived_stories_v2_skips_archive_notices`

Full suite: 265 passed. Goldens untouched.

## Re-ingest commands (run after the corpus freeze lifts)

The `message_sent` provenance migration (docs/doc-types.md, merged via
PR #19) requires a full rebuild of the messages source; incremental
mode would dedupe every combined-sent chunk by content hash and leave
the old `email_sent` stamps in place. Instagram and facebook are safe
incrementally: new post, story, and comment chunks append, and the
existing DM chunks dedupe as duplicates.

```bash
# from the repo root, with the frozen measurement campaign concluded
python3 -m ingest run --source messages --full   # restamps message_sent, adds ~678 imessage chunks
python3 -m ingest run --source instagram         # appends ig_post / ig_story / ig_comment
python3 -m ingest run --source facebook          # first facebook ingest (~134k records before dedup)
python3 -m ingest export                         # refresh corpus/voice_corpus.txt for score.py / pipeline.py
python3 -m ingest status                         # confirm per-source chunk counts
```
