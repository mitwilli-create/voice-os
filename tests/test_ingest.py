"""Ingestion-layer tests. All fixture data is synthetic (a fictional
"Test Person"); no real corpus content is used or committed."""

import mailbox
import os

from ingest.adapters.email_mbox import EmailMboxAdapter, strip_quoted
from ingest.adapters.instagram import InstagramAdapter
from ingest.adapters.messages_txt import parse_combined_sent, parse_imessage
from ingest.adapters.meta_facebook import FacebookAdapter
from ingest.adapters.video import parse_timed_transcript
from ingest.cli import run_source
from ingest.dedupe import Manifest
from ingest.enrich import build_context
from ingest.export import export_corpus
from ingest.normalize import clean_text, decode_meta_text, normalize_for_hash
from ingest.schema import content_hash
from ingest.tiering import tier_for_chunk_year
from voice_os.calibration import AUDIENCES, CHANNELS
from voice_os.corpus import parse_corpus

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

TEST_IDENTITY = {
    "names": ["Test Person"],
    "usernames": ["testperson"],
    "emails": ["test@example.com"],
    "phone_numbers": ["+15550001111"],
}


def make_config(tmp_path, **sources):
    return {
        "identity": TEST_IDENTITY,
        "sources": sources,
        "corpus_dir": str(tmp_path / "corpus"),
        "export": {},
    }


# --- normalize ---------------------------------------------------------


def test_decode_meta_text_fixes_mojibake():
    assert decode_meta_text("cafÃ©") == "café"


def test_decode_meta_text_passes_clean_text_through():
    assert decode_meta_text("already fine") == "already fine"


def test_clean_text_strips_invisibles_and_collapses_whitespace():
    assert clean_text("a​b   c\n\n\n\nd￼") == "ab c\n\nd"


def test_hash_is_whitespace_insensitive():
    a = content_hash(normalize_for_hash("hello   world\n"))
    b = content_hash(normalize_for_hash("hello world"))
    assert a == b


# --- tiering -----------------------------------------------------------


def test_tier_boundaries_match_scoring_layer():
    assert tier_for_chunk_year(2026) == 1
    assert tier_for_chunk_year(2024) == 1
    assert tier_for_chunk_year(2023) == 2
    assert tier_for_chunk_year(2021) == 2
    assert tier_for_chunk_year(2020) == 3
    assert tier_for_chunk_year(2015) == 3
    assert tier_for_chunk_year(2014) == 4
    assert tier_for_chunk_year(None) == 4


# --- adapters ----------------------------------------------------------


def test_instagram_adapter_extracts_only_self_content(tmp_path):
    config = make_config(
        tmp_path, instagram={"paths": [os.path.join(FIXTURES, "instagram_export")]}
    )
    records = list(InstagramAdapter(config).iter_records())
    texts = [r.text for r in records]
    assert any("café" in t for t in texts), "mojibake must be decoded"
    assert not any("not authored" in t for t in texts), "other senders filtered"
    assert not any("shared a story" in t for t in texts), "share noise filtered"
    assert not any(t == "Liked a message" for t in texts)
    assert any(r.source_type == "ig_post" for r in records)
    assert any(r.source_type == "ig_comment" for r in records)
    dm = next(r for r in records if r.source_type == "instagram_dm")
    assert dm.timestamp.startswith("2025")
    assert dm.relationship_hint == "alexexample_123"


def test_facebook_adapter_extracts_messages_and_posts(tmp_path):
    config = make_config(
        tmp_path, facebook={"paths": [os.path.join(FIXTURES, "facebook_export")]}
    )
    records = list(FacebookAdapter(config).iter_records())
    texts = [r.text for r in records]
    assert any("Throwback message" in t for t in texts)
    assert any("newsroom" in t for t in texts)
    assert not any("not authored" in t.lower() for t in texts)
    assert not any("sent an attachment" in t.lower() for t in texts)


def test_mbox_adapter_keeps_sent_only_and_strips_quotes(tmp_path):
    mbox_path = tmp_path / "test.mbox"
    box = mailbox.mbox(str(mbox_path))
    mine = mailbox.mboxMessage()
    mine["From"] = "Test Person <test@example.com>"
    mine["To"] = "Friend <friend@gmail.com>"
    mine["Subject"] = "revised script"
    mine["Date"] = "Tue, 04 Feb 2025 10:00:00 -0800"
    mine.set_payload(
        "Thanks for the notes, sending the revised script now.\n\n"
        "On Feb 3, 2025, at 09:31, Someone wrote:\n> old quoted text\n"
    )
    box.add(mine)
    theirs = mailbox.mboxMessage()
    theirs["From"] = "Someone Else <else@example.com>"
    theirs["Date"] = "Tue, 04 Feb 2025 11:00:00 -0800"
    theirs.set_payload("Not authored by the test identity.\n")
    box.add(theirs)
    box.flush()

    config = make_config(tmp_path, email={"mbox_paths": [str(mbox_path)]})
    records = list(EmailMboxAdapter(config).iter_records())
    assert len(records) == 1
    record = records[0]
    assert "revised script" in record.text
    assert "quoted" not in record.text
    assert record.timestamp.startswith("2025-02-04")
    assert record.relationship_hint == "friend@gmail.com"


def test_strip_quoted_removes_mobile_signature():
    body = "Real content here.\n\nSent from my iPhone\n"
    assert strip_quoted(body) == "Real content here."


def test_combined_sent_parsing():
    with open(os.path.join(FIXTURES, "combined_sent_sample.txt"), encoding="utf-8") as f:
        raw = f.read()
    records = list(parse_combined_sent(raw, "sample.txt", "sample"))
    assert len(records) == 2
    first = records[0]
    assert first.timestamp.startswith("2015-06-12")
    assert "swing Sunday" in first.text
    assert "quoted" not in first.text
    assert "Sent from my iPhone" not in first.text
    assert first.extra["subject"] == "Re: Sunday plans"


def test_imessage_parsing_filters_and_joins_continuations():
    with open(os.path.join(FIXTURES, "imessage_sample.txt"), encoding="utf-8") as f:
        raw = f.read()
    records, skipped = parse_imessage(raw, TEST_IDENTITY, "sample.txt", "sample")
    assert len(records) == 2
    assert skipped == 1
    assert "continuation line" in records[0].text
    assert records[0].timestamp == "2022-09-14T12:10:55"


def test_srt_transcript_parse_and_pacing():
    with open(os.path.join(FIXTURES, "transcript_sample.srt"), encoding="utf-8") as f:
        raw = f.read()
    text, wpm = parse_timed_transcript(raw)
    assert "Good evening" in text and "through town" in text
    assert wpm is not None and 20 < wpm < 26


# --- enrichment --------------------------------------------------------


def test_enrich_emits_calibration_vocabulary():
    context = build_context("instagram_dm", "Are you coming tonight?")
    assert context.channel in CHANNELS
    assert context.audience in AUDIENCES
    assert context.channel == "chat"
    assert context.audience == "friend-family"
    assert context.goal == "request"
    assert context.tone_signals["question_ratio"] >= 0.5


def test_enrich_email_audience_by_domain():
    personal = build_context("email_sent", "Some words here", "friend@gmail.com")
    work = build_context("email_sent", "Some words here", "colleague@company.com")
    assert personal.audience == "friend-family"
    assert work.audience == "peer"


# --- dedup + incremental ----------------------------------------------


def test_incremental_second_run_adds_nothing(tmp_path):
    config = make_config(
        tmp_path, instagram={"paths": [os.path.join(FIXTURES, "instagram_export")]}
    )
    corpus_dir = config["corpus_dir"]
    manifest = Manifest(os.path.join(corpus_dir, "manifest.json"))

    first = run_source(InstagramAdapter(config), manifest, corpus_dir)
    assert first["new"] == 3
    assert first["duplicate"] == 0

    second = run_source(InstagramAdapter(config), manifest, corpus_dir)
    assert second["new"] == 0
    assert second["duplicate"] == 3

    chunks_file = os.path.join(corpus_dir, "chunks", "instagram.jsonl")
    with open(chunks_file, encoding="utf-8") as f:
        assert len(f.readlines()) == 3


# --- export: handoff to the scoring layer ------------------------------


def test_export_roundtrips_into_scoring_corpus(tmp_path):
    config = make_config(
        tmp_path, instagram={"paths": [os.path.join(FIXTURES, "instagram_export")]}
    )
    corpus_dir = config["corpus_dir"]
    manifest = Manifest(os.path.join(corpus_dir, "manifest.json"))
    run_source(InstagramAdapter(config), manifest, corpus_dir)

    out_path = str(tmp_path / "voice_corpus.txt")
    report = export_corpus(corpus_dir, out_path, min_words=3)
    assert report["entries"] == 3

    entries = parse_corpus(out_path)
    assert len(entries) == 3
    assert {e.tier for e in entries} == {1, 2, 3}
    dm_entry = next(e for e in entries if e.tier == 1)
    assert "café" in dm_entry.text
    assert dm_entry.channel.strip() == "chat"
    assert dm_entry.audience.strip() == "friend-family"


# --- Qodo round-1 regression tests --------------------------------------


def test_header_like_body_line_stays_one_entry(tmp_path):
    """Finding 1: a body line that looks like a corpus header must not
    split the entry when parsed back."""
    chunk = {
        "text": "Real content first line\n--- 2020-01-01 | chat | peer ---\nand the line after it",
        "provenance": {"timestamp": "2025-03-01T10:00:00"},
        "context": {"channel": "chat", "audience": "friend-family"},
    }
    from ingest.export import render_entry

    out = tmp_path / "corpus.txt"
    out.write_text(render_entry(chunk), encoding="utf-8")
    entries = parse_corpus(str(out))
    assert len(entries) == 1
    assert entries[0].year == 2025
    assert "line after it" in entries[0].text


def test_instagram_adapter_accepts_relative_base_path(tmp_path):
    """Finding 2: relative source paths must not crash relative_to()."""
    rel = os.path.relpath(os.path.join(FIXTURES, "instagram_export"), os.getcwd())
    config = make_config(tmp_path, instagram={"paths": [rel]})
    records = list(InstagramAdapter(config).iter_records())
    assert len(records) == 3
    assert not any(os.path.isabs(r.origin_file) for r in records)


def test_extra_metadata_kept_out_of_tone_signals(tmp_path):
    """Finding 4: string metadata lands in context.extra, tone_signals
    stays numeric-only."""
    from ingest.adapters.messages_txt import MessagesAdapter
    import json as json_mod

    config = make_config(
        tmp_path,
        messages={
            "combined_sent_paths": [os.path.join(FIXTURES, "combined_sent_sample.txt")]
        },
    )
    corpus_dir = config["corpus_dir"]
    manifest = Manifest(os.path.join(corpus_dir, "manifest.json"))
    run_source(MessagesAdapter(config), manifest, corpus_dir)

    with open(os.path.join(corpus_dir, "chunks", "messages.jsonl"), encoding="utf-8") as f:
        chunks = [json_mod.loads(line) for line in f]
    assert chunks
    for chunk in chunks:
        for value in chunk["context"]["tone_signals"].values():
            assert isinstance(value, (int, float))
        assert chunk["context"]["extra"].get("subject")


def test_split_paragraph_chunks_bounds_oversized_paragraph():
    """Finding 5: a single paragraph over the limit is sliced, never
    emitted oversized."""
    from ingest.adapters.documents import split_paragraph_chunks

    huge = " ".join(f"word{i}" for i in range(950))
    chunks = split_paragraph_chunks(huge, max_words=400)
    assert all(len(c.split()) <= 400 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 950


# --- doc_type subtypes (docs/doc-types.md) ------------------------------


def test_documents_adapter_derives_doc_type_from_first_level_folder(tmp_path):
    docs = tmp_path / "documents"
    (docs / "scripts").mkdir(parents=True)
    (docs / "scripts" / "2024-01-05-show-open.txt").write_text(
        "Tonight we look at three stories shaping the week ahead.",
        encoding="utf-8",
    )
    (docs / "cover-letters" / "nested").mkdir(parents=True)
    (docs / "cover-letters" / "nested" / "2025-02-01-letter.md").write_text(
        "I am writing to express interest in the producer role.",
        encoding="utf-8",
    )
    (docs / "unexpected-folder").mkdir()
    (docs / "unexpected-folder" / "note.txt").write_text(
        "Folder names outside the expected set pass through verbatim.",
        encoding="utf-8",
    )
    (docs / "2023-11-09-root-note.txt").write_text(
        "Files directly in the configured dir carry no doc_type.",
        encoding="utf-8",
    )

    from ingest.adapters.documents import DocumentsAdapter

    config = make_config(tmp_path, documents={"dirs": [str(docs)]})
    records = list(DocumentsAdapter(config).iter_records())
    by_file = {r.origin_file: r for r in records}
    assert by_file["scripts/2024-01-05-show-open.txt"].doc_type == "scripts"
    assert (
        by_file["cover-letters/nested/2025-02-01-letter.md"].doc_type
        == "cover-letters"
    ), "only the first-level folder names the doc_type"
    assert by_file["unexpected-folder/note.txt"].doc_type == "unexpected-folder"
    assert by_file["2023-11-09-root-note.txt"].doc_type == ""


def test_doc_type_lands_in_chunk_context(tmp_path):
    import json as json_mod

    from ingest.adapters.documents import DocumentsAdapter

    docs = tmp_path / "documents"
    (docs / "segment-intros").mkdir(parents=True)
    (docs / "segment-intros" / "2024-06-01-intro.txt").write_text(
        "Coming up after the break, a conversation about local news deserts.",
        encoding="utf-8",
    )
    config = make_config(tmp_path, documents={"dirs": [str(docs)]})
    corpus_dir = config["corpus_dir"]
    manifest = Manifest(os.path.join(corpus_dir, "manifest.json"))
    run_source(DocumentsAdapter(config), manifest, corpus_dir)

    with open(os.path.join(corpus_dir, "chunks", "documents.jsonl"), encoding="utf-8") as f:
        chunks = [json_mod.loads(line) for line in f]
    assert chunks
    assert all(c["context"]["doc_type"] == "segment-intros" for c in chunks)


def test_video_adapter_chunks_plain_txt_and_stamps_doc_type(tmp_path):
    from ingest.adapters.video import VideoAdapter

    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    long_transcript = " ".join(f"word{i}" for i in range(950))
    (tdir / "2023-05-01-episode.txt").write_text(long_transcript, encoding="utf-8")

    config = make_config(tmp_path, video={"transcript_paths": [str(tdir)]})
    records = list(VideoAdapter(config).iter_records())
    assert len(records) == 3, "950 words at max 400 must yield three chunks"
    assert all(r.source_type == "video_transcript" for r in records)
    assert all(r.doc_type == "on-camera" for r in records)
    assert all(len(r.text.split()) <= 400 for r in records)
    assert [r.extra["chunk_index"] for r in records] == [0, 1, 2]
    assert all(
        "words_per_minute" not in r.extra for r in records
    ), "plain txt has no timing info, so no WPM"


def test_video_adapter_srt_keeps_wpm_and_doc_type(tmp_path):
    from ingest.adapters.video import VideoAdapter

    config = make_config(
        tmp_path,
        video={"transcript_paths": [os.path.join(FIXTURES, "transcript_sample.srt")]},
    )
    records = list(VideoAdapter(config).iter_records())
    assert len(records) == 1
    assert records[0].doc_type == "on-camera"
    assert 20 < records[0].extra["words_per_minute"] < 26


def test_video_adapter_keeps_computed_zero_wpm(tmp_path):
    """Qodo round 1: a computed 0.0 WPM is real pacing data and must not
    be dropped by a truthiness check; only None (no timing) is dropped."""
    from ingest.adapters.video import VideoAdapter

    slow = tmp_path / "transcripts"
    slow.mkdir()
    (slow / "2024-03-01-slow.srt").write_text(
        "1\n00:00:00,000 --> 00:30:00,000\nhello\n",
        encoding="utf-8",
    )
    config = make_config(tmp_path, video={"transcript_paths": [str(slow)]})
    records = list(VideoAdapter(config).iter_records())
    assert len(records) == 1
    assert records[0].extra["words_per_minute"] == 0.0


def test_chunk_index_is_metadata_not_tone_signal(tmp_path):
    """Qodo round 1: numeric bookkeeping like chunk_index belongs in
    context.extra; tone_signals carries style metrics like WPM only."""
    import json as json_mod

    from ingest.adapters.video import VideoAdapter

    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "2024-04-01-clip.srt").write_text(
        "1\n00:00:00,000 --> 00:00:30,000\n"
        "A short clip transcript with a handful of spoken words in it.\n",
        encoding="utf-8",
    )
    config = make_config(tmp_path, video={"transcript_paths": [str(tdir)]})
    corpus_dir = config["corpus_dir"]
    manifest = Manifest(os.path.join(corpus_dir, "manifest.json"))
    run_source(VideoAdapter(config), manifest, corpus_dir)

    with open(os.path.join(corpus_dir, "chunks", "video.jsonl"), encoding="utf-8") as f:
        chunks = [json_mod.loads(line) for line in f]
    assert chunks
    for chunk in chunks:
        assert chunk["context"]["extra"]["chunk_index"] == 0
        assert "chunk_index" not in chunk["context"]["tone_signals"]
        assert chunk["context"]["tone_signals"]["words_per_minute"] > 0
        assert "words_per_minute" not in chunk["context"]["extra"]


def test_combined_sent_gets_message_source_type_not_email():
    with open(os.path.join(FIXTURES, "combined_sent_sample.txt"), encoding="utf-8") as f:
        raw = f.read()
    records = list(parse_combined_sent(raw, "sample.txt", "sample"))
    assert records
    assert all(r.source_type == "message_sent" for r in records)

    context = build_context("message_sent", "Let me see if I can swing Sunday.")
    assert context.channel == "text"
    assert context.medium == "sms"
    assert context.audience == "friend-family"


def test_old_chunk_without_doc_type_key_still_loads():
    from ingest.schema import Chunk

    old = {
        "id": "abcdef0123456789",
        "text": "an old chunk written before doc_type existed",
        "hash": "abcdef0123456789" + "0" * 48,
        "tier": 1,
        "provenance": {
            "source_type": "instagram_dm",
            "origin_file": "messages.json",
            "export_id": "old-export",
            "timestamp": "2024-05-01T10:00:00",
            "extractor": "ingest.adapters.instagram@1.0",
        },
        "context": {
            "channel": "chat",
            "audience": "friend-family",
            "medium": "dm",
            "relationship_hint": "old friend",
            "goal": "connect",
            "tone_signals": {},
            "extra": {},
            "inference": "heuristic-v1",
        },
        "schema_version": "1.0",
    }
    chunk = Chunk.from_dict(old)
    assert chunk.context.doc_type == ""
    assert chunk.to_dict()["context"]["doc_type"] == ""


def test_instagram_adapter_survives_symlinked_subdirectory(tmp_path):
    """Qodo round 2: a symlinked directory inside the export must not
    crash origin_file computation or lose records."""
    real = tmp_path / "elsewhere" / "content"
    real.mkdir(parents=True)
    (real / "posts_1.json").write_text(
        '[{"media": [{"title": "Caption living behind a symlink", '
        '"creation_timestamp": 1600000000}]}]',
        encoding="utf-8",
    )
    export = tmp_path / "export" / "your_instagram_activity"
    export.mkdir(parents=True)
    (export / "content").symlink_to(real, target_is_directory=True)

    config = make_config(tmp_path, instagram={"paths": [str(tmp_path / "export")]})
    records = list(InstagramAdapter(config).iter_records())
    assert len(records) == 1
    assert "symlink" in records[0].text
    assert records[0].origin_file
