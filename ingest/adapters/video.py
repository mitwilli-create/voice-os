"""Spoken-voice adapter: transcripts of videos Mitchell anchored or led.

Consumes transcript files (.srt, .vtt, .txt) sitting next to or configured
alongside the video files. For timed formats it also computes words per
minute, recorded in extra as a pacing signal for the tone layer.

Transcription itself is a hook, not a dependency: transcribe_video() tells
you the exact whisper command to produce the .srt this adapter consumes.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

from .base import RawRecord, SourceAdapter
from .documents import file_date

_SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
TRANSCRIPT_EXTENSIONS = (".srt", ".vtt", ".txt")


def transcribe_video(video_path: str) -> None:
    raise NotImplementedError(
        "Local transcription is a manual hook. Run, for example:\n"
        f"  whisper {video_path!r} --model medium --output_format srt\n"
        "then add the resulting .srt path (or its directory) to "
        "sources.video.transcript_paths in ingest.local.json."
    )


def parse_timed_transcript(raw: str) -> tuple[str, float | None]:
    """Extract cue text and words-per-minute from .srt/.vtt content."""
    lines = []
    first_start = None
    last_end = None
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.isdigit() or stripped.upper() == "WEBVTT":
            continue
        match = _SRT_TIME.search(stripped)
        if match:
            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
            start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
            if first_start is None:
                first_start = start
            last_end = end
            continue
        lines.append(stripped)
    text = " ".join(lines)
    wpm = None
    if first_start is not None and last_end and last_end > first_start:
        minutes = (last_end - first_start) / 60
        if minutes > 0:
            wpm = round(len(text.split()) / minutes, 1)
    return text, wpm


class VideoAdapter(SourceAdapter):
    name = "video"

    def configured_paths(self) -> list[str]:
        return list(self.options.get("transcript_paths", []))

    def _transcript_files(self) -> Iterator[Path]:
        for configured in self.configured_paths():
            path = Path(configured)
            if path.is_dir():
                for ext in TRANSCRIPT_EXTENSIONS:
                    yield from sorted(path.rglob(f"*{ext}"))
            elif path.is_file():
                yield path

    def iter_records(self) -> Iterator[RawRecord]:
        for path in self._transcript_files():
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if path.suffix.lower() in (".srt", ".vtt"):
                text, wpm = parse_timed_transcript(raw)
            else:
                text, wpm = raw, None
            if not text.strip():
                continue
            extra = {"words_per_minute": wpm} if wpm else {}
            yield RawRecord(
                text=text,
                source_type="video_transcript",
                origin_file=path.name,
                export_id=path.parent.name or "video",
                timestamp=file_date(str(path)),
                extra=extra,
            )
