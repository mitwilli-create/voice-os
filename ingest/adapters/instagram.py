"""Instagram export adapter.

Port of ingest/legacy/extract_instagram_voice_corpus.py: post/story/reel
captions, comments, and self-authored DMs, with the mojibake fix and the
share/like noise filters. Multiple export drops may be configured; overlap
between them is collapsed downstream by content-hash dedup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..normalize import decode_meta_text, is_self, ms_to_iso, s_to_iso
from .base import RawRecord, SourceAdapter


def _load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _caption_and_ts(item: dict) -> tuple[str, object]:
    """Captions hide in media[].title, title, or caption depending on the
    export vintage. Album posts carry empty per-media titles with the
    real caption on the item, so only a non-empty media title wins."""
    for media_item in item.get("media", []):
        if media_item.get("title"):
            return (
                decode_meta_text(media_item["title"]),
                media_item.get("creation_timestamp", item.get("creation_timestamp")),
            )
    if item.get("title"):
        return (
            decode_meta_text(item["title"]),
            item.get("creation_timestamp", item.get("taken_at")),
        )
    if item.get("caption"):
        return (
            decode_meta_text(item["caption"]),
            item.get("creation_timestamp", item.get("taken_at")),
        )
    return "", None


class InstagramAdapter(SourceAdapter):
    name = "instagram"

    def iter_records(self) -> Iterator[RawRecord]:
        for base in self.configured_paths():
            # Resolve once so relative_to() in the extractors always shares
            # a path basis with the resolved files from _content_files().
            base_path = Path(base).expanduser().resolve()
            if not base_path.exists():
                continue
            export_id = base_path.name
            yield from self._captions(base_path, export_id, "posts*.json", "ig_post", "ig_posts")
            yield from self._captions(base_path, export_id, "stories.json", "ig_story", "ig_stories")
            yield from self._captions(base_path, export_id, "reels.json", "ig_reel", "ig_reels_media")
            yield from self._comments(base_path, export_id)
            yield from self._messages(base_path, export_id)

    def _content_files(self, base_path: Path, pattern: str) -> list[Path]:
        # Dedup on the as-discovered path, never on .resolve(): resolving a
        # file inside a symlinked subdirectory escapes base_path and breaks
        # relative_to(). Files reachable twice through symlinks are
        # collapsed downstream by content-hash dedup anyway.
        seen: dict[Path, None] = {}
        for candidate in (
            base_path / "content",
            base_path / "your_instagram_activity" / "content",
            base_path / "your_instagram_activity" / "media",
            base_path / "your_instagram_activity" / "posts",
            base_path / "media" / "posts",
        ):
            if candidate.exists():
                for f in sorted(candidate.glob(pattern)):
                    seen[f] = None
        # 2026 exports moved content JSON under your_instagram_activity/media;
        # accept any of the content-carrying directory names, which still
        # excludes lookalikes such as logged_information insights. Only
        # directories INSIDE the export count: filtering on absolute path
        # components would admit unrelated JSON whenever some parent folder
        # above the export happens to be named media, posts, or content.
        for f in sorted(base_path.rglob(pattern)):
            try:
                inner_dirs = f.relative_to(base_path).parts[:-1]
            except ValueError:
                inner_dirs = ()
            if {"content", "posts", "media"} & set(inner_dirs):
                seen[f] = None
        return list(seen)

    @staticmethod
    def _origin(json_file: Path, base_path: Path) -> str:
        try:
            return str(json_file.relative_to(base_path))
        except ValueError:
            return str(json_file)

    def _captions(
        self, base_path: Path, export_id: str, pattern: str, source_type: str, list_key: str
    ) -> Iterator[RawRecord]:
        for json_file in self._content_files(base_path, pattern):
            data = _load_json(json_file)
            if not data:
                continue
            items = data if isinstance(data, list) else data.get(list_key, data.get("posts", []))
            for item in items:
                caption, ts = _caption_and_ts(item)
                if not caption:
                    continue
                yield RawRecord(
                    text=caption,
                    source_type=source_type,
                    origin_file=self._origin(json_file, base_path),
                    export_id=export_id,
                    timestamp=s_to_iso(ts),
                )

    def _comments(self, base_path: Path, export_id: str) -> Iterator[RawRecord]:
        seen: dict[Path, None] = {}
        for f in sorted(base_path.rglob("*comment*.json")):
            seen[f] = None
        for json_file in seen:
            data = _load_json(json_file)
            if not data:
                continue
            if isinstance(data, dict):
                comment_list = data.get(
                    "comments_media_comments",
                    data.get(
                        "comments_reels_comments",
                        data.get("post_comments", data.get("comments", [])),
                    ),
                )
            else:
                comment_list = data
            for comment in comment_list:
                content = ""
                ts = None
                for item in comment.get("string_list_data", []):
                    content = decode_meta_text(item.get("value", ""))
                    ts = item.get("timestamp")
                    break
                if not content:
                    # 2026 exports carry the comment in string_map_data
                    # keyed "Comment", with the epoch under "Time".
                    string_map = comment.get("string_map_data", {})
                    entry = string_map.get("Comment")
                    if isinstance(entry, dict):
                        content = decode_meta_text(entry.get("value", ""))
                        time_entry = string_map.get("Time")
                        if isinstance(time_entry, dict):
                            ts = time_entry.get("timestamp")
                if not content and "comment" in comment:
                    content = decode_meta_text(comment["comment"])
                    ts = comment.get("timestamp")
                if not content and "text" in comment:
                    content = decode_meta_text(comment["text"])
                    ts = comment.get("timestamp", comment.get("created_at"))
                if not content:
                    continue
                yield RawRecord(
                    text=content,
                    source_type="ig_comment",
                    origin_file=self._origin(json_file, base_path),
                    export_id=export_id,
                    timestamp=s_to_iso(ts),
                )

    def _messages(self, base_path: Path, export_id: str) -> Iterator[RawRecord]:
        for inbox in (
            base_path / "messages" / "inbox",
            base_path / "your_instagram_activity" / "messages" / "inbox",
        ):
            if not inbox.exists():
                continue
            for conv_folder in sorted(inbox.iterdir()):
                if not conv_folder.is_dir():
                    continue
                for json_file in sorted(conv_folder.glob("message_*.json")):
                    data = _load_json(json_file)
                    if not data or "messages" not in data:
                        continue
                    for msg in data["messages"]:
                        sender = msg.get("sender_name", "")
                        if not is_self(sender, self.identity):
                            continue
                        content = decode_meta_text(msg.get("content", ""))
                        if not content:
                            continue
                        lowered = content.lower()
                        if "shared a" in lowered and any(
                            kind in lowered for kind in ("story", "post", "reel")
                        ):
                            continue
                        if content.startswith("Liked a message"):
                            continue
                        yield RawRecord(
                            text=content,
                            source_type="instagram_dm",
                            origin_file=self._origin(json_file, base_path),
                            export_id=export_id,
                            timestamp=ms_to_iso(msg.get("timestamp_ms")),
                            relationship_hint=conv_folder.name,
                        )
