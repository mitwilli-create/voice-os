"""Facebook/Messenger export adapter.

Port of ingest/legacy/extract_meta_voice_corpus_v2.py: walks the 2026 Meta
export layout under your_facebook_activity/, keeps only self-authored
content, fixes mojibake, and skips reactions/attachments/call markers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from ..normalize import decode_meta_text, is_self, ms_to_iso, s_to_iso
from .base import RawRecord, SourceAdapter

_SKIP_MARKERS = ("sent an attachment", "started a call")


def _load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class FacebookAdapter(SourceAdapter):
    name = "facebook"

    def iter_records(self) -> Iterator[RawRecord]:
        for base in self.configured_paths():
            base_path = Path(base)
            if not base_path.exists():
                continue
            export_id = base_path.name
            yield from self._messages(base_path, export_id, "inbox", "messenger")
            yield from self._messages(
                base_path, export_id, "archived_threads", "messenger_archived"
            )
            yield from self._posts(base_path, export_id)
            yield from self._comments(base_path, export_id)
            yield from self._stories(base_path, export_id)

    def _messages(
        self, base_path: Path, export_id: str, folder: str, source_type: str
    ) -> Iterator[RawRecord]:
        root = base_path / "your_facebook_activity" / "messages" / folder
        if not root.exists():
            if folder == "inbox":
                self.warnings.append(f"messages inbox not found under {base_path}")
            return
        for conv_folder in sorted(root.iterdir()):
            if not conv_folder.is_dir():
                continue
            for json_file in sorted(conv_folder.glob("message_*.json")):
                data = _load_json(json_file)
                if not data or "messages" not in data:
                    continue
                for msg in data["messages"]:
                    sender = decode_meta_text(msg.get("sender_name", ""))
                    if not is_self(sender, self.identity):
                        continue
                    content = decode_meta_text(msg.get("content", ""))
                    if not content:
                        continue
                    lowered = content.lower()
                    if any(marker in lowered for marker in _SKIP_MARKERS):
                        continue
                    if content.startswith("You ") and "reaction" in lowered:
                        continue
                    yield RawRecord(
                        text=content,
                        source_type=source_type,
                        origin_file=str(json_file.relative_to(base_path)),
                        export_id=export_id,
                        timestamp=ms_to_iso(msg.get("timestamp_ms")),
                        relationship_hint=conv_folder.name,
                    )

    def _posts(self, base_path: Path, export_id: str) -> Iterator[RawRecord]:
        posts_dir = base_path / "your_facebook_activity" / "posts"
        if not posts_dir.exists():
            return
        for posts_file in sorted(posts_dir.glob("your_posts*.json")):
            data = _load_json(posts_file)
            if not data:
                continue
            post_list = data if isinstance(data, list) else data.get("posts", [])
            for post in post_list:
                content = ""
                for item in post.get("data", []):
                    if "post" in item:
                        content = decode_meta_text(item["post"])
                        break
                if not content and "post" in post:
                    content = decode_meta_text(post["post"])
                if not content:
                    content = decode_meta_text(post.get("title", ""))
                if not content:
                    continue
                yield RawRecord(
                    text=content,
                    source_type="fb_post",
                    origin_file=str(posts_file.relative_to(base_path)),
                    export_id=export_id,
                    timestamp=s_to_iso(post.get("timestamp")),
                )

    def _comments(self, base_path: Path, export_id: str) -> Iterator[RawRecord]:
        comments_dir = base_path / "your_facebook_activity" / "comments_and_reactions"
        if not comments_dir.exists():
            return
        for comments_file in sorted(comments_dir.glob("comments*.json")):
            data = _load_json(comments_file)
            if not data:
                continue
            if isinstance(data, dict):
                comment_list = data.get("comments_v2", data.get("comments", []))
            else:
                comment_list = data
            for comment in comment_list:
                content = ""
                for item in comment.get("data", []):
                    if "comment" in item:
                        inner = item["comment"]
                        if isinstance(inner, dict):
                            content = decode_meta_text(inner.get("comment", ""))
                        else:
                            content = decode_meta_text(inner)
                        break
                if not content and "comment" in comment:
                    inner = comment["comment"]
                    if isinstance(inner, str):
                        content = decode_meta_text(inner)
                if not content:
                    continue
                yield RawRecord(
                    text=content,
                    source_type="fb_comment",
                    origin_file=str(comments_file.relative_to(base_path)),
                    export_id=export_id,
                    timestamp=s_to_iso(comment.get("timestamp")),
                )

    def _stories(self, base_path: Path, export_id: str) -> Iterator[RawRecord]:
        stories_dir = base_path / "your_facebook_activity" / "stories"
        if not stories_dir.exists():
            return
        for json_file in sorted(stories_dir.glob("*.json")):
            data = _load_json(json_file)
            if not data:
                continue
            story_list = data if isinstance(data, list) else data.get("stories", [])
            for story in story_list:
                content = decode_meta_text(story.get("title", story.get("text", "")))
                if not content:
                    continue
                yield RawRecord(
                    text=content,
                    source_type="fb_story",
                    origin_file=str(json_file.relative_to(base_path)),
                    export_id=export_id,
                    timestamp=s_to_iso(story.get("timestamp")),
                )
