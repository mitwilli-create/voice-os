"""Written documents adapter: scripts, essays, notes, and comments kept as
local files.

Walks configured directories for .txt/.md/.docx, splits long documents on
paragraph boundaries so no chunk exceeds max_chunk_words, and dates each
chunk from the file's modification time (weakest date signal in the
system; override by embedding YYYY-MM-DD in the filename).
"""

from __future__ import annotations

import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .base import RawRecord, SourceAdapter

_FILENAME_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_DOCX_PARA = re.compile(r"</w:p>")
_XML_TAG = re.compile(r"<[^>]+>")
DEFAULT_EXTENSIONS = (".txt", ".md", ".docx")


def read_docx_text(path: str) -> str:
    """Minimal .docx text extraction, no dependencies: paragraphs from
    word/document.xml."""
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    except (OSError, KeyError, zipfile.BadZipFile):
        return ""
    xml = _DOCX_PARA.sub("\n", xml)
    return _XML_TAG.sub("", xml)


def split_paragraph_chunks(text: str, max_words: int) -> list[str]:
    chunks: list[str] = []
    bucket: list[str] = []
    bucket_words = 0
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        n = len(para.split())
        if bucket and bucket_words + n > max_words:
            chunks.append("\n\n".join(bucket))
            bucket, bucket_words = [], 0
        bucket.append(para)
        bucket_words += n
    if bucket:
        chunks.append("\n\n".join(bucket))
    return chunks


def file_date(path: str) -> str | None:
    match = _FILENAME_DATE.search(os.path.basename(path))
    if match:
        return match.group(1) + "T00:00:00"
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    except OSError:
        return None


class DocumentsAdapter(SourceAdapter):
    name = "documents"

    def configured_paths(self) -> list[str]:
        return list(self.options.get("dirs", []))

    def iter_records(self) -> Iterator[RawRecord]:
        extensions = tuple(self.options.get("extensions", DEFAULT_EXTENSIONS))
        max_words = int(self.options.get("max_chunk_words", 400))
        for root in self.configured_paths():
            root_path = Path(root)
            if not root_path.exists():
                continue
            export_id = root_path.name
            for path in sorted(root_path.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in extensions:
                    continue
                if path.suffix.lower() == ".docx":
                    text = read_docx_text(str(path))
                else:
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                timestamp = file_date(str(path))
                for i, chunk_text in enumerate(
                    split_paragraph_chunks(text, max_words)
                ):
                    yield RawRecord(
                        text=chunk_text,
                        source_type="document",
                        origin_file=str(path.relative_to(root_path)),
                        export_id=export_id,
                        timestamp=timestamp,
                        extra={"chunk_index": i},
                    )
