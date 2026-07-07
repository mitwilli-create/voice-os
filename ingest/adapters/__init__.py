"""Source adapters. Each adapter walks one family of local exports and
yields RawRecord objects; the CLI turns those into deduplicated,
provenance-tagged chunks."""

from .base import RawRecord, SourceAdapter
from .documents import DocumentsAdapter
from .email_mbox import EmailMboxAdapter
from .instagram import InstagramAdapter
from .messages_txt import MessagesAdapter
from .meta_facebook import FacebookAdapter
from .video import VideoAdapter

ADAPTERS = {
    adapter.name: adapter
    for adapter in (
        InstagramAdapter,
        FacebookAdapter,
        EmailMboxAdapter,
        MessagesAdapter,
        DocumentsAdapter,
        VideoAdapter,
    )
}

__all__ = ["ADAPTERS", "RawRecord", "SourceAdapter"]
