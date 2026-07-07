"""Voice OS ingestion layer.

Consolidates the legacy Google Drive extractors (see ingest/legacy/) into a
single incremental pipeline: source adapters emit raw records, the core
normalizes and deduplicates them, and every surviving chunk carries full
provenance back to its original export file.

Output contract (consumed by score.py / pipeline.py via ingest.export):
    corpus/chunks/<source>.jsonl   provenance-rich chunk store (one JSON per line)
    corpus/manifest.json           content-hash index + run ledger
    corpus/voice_corpus.txt        rendered corpus in the voice_os.corpus format
"""

from .schema import SCHEMA_VERSION, Chunk, Context, Provenance

__all__ = ["Chunk", "Context", "Provenance", "SCHEMA_VERSION"]
