"""mine: batch mining of voice model artifacts from the local chunk store.

Lives outside voice_os/ for the same reason ingest/ does: it is offline
batch tooling over personal JSONL data. The voice_os runtime never mines;
it only loads the validated JSON artifacts this package writes to the
gitignored corpus/mined/ directory (see docs/extended-model.md).
"""
