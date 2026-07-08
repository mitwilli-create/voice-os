"""Ingestion CLI.

    python -m ingest run [--source NAME ...] [--full] [--config PATH]
    python -m ingest export [--min-words N] [--out PATH]
    python -m ingest status

run is incremental by default: content whose hash is already in the
manifest is skipped, so pointing it at old exports plus a new drop appends
only new chunks. --full rebuilds the selected sources from scratch without
touching the others. Every run appends a ledger entry to the manifest and
writes a report under corpus/runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from .adapters import ADAPTERS
from .config import load_config
from .dedupe import Manifest
from .enrich import build_context
from .export import export_corpus
from .normalize import clean_text, normalize_for_hash
from .schema import Chunk, Provenance, content_hash
from .tiering import tier_for_chunk_year


def _chunks_path(corpus_dir: str, source: str) -> str:
    return os.path.join(corpus_dir, "chunks", f"{source}.jsonl")


def run_source(adapter, manifest: Manifest, corpus_dir: str) -> dict:
    stats = {"new": 0, "duplicate": 0, "empty": 0, "words": 0}
    out_path = _chunks_path(corpus_dir, adapter.name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out:
        for record in adapter.iter_records():
            text = clean_text(record.text)
            if not text:
                stats["empty"] += 1
                continue
            digest = content_hash(normalize_for_hash(text))
            if manifest.seen(digest):
                stats["duplicate"] += 1
                continue
            year = int(record.timestamp[:4]) if record.timestamp else None
            context = build_context(
                record.source_type, text, record.relationship_hint, record.doc_type
            )
            for key, value in record.extra.items():
                if value is None:
                    continue
                # tone_signals stays numeric-only; everything else is
                # adapter metadata and belongs in context.extra.
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    context.tone_signals[key] = value
                else:
                    context.extra[key] = value
            chunk = Chunk(
                id=digest[:16],
                text=text,
                hash=digest,
                tier=tier_for_chunk_year(year),
                provenance=Provenance(
                    source_type=record.source_type,
                    origin_file=record.origin_file,
                    export_id=record.export_id,
                    timestamp=record.timestamp,
                    extractor=adapter.extractor_id,
                ),
                context=context,
            )
            out.write(chunk.to_json() + "\n")
            manifest.add(digest, chunk.id, adapter.name)
            stats["new"] += 1
            stats["words"] += len(text.split())
    return stats


def cmd_run(args) -> int:
    config = load_config(args.config)
    corpus_dir = config["corpus_dir"]
    manifest = Manifest(os.path.join(corpus_dir, "manifest.json"))
    selected = args.source or sorted(ADAPTERS)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {
        "run_id": run_id,
        "mode": "full" if args.full else "incremental",
        "config": os.path.basename(config["_config_path"]),
        "sources": {},
        "warnings": [],
    }

    for name in selected:
        if name not in ADAPTERS:
            print(f"unknown source: {name} (known: {', '.join(sorted(ADAPTERS))})")
            return 2
        adapter = ADAPTERS[name](config)
        if not adapter.available():
            report["sources"][name] = {"skipped": "no configured path exists"}
            print(f"[{name}] skipped: no configured path exists")
            continue
        if args.full:
            dropped = manifest.drop_source(name)
            chunk_file = _chunks_path(corpus_dir, name)
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            print(f"[{name}] full rebuild: dropped {dropped} hashes")
        print(f"[{name}] ingesting...")
        stats = run_source(adapter, manifest, corpus_dir)
        report["sources"][name] = stats
        report["warnings"].extend(f"[{name}] {w}" for w in adapter.warnings)
        print(
            f"[{name}] new: {stats['new']:,}  duplicate: {stats['duplicate']:,}  "
            f"empty: {stats['empty']:,}  new words: {stats['words']:,}"
        )
        for warning in adapter.warnings:
            print(f"[{name}] warning: {warning}", file=sys.stderr)

    manifest.record_run(report)
    manifest.save()
    runs_dir = os.path.join(corpus_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    report_path = os.path.join(runs_dir, f"{run_id}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nmanifest: {manifest.count():,} unique chunks total")
    print(f"report:   {report_path}")
    return 0


def cmd_export(args) -> int:
    config = load_config(args.config)
    corpus_dir = config["corpus_dir"]
    out_path = args.out or os.path.join(corpus_dir, "voice_corpus.txt")
    min_words = args.min_words
    if min_words is None:
        min_words = int(config.get("export", {}).get("min_words", 8))
    report = export_corpus(corpus_dir, out_path, min_words=min_words)
    print(
        f"wrote {report['entries']:,} entries to {report['out_path']} "
        f"(skipped {report['skipped_short']:,} short, "
        f"{report['skipped_undated']:,} undated)"
    )
    return 0


def cmd_status(args) -> int:
    config = load_config(args.config)
    corpus_dir = config["corpus_dir"]
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("no manifest yet; run: python -m ingest run")
        return 0
    manifest = Manifest(manifest_path)
    by_source: dict[str, int] = {}
    for meta in manifest.data["hashes"].values():
        by_source[meta["source"]] = by_source.get(meta["source"], 0) + 1
    print(f"unique chunks: {manifest.count():,}")
    for source, count in sorted(by_source.items()):
        print(f"  {source:<12} {count:,}")
    runs = manifest.data["runs"]
    if runs:
        last = runs[-1]
        print(f"runs: {len(runs)} (last: {last['run_id']}, {last['mode']})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingest", description=__doc__)
    parser.add_argument("--config", default=None, help="path to ingest config JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="ingest configured sources")
    p_run.add_argument("--source", action="append", help="limit to one source (repeatable)")
    p_run.add_argument("--full", action="store_true", help="rebuild selected sources from scratch")
    p_run.set_defaults(func=cmd_run)

    p_export = sub.add_parser("export", help="render the corpus for score.py/pipeline.py")
    p_export.add_argument("--out", default=None)
    p_export.add_argument("--min-words", type=int, default=None)
    p_export.set_defaults(func=cmd_export)

    p_status = sub.add_parser("status", help="show manifest summary")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)
