#!/usr/bin/env python
"""Bulk-ingest text documents into a running LightRAG server via its REST API.

Reads files from an input directory, optionally pre-segments each file into
entries on a blank-line boundary (so each entry becomes its own chunk rather
than being split blindly by token count), posts them to /documents/texts in
batches, then polls /documents/pipeline_status until indexing finishes.

Usage::

    export LIGHTRAG_API_KEY=...
    python scripts/ingest_documents.py --input-dir ./corpus --url https://lightrag-c3p7.onrender.com

    # Treat each blank-line-delimited block within a file as its own entry
    # (recommended for source material that already has natural entry
    # boundaries, e.g. numbered notebook passages):
    python scripts/ingest_documents.py --input-dir ./corpus --split-entries

    # Dry run: show what would be sent without calling the API
    python scripts/ingest_documents.py --input-dir ./corpus --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_CHUNK_TOKEN_SIZE = 400
DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE = 30
DEFAULT_BATCH_SIZE = 20
DEFAULT_POLL_INTERVAL_SECONDS = 5


def load_entries_from_jsonl(path: Path, include_review_flagged: bool) -> list[tuple[str, str]]:
    """Reads {"file_source", "text", "review", ...} entries produced by
    preprocess_corpus.py. review=true entries are skipped unless
    --include-review-flagged is passed, since they're best-effort guesses
    at possible footnote/editorial content, not confirmed source text.
    """
    entries: list[tuple[str, str]] = []
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("review") and not include_review_flagged:
                skipped += 1
                continue
            entries.append((record["file_source"], record["text"]))
    if skipped:
        print(f"Skipped {skipped} review-flagged entries from {path} (pass --include-review-flagged to include)")
    return entries


def load_entries(input_dir: Path, pattern: str, split_entries: bool) -> list[tuple[str, str]]:
    """Returns a list of (file_source, text) pairs."""
    entries: list[tuple[str, str]] = []
    for path in sorted(input_dir.rglob(pattern)):
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        rel = path.relative_to(input_dir).as_posix()
        if not split_entries:
            entries.append((rel, raw))
            continue
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        for i, block in enumerate(blocks):
            entries.append((f"{rel}#{i}", block))
    return entries


def post_batch(
    session: requests.Session,
    base_url: str,
    texts: list[str],
    file_sources: list[str],
    chunk_token_size: int,
    chunk_overlap_token_size: int,
    split_by_character: str | None,
) -> str:
    params: dict = {
        "chunk_token_size": chunk_token_size,
        "chunk_overlap_token_size": chunk_overlap_token_size,
    }
    if split_by_character:
        params["split_by_character"] = split_by_character
        params["split_by_character_only"] = True

    resp = session.post(
        f"{base_url}/documents/texts",
        json={
            "texts": texts,
            "file_sources": file_sources,
            "chunking": {"strategy": "fixed_token", "params": params},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["track_id"]


def wait_for_pipeline_idle(session: requests.Session, base_url: str, poll_interval: int) -> None:
    while True:
        resp = session.get(f"{base_url}/documents/pipeline_status", timeout=30)
        resp.raise_for_status()
        status = resp.json()
        if not status.get("busy"):
            return
        print(
            f"  indexing in progress: {status.get('cur_batch', '?')}/{status.get('batchs', '?')} "
            f"batches, latest: {status.get('latest_message', '')}"
        )
        time.sleep(poll_interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, help="Directory of source text files")
    parser.add_argument("--pattern", default="*.txt", help="Glob pattern for source files (default: *.txt)")
    parser.add_argument("--jsonl", type=Path, help="Pre-segmented JSONL from preprocess_corpus.py (use instead of --input-dir)")
    parser.add_argument("--include-review-flagged", action="store_true", help="With --jsonl, also ingest entries preprocess_corpus.py flagged review=true")
    parser.add_argument("--url", default=os.environ.get("LIGHTRAG_URL", "http://localhost:9621"), help="LightRAG base URL")
    parser.add_argument("--api-key", default=os.environ.get("LIGHTRAG_API_KEY"), help="LightRAG API key (default: $LIGHTRAG_API_KEY)")
    parser.add_argument("--split-entries", action="store_true", help="Split each file on blank lines into separate entries/chunks")
    parser.add_argument("--chunk-token-size", type=int, default=DEFAULT_CHUNK_TOKEN_SIZE)
    parser.add_argument("--chunk-overlap-token-size", type=int, default=DEFAULT_CHUNK_OVERLAP_TOKEN_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be sent without calling the API")
    args = parser.parse_args()

    if not args.dry_run and not args.api_key:
        parser.error("--api-key or $LIGHTRAG_API_KEY is required unless --dry-run")
    if not args.jsonl and not args.input_dir:
        parser.error("one of --input-dir or --jsonl is required")

    if args.jsonl:
        entries = load_entries_from_jsonl(args.jsonl, args.include_review_flagged)
        source_desc = str(args.jsonl)
        # Entries from preprocess_corpus.py are already coalesced to sensible
        # sizes and may contain internal blank lines from merging — splitting
        # on "\n\n" again here would re-fragment them.
        split_entries_effective = False
    else:
        entries = load_entries(args.input_dir, args.pattern, args.split_entries)
        source_desc = str(args.input_dir)
        split_entries_effective = args.split_entries

    if not entries:
        print(f"No entries found from {source_desc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(entries)} entries from {source_desc}")

    if args.dry_run:
        for file_source, text in entries[:5]:
            preview = text[:120].replace("\n", " ")
            print(f"  [{file_source}] {preview}{'...' if len(text) > 120 else ''}")
        if len(entries) > 5:
            print(f"  ... and {len(entries) - 5} more")
        print("Dry run — nothing sent.")
        return 0

    session = requests.Session()
    session.headers["X-API-Key"] = args.api_key

    split_by_character = "\n\n" if split_entries_effective else None

    track_ids = []
    for i in range(0, len(entries), args.batch_size):
        batch = entries[i : i + args.batch_size]
        texts = [t for _, t in batch]
        file_sources = [f for f, _ in batch]
        track_id = post_batch(
            session,
            args.url,
            texts,
            file_sources,
            args.chunk_token_size,
            args.chunk_overlap_token_size,
            split_by_character,
        )
        track_ids.append(track_id)
        print(f"Submitted batch {i // args.batch_size + 1} ({len(batch)} entries), track_id={track_id}")

    print("Waiting for indexing pipeline to finish...")
    wait_for_pipeline_idle(session, args.url, args.poll_interval)
    print(f"Done. Submitted {len(entries)} entries in {len(track_ids)} batches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
