#!/usr/bin/env python
"""Clean and re-segment historical-document text/markdown before it goes
anywhere near LightRAG's own chunker.

LightRAG's built-in chunking strategies split on token count or blank
lines; neither one knows that a transcription is full of editorial
scaffolding (figure captions, page headers, footnotes) that isn't part of
the source author's own words, or that a sentence got physically split by
a page break in the scan it was OCR'd from. This script handles that
layer first, so what reaches LightRAG is one JSONL entry per coherent
"thought" — no editorial noise, no sentences severed by a page break, no
oversized dumps that blur several unrelated notes into one embedding.

Pipeline per file:
  1. If the source has form-feed (\\f) page breaks (typical of text
     extracted from a paginated scan), strip the running header/footer
     line that follows each one and rejoin the sentence across the break.
  2. Split into blank-line-delimited blocks.
  3. Classify each block: structural noise (dropped), section header
     (used only as a coalescing boundary, not emitted on its own),
     possible footnote/editorial commentary (flagged for review, kept
     separate from clean output), or content.
  4. Strip a trailing source-citation code from content blocks (e.g.
     "C.A. 154 1. c") and carry it as provenance instead of leaving it
     glued onto the text.
  5. Coalesce consecutive content blocks within the same section up to a
     target token count, so retrieval doesn't return isolated one-line
     fragments, but never merge across a header/section boundary.

Output is JSONL: {"file_source": ..., "text": ..., "token_count": ...,
"review": bool, "review_reason": str|null}. Feed it to
ingest_documents.py --jsonl.

The footnote/editorial heuristic is best-effort (keyword match, extended
forward until a block ends in a recognized citation code) — always
spot-check the review=true entries before deciding to include or drop
them; don't trust it blindly.

Usage::

    python scripts/preprocess_corpus.py --input "path/to/file_or_dir" --pattern "*.md" --out corpus.jsonl
    python scripts/preprocess_corpus.py --input "path/to/file_or_dir" --out corpus.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")

DEFAULT_TARGET_TOKENS = 200
DEFAULT_CEILING_TOKENS = 500

# --- page-break handling (form-feed-delimited text extracted from a scan) ---

PAGE_RUNNING_HEADER_RE = re.compile(
    r"^\s*(?:[0-9]{1,4}\s+[A-Z][A-Za-z ]{2,60}|[A-Z][A-Za-z ]{2,60}\s+[0-9]{1,4})\s*$"
)


def dejoin_page_breaks(text: str) -> str:
    """Strip running header/footer lines at form-feed page breaks and
    rejoin the surrounding text so a sentence split across a page isn't
    left in two pieces. This deliberately collapses whitespace at the
    seam to a single space, which can occasionally fuse two paragraphs
    that happened to break exactly at a page boundary — rare, and far
    less harmful than leaving a sentence severed mid-word.
    """
    if "\f" not in text:
        return text
    pages = text.split("\f")
    result = pages[0]
    for page in pages[1:]:
        # Strip a trailing bare page-footer number from the page that's
        # ending (e.g. "...free.\n\n61\n" before the form feed) — otherwise
        # it leaks into the start of the next page's first sentence.
        result = re.sub(r"\n[0-9]{1,4}\n?\Z", "\n", result)
        stripped = page.lstrip("\n")
        first_line, _, rest = stripped.partition("\n")
        if PAGE_RUNNING_HEADER_RE.match(first_line.strip()):
            page = rest.lstrip("\n")
        result = result.rstrip() + " " + page.lstrip()
    return result


# --- structural noise: editorial scaffolding that isn't the author's words ---

FIGURE_CAPTION_RE = re.compile(
    r"^(?:"
    r"(?:one|two|three|four|five|six)?\s*figures?\s+with\s+[a-z ]+[.:]?$"
    r"|on the (left|right) of the figures?:?$"
    r"|(alongside|beside|below|above) the figures?:?$"
    r"|at the side of the figures?:?$"
    r"|to the (side|left|right) of the figures?:?$"
    r"|next to the figures?:?$"
    r"|in the (upper|lower) half of the sheet:?$"
    r"|in the (left|right) margin(?: of the sheet)?:?$"
    r"|(?:below|above)\, crossed out:?$"
    r"|s\. l\.$"
    r")",
    re.IGNORECASE,
)

BARE_PAGE_NUMBER_RE = re.compile(r"^[0-9]{1,4}$")


def is_noise_block(block: str) -> bool:
    if "\n" in block:
        return False
    line = block.strip()
    if BARE_PAGE_NUMBER_RE.match(line):
        return True
    return bool(FIGURE_CAPTION_RE.match(line))


# --- section headers: used only as coalescing boundaries ---

HEADER_PREFIX_RE = re.compile(r"^(Of|On)\s+[a-z]")


def is_header_block(block: str) -> bool:
    if "\n" in block:
        return False
    line = block.strip()
    if not line or len(line) > 90:
        return False
    if line.isupper() and 1 <= len(line.split()) <= 8:
        return True
    if HEADER_PREFIX_RE.match(line) and line.endswith((".", ":")):
        return count_tokens(line) < 25
    return False


# --- trailing source-citation codes (kept as provenance, not embedded text) ---

CITATION_RE = re.compile(
    r"\s+((?:C\.?A\.?|Tr|B\.?M\.?|Forster|Fogli|Quaderni|Sul Volo|Windsor|Leic|Ms|MS)"
    r"\.?\s*[IVXLCDM0-9]+\s*[a-z]{0,3}\.?\s*\.?\s*[rRvV]?\.?\s*[a-zA-Z0-9]{0,2}\.?)\s*$"
)


def strip_citation(text: str) -> tuple[str, str | None]:
    match = CITATION_RE.search(text)
    if not match:
        return text, None
    return text[: match.start()].rstrip(), match.group(1).strip()


# --- possible footnote / editorial commentary (flagged for human review) ---

FOOTNOTE_HINT_RE = re.compile(
    r"\b(MS\.|Ibid|Cf\.|Loeb|Richter|Calvi|Vasari|the reading adopted by|"
    r"Archivio Storico|op\. cit\.)\b"
)

# Hard cap on how many blocks a single footnote-hint trigger can flag
# forward before it's assumed to be a false positive (e.g. a citation-code
# convention the source doesn't use, so the normal reset never fires).
FOOTNOTE_RUN_MAX_BLOCKS = 4


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def split_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def preprocess_file(path: Path, rel_source: str) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    raw = dejoin_page_breaks(raw)
    blocks = split_blocks(raw)

    entries: list[dict] = []
    group_texts: list[str] = []
    group_citations: list[str] = []
    group_tokens = 0
    group_index = 0
    footnote_run = False
    footnote_run_len = 0

    def flush_group():
        nonlocal group_texts, group_citations, group_tokens, group_index
        if not group_texts:
            return
        text = "\n\n".join(group_texts)
        file_source = f"{rel_source}#{group_index}"
        if group_citations:
            file_source += f" ({'; '.join(group_citations)})"
        entries.append(
            {
                "file_source": file_source,
                "text": text,
                "token_count": group_tokens,
                "review": False,
                "review_reason": None,
            }
        )
        group_index += 1
        group_texts = []
        group_citations = []
        group_tokens = 0

    for block in blocks:
        if is_noise_block(block):
            continue

        if is_header_block(block):
            flush_group()
            footnote_run = False
            footnote_run_len = 0
            continue

        cleaned, citation = strip_citation(block)

        is_hint = bool(FOOTNOTE_HINT_RE.search(block))
        if is_hint:
            footnote_run = True
            footnote_run_len = 0
        if footnote_run:
            entries.append(
                {
                    "file_source": f"{rel_source}#{group_index}",
                    "text": cleaned,
                    "token_count": count_tokens(cleaned),
                    "review": True,
                    "review_reason": "possible_footnote_or_editorial_commentary",
                }
            )
            group_index += 1
            footnote_run_len += 1
            # A block that ends with a recognized citation code is a strong
            # signal we're back in primary quoted text, so stop propagating.
            # Also give up after a few blocks regardless, in case this
            # source doesn't use a citation-code convention at all (the
            # propagation would otherwise never reset).
            if citation or footnote_run_len >= FOOTNOTE_RUN_MAX_BLOCKS:
                footnote_run = False
            continue

        block_tokens = count_tokens(cleaned)
        if group_tokens and group_tokens + block_tokens > DEFAULT_CEILING_TOKENS:
            flush_group()
        group_texts.append(cleaned)
        group_tokens += block_tokens
        if citation:
            group_citations.append(citation)
        if group_tokens >= DEFAULT_TARGET_TOKENS:
            flush_group()

    flush_group()
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Source file or directory")
    parser.add_argument("--pattern", default="*", help="Glob pattern when --input is a directory (default: *)")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="Print a summary instead of writing output")
    args = parser.parse_args()

    if args.input.is_file():
        files = [args.input]
        base = args.input.parent
    else:
        files = sorted(p for p in args.input.rglob(args.pattern) if p.is_file())
        base = args.input

    all_entries: list[dict] = []
    for path in files:
        rel_source = path.relative_to(base).as_posix() if path.is_relative_to(base) else path.name
        all_entries.extend(preprocess_file(path, rel_source))

    clean = [e for e in all_entries if not e["review"]]
    flagged = [e for e in all_entries if e["review"]]

    print(f"Processed {len(files)} file(s): {len(clean)} clean entries, {len(flagged)} flagged for review")
    if clean:
        avg_tokens = sum(e["token_count"] for e in clean) / len(clean)
        print(f"  clean entries: avg {avg_tokens:.0f} tokens, max {max(e['token_count'] for e in clean)}")
    for e in flagged[:10]:
        preview = e["text"][:100].replace("\n", " ")
        print(f"  [REVIEW] {e['file_source']}: {preview}{'...' if len(e['text']) > 100 else ''}")
    if len(flagged) > 10:
        print(f"  ... and {len(flagged) - 10} more flagged entries")

    if args.dry_run:
        print("Dry run — nothing written.")
        return 0

    with args.out.open("w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_entries)} entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
