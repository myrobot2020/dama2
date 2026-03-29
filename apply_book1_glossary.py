#!/usr/bin/env python3
"""
Apply damaglossary_book1.json to Book 1 transcripts using non-overlapping greedy fuzzy matches.

  python apply_book1_glossary.py --dry-run
  python apply_book1_glossary.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set, Tuple

try:
    from rapidfuzz import fuzz, process
except ImportError:
    print("Install rapidfuzz: pip install rapidfuzz", file=sys.stderr)
    raise SystemExit(1)

from suggest_glossary_fixes import (
    BOOK1_GLOB,
    DEFAULT_GLOSSARY,
    book1_files,
    exact_glossary_keys,
    load_glossary,
    tokenize,
)

ROOT = Path(__file__).resolve().parent


def collect_hits(
    text: str,
    glossary: List[str],
    exact_cf: Set[str],
    min_len: int,
    min_score: float,
    max_window: int,
) -> List[Tuple[int, int, str, float]]:
    """Return (start, end, canonical, score) possibly overlapping."""
    toks = tokenize(text)
    hits: List[Tuple[int, int, str, float]] = []
    for i in range(len(toks)):
        for w in range(1, min(max_window, len(toks) - i) + 1):
            slice_t = toks[i : i + w]
            window = " ".join(t for t, _, _ in slice_t).strip()
            if len(window) < min_len:
                continue
            if "'" in window:
                continue
            wcf = window.casefold()
            if wcf in exact_cf:
                continue
            hit = process.extractOne(window, glossary, scorer=fuzz.ratio)
            if hit is None:
                continue
            canon, score, _ = hit
            if score < min_score:
                continue
            if canon.casefold() == wcf:
                continue
            start = slice_t[0][1]
            end = slice_t[-1][2]
            hits.append((start, end, canon, float(score)))
    return hits


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def greedy_non_overlapping(
    hits: List[Tuple[int, int, str, float]],
) -> List[Tuple[int, int, str, float]]:
    """Prefer higher score, then earlier start."""
    sorted_hits = sorted(hits, key=lambda h: (-h[3], h[0]))
    chosen: List[Tuple[int, int, str, float]] = []
    for h in sorted_hits:
        span = (h[0], h[1])
        if any(_overlaps(span, (c[0], c[1])) for c in chosen):
            continue
        chosen.append(h)
    chosen.sort(key=lambda h: h[0])
    return chosen


def apply_to_file(
    path: Path,
    glossary: List[str],
    exact_cf: Set[str],
    min_len: int,
    min_score: float,
    max_window: int,
    dry_run: bool,
) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = collect_hits(text, glossary, exact_cf, min_len, min_score, max_window)
    chosen = greedy_non_overlapping(hits)
    if not chosen:
        print(f"{path.name}: no replacements")
        return 0
    print(f"{path.name}: {len(chosen)} replacement(s)")
    for start, end, canon, score in chosen:
        was = text[start:end]
        print(f"  {start}-{end} score={score:.1f} {was!r} -> {canon!r}")
    if dry_run:
        return len(chosen)
    new_text = text
    for start, end, canon, _ in reversed(chosen):
        new_text = new_text[:start] + canon + new_text[end:]
    path.write_text(new_text, encoding="utf-8", newline="\n")
    return len(chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply Book 1 glossary fuzzy fixes to transcripts")
    ap.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=88.0)
    ap.add_argument("--max-window", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="Print only; do not write files")
    ap.add_argument("--apply", action="store_true", help="Write files in place")
    ap.add_argument("files", nargs="*", type=Path, help="Default: 001–003 Book 1")
    args = ap.parse_args()
    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply", file=sys.stderr)
        return 1
    if not args.glossary.is_file():
        print(f"Missing glossary: {args.glossary}", file=sys.stderr)
        return 1
    glossary = load_glossary(args.glossary)
    exact_cf = exact_glossary_keys(glossary)
    paths = [p.resolve() for p in args.files] if args.files else book1_files(args.root)
    if not paths:
        print(f"No files (glob {BOOK1_GLOB!r})", file=sys.stderr)
        return 1
    total = 0
    for p in paths:
        if not p.is_file():
            print(f"Skip missing: {p}", file=sys.stderr)
            continue
        total += apply_to_file(
            p, glossary, exact_cf, args.min_len, args.min_score, args.max_window, args.dry_run
        )
    print(f"Total replacements: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
