#!/usr/bin/env python3
"""
Suggest transcript tokens/phrases that fuzzy-match damaglossary_book1.json (read-only).

Default: Anguttara Book 1 lectures only (001_, 002_, 003_).

  python suggest_glossary_fixes.py
  python suggest_glossary_fixes.py --min-score 90 --max-window 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

try:
    from rapidfuzz import fuzz, process
except ImportError:
    print("Install rapidfuzz: pip install rapidfuzz", file=sys.stderr)
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent
DEFAULT_GLOSSARY = ROOT / "damaglossary_book1.json"
BOOK1_GLOB = "00[123]_*.txt"
# Latin letters + digits + apostrophe + common Pāli diacritics (NFC)
_TOKEN_RE = re.compile(
    r"[0-9A-Za-z\u00C0-\u024F\u1E00-\u1EFF']+",
    re.UNICODE,
)


def load_glossary(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Glossary must be a JSON array of strings")
    out: List[str] = []
    for x in data:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def book1_files(base: Path) -> List[Path]:
    files = sorted(base.glob(BOOK1_GLOB))
    return [p for p in files if p.is_file() and p.suffix.lower() == ".txt"]


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Return (token, start, end)."""
    out: List[Tuple[str, int, int]] = []
    for m in _TOKEN_RE.finditer(text):
        out.append((m.group(0), m.start(), m.end()))
    return out


def exact_glossary_keys(glossary: List[str]) -> Set[str]:
    s: Set[str] = set()
    for g in glossary:
        s.add(g.casefold())
        for part in g.split():
            if len(part) >= 2:
                s.add(part.casefold())
    return s


def scan_file(
    path: Path,
    glossary: List[str],
    exact_cf: Set[str],
    min_len: int,
    min_score: float,
    max_window: int,
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    toks = tokenize(text)
    if not toks:
        print(f"{path.name}: (no tokens)")
        return

    reported: Set[Tuple[int, int, str]] = set()

    for i in range(len(toks)):
        for w in range(1, min(max_window, len(toks) - i) + 1):
            slice_t = toks[i : i + w]
            window = " ".join(t for t, _, _ in slice_t).strip()
            if len(window) < min_len:
                continue
            # Skip English possessives ("buddha's" is not a typo for lemma "buddha")
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
            key = (start, end, canon)
            if key in reported:
                continue
            reported.add(key)
            snippet = text[max(0, start - 40) : min(len(text), end + 40)].replace("\n", " ")
            print(f"{path.name}:{start}-{end} score={score:.1f}")
            print(f"  was: {window!r}")
            print(f"  canon: {canon!r}")
            print(f"  …{snippet}…")
            print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fuzzy-match Book 1 transcripts against damaglossary_book1.json")
    ap.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY, help="JSON array of canonical strings")
    ap.add_argument("--root", type=Path, default=ROOT, help="Project root (Book 1 txt files live here)")
    ap.add_argument("--min-len", type=int, default=4, help="Minimum window character length")
    ap.add_argument("--min-score", type=float, default=88.0, help="Minimum fuzz.ratio to report")
    ap.add_argument("--max-window", type=int, default=5, help="Max words per window (1=single tokens only)")
    ap.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Optional specific .txt files (default: 001–003 Book 1)",
    )
    args = ap.parse_args()

    if not args.glossary.is_file():
        print(f"Missing glossary: {args.glossary}", file=sys.stderr)
        return 1

    glossary = load_glossary(args.glossary)
    exact_cf = exact_glossary_keys(glossary)

    if args.files:
        paths = [p.resolve() for p in args.files]
    else:
        paths = book1_files(args.root)
    if not paths:
        print(f"No transcript files found (glob {BOOK1_GLOB!r} under {args.root})", file=sys.stderr)
        return 1

    print(f"Glossary: {args.glossary} ({len(glossary)} entries)")
    print(f"Files: {len(paths)}  min_score={args.min_score} max_window={args.max_window}\n")

    for p in paths:
        if not p.is_file():
            print(f"Skip missing: {p}", file=sys.stderr)
            continue
        scan_file(p, glossary, exact_cf, args.min_len, args.min_score, args.max_window)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
