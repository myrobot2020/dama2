from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


BOOK_RE = re.compile(r"Book\s+(\d+)", re.IGNORECASE)


@dataclass
class SourceFile:
    path: Path
    book: int


def book_from_filename(name: str) -> int:
    m = BOOK_RE.search(name or "")
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "raw"
    out_dir = root / "raw2"

    if not raw_dir.is_dir():
        raise SystemExit(f"Missing raw dir: {raw_dir}")

    files: List[SourceFile] = []
    for p in sorted(raw_dir.glob("*.txt")):
        if not p.is_file():
            continue
        bk = book_from_filename(p.name)
        if 1 <= bk <= 11:
            files.append(SourceFile(path=p, book=bk))

    by_book: Dict[int, List[Path]] = {b: [] for b in range(1, 12)}
    for sf in files:
        by_book[sf.book].append(sf.path)

    out_dir.mkdir(parents=True, exist_ok=True)

    for b in range(1, 12):
        paths = sorted(by_book[b], key=lambda p: p.name)
        out_path = out_dir / f"AN_Book_{b:02d}.txt"
        parts: List[str] = []
        parts.append(f"===== MERGED BOOK {b} =====\n")
        for p in paths:
            parts.append(f"\n\n===== SOURCE: {p.name} =====\n")
            parts.append(p.read_text(encoding="utf-8", errors="replace").rstrip() + "\n")
        out_path.write_text("".join(parts), encoding="utf-8")
        print(f"Book {b:02d}: {len(paths)} source file(s) -> {out_path.name}")

    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()

