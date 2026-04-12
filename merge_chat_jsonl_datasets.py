"""
Merge multiple chat JSONL datasets (same record shape) into one JSONL file.

Assumes each line is a JSON object with at least:
  - source: str
  - messages: [{role, content}, ...]
Optionally:
  - conversation_id, title, created_at, path, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _has_messages(obj: Dict[str, Any]) -> bool:
    msgs = obj.get("messages")
    return isinstance(msgs, list) and len(msgs) > 0


def merge(inputs: List[Path], out_path: Path) -> dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as out:
        for inp in inputs:
            for obj in _iter_jsonl(inp):
                if not _has_messages(obj):
                    skipped += 1
                    continue
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

    return {"inputs": len(inputs), "written": written, "skipped": skipped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        required=True,
        help="Input JSONL files",
    )
    ap.add_argument(
        "--out",
        default=str(Path("training-data") / "all_chats.chat.jsonl"),
        help="Output JSONL file",
    )
    args = ap.parse_args()

    stats = merge([Path(p) for p in args.inputs], Path(args.out))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

