from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on line {ln}: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"Expected JSON object on line {ln}, got {type(obj).__name__}")
            rows.append(obj)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert JSONL to pretty JSON array for easier reading.")
    ap.add_argument("input_jsonl", type=Path)
    ap.add_argument(
        "-o",
        "--output-json",
        type=Path,
        default=None,
        help="Output .json path (default: <input>.pretty.json)",
    )
    args = ap.parse_args()

    inp = args.input_jsonl.resolve()
    if not inp.is_file():
        raise SystemExit(f"Not a file: {inp}")

    out = args.output_json
    if out is None:
        out = inp.with_suffix("")  # drop .jsonl
        out = out.with_name(out.name + ".pretty").with_suffix(".json")
    out = out.resolve()

    rows = load_jsonl(inp)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} records -> {out}")


if __name__ == "__main__":
    main()

