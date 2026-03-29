#!/usr/bin/env python3
"""
Run the transcript normalization pipeline on .txt lecture files.

Extensibility: add a Transform subclass under transcript_pipeline/transforms/, register it
with @register("name"), then append {"name": "...", "options": {...}} to
transcript_pipeline/pipeline.default.json (or pass --config).

Examples:
  python normalize_transcripts.py --list-transforms
  python normalize_transcripts.py --only-prefix 025_
  python normalize_transcripts.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import transcript_pipeline.transforms  # noqa: F401 — register built-ins
from transcript_pipeline.registry import TRANSFORM_CLASSES
from transcript_pipeline.runner import run_pipeline_on_file

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "transcript_pipeline" / "pipeline.default.json"
DEFAULT_OUT = BASE_DIR / "transcripts_std"
SKIP_NAMES = frozenset({"all_transcripts.txt"})


def _iter_inputs(input_dir: Path, only_prefix: str) -> list[Path]:
    files = sorted(
        p
        for p in input_dir.glob("*.txt")
        if p.is_file() and p.name.lower() not in SKIP_NAMES
    )
    if only_prefix:
        files = [p for p in files if p.name.startswith(only_prefix)]
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize transcripts via transcript_pipeline.")
    ap.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Pipeline JSON (ordered transforms + options)",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory containing source *.txt files",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (files named <stem>.std.txt)",
    )
    ap.add_argument(
        "--resource-root",
        type=Path,
        default=BASE_DIR,
        help="Project root for resolving map_path and other relative assets",
    )
    ap.add_argument(
        "--only-prefix",
        default="",
        help="Only process files whose basename starts with this (e.g. 025_)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print planned outputs only")
    ap.add_argument("--list-transforms", action="store_true", help="Print registered names")
    args = ap.parse_args()

    if args.list_transforms:
        print("Registered transforms:", ", ".join(sorted(TRANSFORM_CLASSES)))
        return

    cfg = args.config.resolve()
    if not cfg.is_file():
        raise SystemExit(f"Missing config: {cfg}")

    resource_root = args.resource_root.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    files = _iter_inputs(input_dir, args.only_prefix)
    if not files:
        raise SystemExit("No matching .txt files.")

    for src in files:
        dest = output_dir / f"{src.stem}.std.txt"
        if args.dry_run:
            print(f"{src.name} -> {dest}")
            continue
        run_pipeline_on_file(
            src,
            resource_root=resource_root,
            config_path=cfg,
            dest=dest,
        )
        print(dest)


if __name__ == "__main__":
    main()
