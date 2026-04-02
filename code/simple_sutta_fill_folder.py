#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# --- transcript id detection -------------------------------------------------

WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

TENS_MAP = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

ONES_1_9 = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

PAT_ANY_ID3 = re.compile(r"(?<!\d)(\d{1,2}\.\d{1,3}\.\d{1,3})(?!\d)")
PAT_ANY_ID2 = re.compile(r"(?<!\d)(\d{1,2}\.\d{1,3})(?!\.\d)(?!\d)")
SPACED_TRIPLE_RE = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,3})\s+(\d{1,3})(?!\.\d)(?!\d)")


def normalize_spaced_an_triples(text: str) -> str:
    return SPACED_TRIPLE_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}.{m.group(3)}", text)


def _third_component_spoken_to_str(s: str) -> str | None:
    s = (s or "").strip().lower()
    if not s:
        return None
    if s.isdigit():
        return str(int(s))
    if s in WORD_TO_DIGIT:
        return WORD_TO_DIGIT[s]
    if s in TENS_MAP:
        return str(TENS_MAP[s])
    parts = s.split()
    if len(parts) == 2 and parts[0] in TENS_MAP and parts[1] in ONES_1_9:
        return str(TENS_MAP[parts[0]] + ONES_1_9[parts[1]])
    return None


def normalize_spoken_an_triples_loose(text: str) -> str:
    word_alt = "|".join(sorted(WORD_TO_DIGIT.keys(), key=len, reverse=True))
    tens = "|".join(TENS_MAP.keys())
    ones = "|".join(ONES_1_9.keys())
    extra = "|".join(sorted(set(TENS_MAP.keys()) | set(WORD_TO_DIGIT.keys()), key=len, reverse=True))
    third = rf"(?:\d{{1,3}}|(?:{tens})\s+(?:{ones})|(?:{extra}))"
    pat = re.compile(
        rf"\b({word_alt})\s+point\s+({word_alt})\s+point\s+({third})\b",
        re.IGNORECASE,
    )

    def repl(m: re.Match[str]) -> str:
        a = WORD_TO_DIGIT.get(m.group(1).lower())
        b = WORD_TO_DIGIT.get(m.group(2).lower())
        c = _third_component_spoken_to_str(m.group(3))
        if not a or not b or not c:
            return m.group(0)
        return f"{a}.{b}.{c}"

    return pat.sub(repl, text)


def normalize_for_id_search(text: str) -> str:
    text = text or ""
    text = normalize_spaced_an_triples(text)
    text = normalize_spoken_an_triples_loose(text)
    return text


def find_transcript_id(text: str, max_chars: int = 800) -> str:
    t = normalize_for_id_search((text or "")[:max_chars])
    m = PAT_ANY_ID3.search(t)
    if m:
        return m.group(1)
    m = PAT_ANY_ID2.search(t)
    if m:
        return m.group(1)
    return ""


# --- canon id + fetch --------------------------------------------------------

def flatten_to_canon_id(transcript_id: str) -> str:
    s = (transcript_id or "").strip().lower().replace(" ", "")
    if not s:
        return ""
    parts = [p for p in s.split(".") if p]
    if len(parts) >= 3:
        return f"an{parts[0]}.{parts[-1]}"
    if len(parts) == 2:
        return f"an{parts[0]}.{parts[1]}"
    return f"an{parts[0]}"


def bilara_url(uid: str, author: str = "sujato") -> str:
    return f"https://suttacentral.net/api/bilarasuttas/{uid}/{author}"


def extract_translation_text(data: dict[str, Any]) -> str:
    tt = data.get("translation_text")
    ko = data.get("keys_order")
    if not isinstance(tt, dict):
        return ""
    if isinstance(ko, list) and ko:
        return " ".join(str(tt.get(k, "")).strip() for k in ko if str(tt.get(k, "")).strip()).strip()
    return " ".join(str(v).strip() for v in tt.values() if str(v).strip()).strip()


def fetch_canon_sutta(canon_id: str, author: str = "sujato", pause_s: float = 0.0) -> str:
    if not canon_id:
        return ""
    if pause_s > 0:
        time.sleep(pause_s)
    req = urllib.request.Request(
        bilara_url(canon_id, author=author),
        headers={"User-Agent": "simple-sutta-fill-folder/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    return extract_translation_text(data)


# --- folder processing -------------------------------------------------------

def process_json_file(path: Path, out_dir: Path, author: str, pause_s: float) -> tuple[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {path}")

    cache: dict[str, str] = {}
    rows = 0
    filled = 0

    for row in raw:
        if not isinstance(row, dict):
            continue
        rows += 1
        transcript_content = str(row.get("transcript_content", "") or "")
        transcript_id = find_transcript_id(transcript_content)
        canon_id = flatten_to_canon_id(transcript_id)

        row["transcript_id"] = transcript_id
        row["canon_id"] = canon_id

        if canon_id:
            if canon_id not in cache:
                cache[canon_id] = fetch_canon_sutta(canon_id, author=author, pause_s=pause_s)
            row["canon_sutta"] = cache[canon_id]
            if cache[canon_id]:
                filled += 1
        else:
            row["canon_sutta"] = ""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    out_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, filled


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process every JSON file in a folder: transcript_id -> canon_id -> canon_sutta"
    )
    parser.add_argument("input_folder", help="Folder containing JSON files")
    parser.add_argument("-o", "--output-folder", default="filled_json", help="Output folder")
    parser.add_argument("--author", default="sujato", help="Bilara author, default: sujato")
    parser.add_argument("--pause", type=float, default=0.1, help="Pause between uncached HTTP requests")
    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)

    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    files = sorted(input_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in: {input_dir}", file=sys.stderr)
        return 1

    total_rows = 0
    total_filled = 0

    for path in files:
        rows, filled = process_json_file(path, out_dir, author=args.author, pause_s=args.pause)
        total_rows += rows
        total_filled += filled
        print(f"Processed {path.name}: {rows} row(s), {filled} canon_sutta filled")

    print(f"Done. Files: {len(files)}, rows: {total_rows}, canon_sutta filled: {total_filled}")
    print(f"Output folder: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
