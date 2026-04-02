"""
Sutta tagging for AN transcripts: insert ### REFERENCE: markers, split to JSON records,
optional transcript_pipeline Transform, and CLI to build flat sutta_split/*.json.

Each segment object: ``transcript_id``, ``canon_id`` (SuttaCentral-style ``an{book}.{sutta}``;
three-part talk ids ``x.y.z`` → drop middle: ``4.8.79`` → ``an4.79``), ``book`` (from the reference,
or from the source filename when the ref is missing or parses as 0), ``transcript_content``,
``canon_content`` (after ``--enrich-json``).

Spaced sutta numbers (``4.19 186`` → ``4.19.186``) are normalized before digit patterns run.
Lead segments with no ``### REFERENCE:`` still get ``transcript_id`` / ``canon_id`` when the first
lines mention a sutta as digits or spoken English (including ``three point four point thirty six``).

Optional enrichment (SuttaCentral bilara, default author ``sujato``):

``python code/sutta_tagging.py --enrich-json sutta_split/035_....json``

All JSON under ``sutta_split``:

``python code/sutta_tagging.py --enrich-all-json``

Each successful bilara response is saved under ``canonapi/<author>/<uid>.json``. On later runs,
that file is used first so you can re-enrich a sample without calling the API again. Use
``--canon-local-only`` to never hit the network (only ``canonapi/`` + existing JSON).

Drop rows with no fetched canon text (``canon_id`` set, ``canon_content`` empty):

``python code/sutta_tagging.py --drop-missing-canon-json``

Report those rows without modifying JSON:

``python code/sutta_tagging.py --report-missing-canon``

Re-create the dropped-id list from ``raw/*.txt`` + live enrichment (same rows as before the drop; slow):

``python code/sutta_tagging.py --report-missing-canon-from-raw``

Mapping quality (transcript↔canon), ``use``, chains when mapping ok (books 3–6), census, book 7 review TSV:

``python code/sutta_tagging.py --score-and-chains --census-txt metric/census.txt``

Rebuild all raw with preserved ``canon_content`` when IDs match, then enrich only missing:

``python code/sutta_tagging.py --rebuild-all-json --preserve-canon-on-rebuild``
``python code/sutta_tagging.py --enrich-all-json --only-missing-canon``

Run: ``python code/sutta_tagging.py`` from the dama3 project root.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import shutil
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

# -----------------------------------------------------------------------------
# Core: patterns + tagging + split records
# -----------------------------------------------------------------------------

DEFAULT_MARKERS: Dict[str, str] = {
    r"4\.4\.36": "AN 4.36 (Doṇa Sutta)",
    r"4\.4\.37": "AN 4.37 (Aparihāna Sutta)",
    r"4\.5\.41": "AN 4.41 (Samādhi-bhāvanā Sutta)",
    r"4\.5\.49": "AN 4.49 (Vipallāsa Sutta)",
}

PAT_BOOK1_ID3 = re.compile(r"(?<!\d)(1\.\d+\.\d+)(?!\d)")
PAT_BOOK1_ID2 = re.compile(r"(?<!\d)(1\.\d+)(?!\.\d)(?!\d)")
PAT_ANY_ID3 = re.compile(r"(?<!\d)(\d{1,2}\.\d{1,3}\.\d{1,3})(?!\d)")
PAT_ANY_ID2 = re.compile(r"(?<!\d)(\d{1,2}\.\d{1,3})(?!\.\d)(?!\d)")

FILENAME_BOOK_RE = re.compile(r"Book\s+(\d+)([A-Za-z])?", re.IGNORECASE)
FILENAME_BOOK1_SUBBOOK_RE = re.compile(r"Book\s+1[ABCD]\b", re.IGNORECASE)

_REF_SPLIT_RE = re.compile(
    r"###\s*REFERENCE:\s*(AN\s+[\d.]+(?:\s*\([^)]*\))?)\s*",
    re.IGNORECASE,
)
_BOOK_FROM_REF_RE = re.compile(r"AN\s+(\d+)", re.IGNORECASE)
_REF_NUMERIC_TAIL = re.compile(r"AN\s+([\d.]+)", re.IGNORECASE)

CANON_BEGIN = "### CANON_TEXT_BEGIN"
CANON_END = "### CANON_TEXT_END"

# ASR sutta refs only: ``<n> point <n> point <n>`` (e.g. two point three point ten → 2.3.10).
# Does not touch ordinary English like "four noble truths" (no "point" triple pattern).
_SPOKEN_WORD_TO_DIGIT: Dict[str, str] = {
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
_SPOKEN_TRIPLE_RE = re.compile(
    r"\b("
    + "|".join(_SPOKEN_WORD_TO_DIGIT)
    + r")\s+point\s+("
    + "|".join(_SPOKEN_WORD_TO_DIGIT)
    + r")\s+point\s+("
    + "|".join(_SPOKEN_WORD_TO_DIGIT)
    + r")\b",
    re.IGNORECASE,
)


def normalize_spoken_an_triples(text: str) -> str:
    """
    Turn *only* ASR sutta triples ``<word> point <word> point <word>`` (each word a
    number name) into ``a.b.c`` for tagging. General phrases such as "four noble
    truths" or "two conditions" are left unchanged.
    """

    def repl(m: re.Match[str]) -> str:
        a = _SPOKEN_WORD_TO_DIGIT.get(m.group(1).lower())
        b = _SPOKEN_WORD_TO_DIGIT.get(m.group(2).lower())
        c = _SPOKEN_WORD_TO_DIGIT.get(m.group(3).lower())
        if not a or not b or not c:
            return m.group(0)
        return f"{a}.{b}.{c}"

    return _SPOKEN_TRIPLE_RE.sub(repl, text)


# ASR sometimes splits the third number: ``4.19 186`` → ``4.19.186``. Do not merge when the
# third token is the start of a decimal (e.g. ``2.4 0.6`` → must stay, not ``2.4.0``).
_SPACED_TRIPLE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\.(\d{1,3})\s+(\d{1,3})(?!\.\d)(?!\d)"
)


def normalize_spaced_an_triples(text: str) -> str:
    """Join ``book.chapter sutta`` into ``book.chapter.sutta`` when the third part is an integer."""

    def repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"

    return _SPACED_TRIPLE_RE.sub(repl, text)


_TENS_MAP: Dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ONES_1_9: Dict[str, int] = {
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


def _third_component_spoken_to_str(s: str) -> str | None:
    """ASR third part: digits, sixteen, thirty six, fifty → string for sutta number."""
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        return str(int(s))
    sl = s.lower()
    parts = sl.split()
    if len(parts) == 1:
        v = _SPOKEN_WORD_TO_DIGIT.get(sl)
        if v is not None:
            return v
        if sl in _TENS_MAP:
            return str(_TENS_MAP[sl])
        return None
    if len(parts) == 2:
        a, b = parts[0], parts[1]
        if a in _TENS_MAP and b in _ONES_1_9:
            return str(_TENS_MAP[a] + _ONES_1_9[b])
    return None


def _spoken_word_alt_pattern() -> str:
    return "|".join(sorted(_SPOKEN_WORD_TO_DIGIT.keys(), key=len, reverse=True))


def _third_part_pattern_for_regex() -> str:
    tens = "|".join(_TENS_MAP.keys())
    ones = "|".join(_ONES_1_9.keys())
    extra_singles = "|".join(
        sorted(set(_TENS_MAP.keys()) | set(_SPOKEN_WORD_TO_DIGIT.keys()), key=len, reverse=True)
    )
    return rf"(?:\d{{1,3}}|(?:{tens})\s+(?:{ones})|(?:{extra_singles}))"


_SPOKEN_TRIPLE_LOOSE = re.compile(
    rf"\b({_spoken_word_alt_pattern()})\s+point\s+({_spoken_word_alt_pattern()})\s+point\s+({_third_part_pattern_for_regex()})\b",
    re.IGNORECASE,
)


def _loose_triple_repl(m: re.Match[str]) -> str:
    a = _SPOKEN_WORD_TO_DIGIT.get(m.group(1).lower())
    b = _SPOKEN_WORD_TO_DIGIT.get(m.group(2).lower())
    c = _third_component_spoken_to_str(m.group(3))
    if not a or not b or not c:
        return m.group(0)
    return f"{a}.{b}.{c}"


def normalize_spoken_an_triples_loose(text: str) -> str:
    """
    Like ``normalize_spoken_an_triples`` but allows a compound third part, e.g.
    ``three point four point thirty six`` → ``3.4.36`` (ASR splits ``36`` into two words).
    """

    return _SPOKEN_TRIPLE_LOOSE.sub(_loose_triple_repl, text)


def normalize_text_for_sutta_tagging(text: str) -> str:
    """Spoken + spaced + loose triple normalization (shared by tagging and pipeline Transform)."""
    t = normalize_spoken_an_triples(text or "")
    t = normalize_spaced_an_triples(t)
    return normalize_spoken_an_triples_loose(t)


def infer_transcript_id_from_lead(text: str, *, max_chars: int = 560) -> str:
    """
    Find an AN-style ``book.chapter.sutta`` id in the first lines of a lead segment:
    digit triples (after spacing fixes), single-word spoken triples, then loose spoken triples.
    Returns ``""`` when nothing is found (e.g. pure introduction with no sutta number).
    """
    prefix = (text or "")[:max_chars]
    if not prefix.strip():
        return ""
    t = normalize_spoken_an_triples(prefix)
    t = normalize_spaced_an_triples(t)
    t = normalize_spoken_an_triples_loose(t)
    m3 = PAT_ANY_ID3.search(t)
    if m3:
        return m3.group(1).strip()
    m2 = PAT_ANY_ID2.search(t)
    if m2:
        return m2.group(1).strip()
    return ""


def _is_pure_next_sutta_filler(body: str) -> bool:
    """True for a standalone transition like ``the next sutta is`` (no sutta id in this segment)."""
    b = (body or "").strip().lower()
    return bool(re.match(r"^(the )?next (sutta|sura|sultana) is\.?$", b))


def transcript_content_supports_transcript_id(tid: str, content: str) -> bool:
    """
    True when this row's text actually mentions the sutta id (digits or spoken inference),
    not e.g. only \"the next sutta is\" while the id appeared only in the following segment.
    """
    tid = (tid or "").strip()
    if not tid or not (content or "").strip():
        return False
    if infer_transcript_id_from_lead(content) == tid:
        return True
    c = (content or "")[:2000]
    if tid in c:
        return True
    c_compact = re.sub(r"\s+", "", c)
    if tid in c_compact:
        return True
    return False


def fill_inferred_ids_for_empty_rows(records: List[Dict[str, Any]]) -> None:
    """Set ``transcript_id``, ``canon_id``, and ``book`` from lead text when ids are still empty."""
    for i, r in enumerate(records):
        if (r.get("transcript_id") or "").strip():
            continue
        body = (r.get("transcript_content") or "").strip()
        tid = infer_transcript_id_from_lead(body)
        if not tid and i + 1 < len(records):
            nxt = (records[i + 1].get("transcript_content") or "").strip()
            if nxt and not _is_pure_next_sutta_filler(body):
                tid_peek = infer_transcript_id_from_lead(f"{body} {nxt}")
                if tid_peek:
                    tid = tid_peek
        if not tid:
            continue
        r["transcript_id"] = tid
        r["canon_id"] = canon_id_from_transcript_id(tid)
        head = tid.split(".", 1)[0].strip()
        if head.isdigit():
            ib = int(head)
            if ib > 0:
                r["book"] = ib


class SuttaSegmentRecord(TypedDict, total=False):
    transcript_id: str
    canon_id: str
    book: int
    transcript_content: str
    canon_content: str


def parse_book_number_from_filename(name: str) -> int:
    m = FILENAME_BOOK_RE.search(name or "")
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def is_book_of_ones_filename(name: str) -> bool:
    return FILENAME_BOOK1_SUBBOOK_RE.search(name or "") is not None


def insert_reference_headers_for_matches(text: str, pat: re.Pattern[str]) -> str:
    out_parts: list[str] = []
    last_i = 0
    for m in pat.finditer(text):
        start = m.start()
        window = text[max(0, start - 90) : start]
        if "### REFERENCE:" in window:
            continue
        token = m.group(1)
        out_parts.append(text[last_i:start])
        out_parts.append(f"\n\n### REFERENCE: AN {token}\n\n{token}")
        last_i = m.end()
    out_parts.append(text[last_i:])
    return "".join(out_parts)


def apply_sutta_tag_rules(
    text: str,
    *,
    filename: str,
    markers: Dict[str, str] | None = None,
) -> str:
    merged = dict(DEFAULT_MARKERS) if markers is None else dict(markers)
    book_n = parse_book_number_from_filename(filename)
    t = normalize_text_for_sutta_tagging(text)

    if is_book_of_ones_filename(filename):
        t = insert_reference_headers_for_matches(t, PAT_BOOK1_ID3)
        t = insert_reference_headers_for_matches(t, PAT_BOOK1_ID2)
    elif book_n > 0:
        t = insert_reference_headers_for_matches(t, PAT_ANY_ID3)
        t = insert_reference_headers_for_matches(t, PAT_ANY_ID2)

    for pattern, canonical in merged.items():
        t = re.sub(pattern, f"\n\n### REFERENCE: {canonical}\n\n", t)
    return t


def book_from_an_ref(ref: str) -> int:
    m = _BOOK_FROM_REF_RE.search(ref or "")
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def canon_id_from_transcript_id(tid: str) -> str:
    """
    SuttaCentral AN uids are two-level: ``an{book}.{sutta}`` (e.g. ``an4.79``).

    Talk transcripts often use three numbers ``book.vagga.sutta`` (e.g. ``4.8.79``).
    Drop the middle segment so ``4.8.79`` → ``an4.79``. Two-part ids stay ``an4.79``.
    """
    s = (tid or "").strip().lower().replace(" ", "")
    if not s:
        return ""
    parts = [p for p in s.split(".") if p]
    if not parts:
        return ""
    if len(parts) >= 3:
        return f"an{parts[0]}.{parts[-1]}"
    if len(parts) == 2:
        return f"an{parts[0]}.{parts[1]}"
    return f"an{parts[0]}"


def ids_from_ref(ref: str) -> tuple[str, str, int]:
    """
    From a header like ``AN 4.8.79`` or ``AN 4.36 (Doṇa)``:
    - transcript_id: ``4.8.79`` (as spoken / tagged in the talk)
    - canon_id: SuttaCentral-style slug, e.g. ``4.8.79`` → ``an4.79`` (middle segment dropped)
    """
    if not (ref or "").strip():
        return "", "", 0
    m = _REF_NUMERIC_TAIL.search(ref)
    if not m:
        return "", "", 0
    transcript_id = m.group(1).strip()
    canon_id = canon_id_from_transcript_id(transcript_id)
    return transcript_id, canon_id, book_from_an_ref(ref)


def _book_with_file_fallback(ref_book: int, file_book: int) -> int:
    """AN has no book 0; use the book parsed from the source filename when ref is missing or bogus."""
    if ref_book > 0:
        return ref_book
    if file_book > 0:
        return file_book
    return 0


def _segment_record(ref: str, content: str, *, file_book: int = 0) -> Dict[str, Any]:
    tid, cid, bk = ids_from_ref(ref)
    bk = _book_with_file_fallback(bk, file_book)
    return {
        "transcript_id": tid,
        "canon_id": cid,
        "book": bk,
        "transcript_content": (content or "").strip(),
    }


def split_reference_records(text: str, *, file_book: int = 0) -> List[Dict[str, Any]]:
    parts = _REF_SPLIT_RE.split(text or "")
    results: List[Dict[str, Any]] = []
    lead = parts[0].strip() if parts else ""
    if lead:
        results.append(_segment_record("", lead, file_book=file_book))

    for i in range(1, len(parts), 2):
        ref = (parts[i] or "").strip()
        body = (parts[i + 1] if i + 1 < len(parts) else "") or ""
        results.append(_segment_record(ref, body, file_book=file_book))
    return results


def _ref_after_reference_header(line: str) -> str:
    s = line.strip()
    if not s.upper().startswith("### REFERENCE:"):
        return ""
    _, _, tail = s.partition(":")
    return tail.strip()


def split_reference_records_canon_safe(text: str, *, file_book: int = 0) -> List[Dict[str, Any]]:
    if "### REFERENCE:" not in (text or ""):
        t = (text or "").strip()
        if not t:
            return []
        return [_segment_record("", t, file_book=file_book)]

    lines = (text or "").splitlines()
    out: List[Dict[str, Any]] = []
    buf: List[str] = []
    seg_ref = ""
    in_canon = False

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return
        out.append(_segment_record(seg_ref, body, file_book=file_book))

    for ln in lines:
        s = ln.strip()
        if s == CANON_BEGIN:
            in_canon = True
        elif s == CANON_END:
            in_canon = False

        if (not in_canon) and s.startswith("### REFERENCE:"):
            flush()
            seg_ref = _ref_after_reference_header(ln)
            continue
        buf.append(ln)

    flush()
    return out if out else [_segment_record("", (text or "").strip(), file_book=file_book)]


# -----------------------------------------------------------------------------
# Optional: transcript_pipeline Transform (only if package is installed)
# -----------------------------------------------------------------------------

SuttaTagTransform: Any = None

try:
    from transcript_pipeline.context import PipelineContext
    from transcript_pipeline.registry import register
    from transcript_pipeline.transforms.base import Transform
except ImportError:
    pass
else:

    @register("sutta_tags")
    class SuttaTagTransform(Transform):  # type: ignore[misc, valid-type]
        name = "sutta_tags"

        def __init__(self, options: Dict[str, Any] | None = None) -> None:
            super().__init__(options)
            custom = self.options.get("markers")
            if isinstance(custom, dict) and custom:
                self._markers = {str(k): str(v) for k, v in custom.items()}
            else:
                self._markers = dict(DEFAULT_MARKERS)

        def apply(self, text: str, ctx: PipelineContext) -> str:
            src = getattr(ctx, "source_path", None)
            name = getattr(src, "name", "") or ""
            return apply_sutta_tag_rules(text, filename=name, markers=self._markers)


# -----------------------------------------------------------------------------
# CLI: raw/*.txt -> sutta_split/<stem>.json
# -----------------------------------------------------------------------------

_CODE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CODE_DIR.parent
# Full JSON from https://suttacentral.net/api/bilarasuttas/{uid}/{author} (one file per uid per author).
CANONAPI_DIR = _PROJECT_ROOT / "canonapi"


def _remove_legacy_nested_dir(out_root: Path, stem: str) -> None:
    nested = out_root / stem
    if nested.is_dir():
        shutil.rmtree(nested, ignore_errors=True)


def merge_canon_from_previous(
    new_recs: List[Dict[str, Any]],
    old_recs: List[Dict[str, Any]],
) -> None:
    """Copy non-empty ``canon_content`` when ``(transcript_id, canon_id)`` matches by index or key."""
    if not old_recs:
        return
    keyed: Dict[Tuple[str, str], str] = {}
    for orow in old_recs:
        kt = ((orow.get("transcript_id") or "").strip(), (orow.get("canon_id") or "").strip())
        occ = (orow.get("canon_content") or "").strip()
        if occ:
            keyed[kt] = occ

    for i, nr in enumerate(new_recs):
        if (nr.get("canon_content") or "").strip():
            continue
        tid = (nr.get("transcript_id") or "").strip()
        cid = (nr.get("canon_id") or "").strip()
        key = (tid, cid)
        if i < len(old_recs):
            orow = old_recs[i]
            if tid == (orow.get("transcript_id") or "").strip() and cid == (orow.get("canon_id") or "").strip():
                occ = (orow.get("canon_content") or "").strip()
                if occ:
                    nr["canon_content"] = occ
                    continue
        if key in keyed:
            nr["canon_content"] = keyed[key]


def process_raw_file(src: Path, out_root: Path, *, preserve_canon: bool = False) -> tuple[str, int]:
    text = src.read_text(encoding="utf-8", errors="replace")
    tagged = apply_sutta_tag_rules(text, filename=src.name)
    file_book = parse_book_number_from_filename(src.name)
    records = split_reference_records(tagged, file_book=file_book)
    fill_inferred_ids_for_empty_rows(records)
    stem = src.stem
    out_path = out_root.resolve() / f"{stem}.json"
    old_data: List[Dict[str, Any]] | None = None
    if preserve_canon and out_path.is_file():
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                old_data = loaded
        except json.JSONDecodeError:
            pass
    if old_data:
        merge_canon_from_previous(records, old_data)
    for r in records:
        if "canon_content" not in r:
            r["canon_content"] = ""
    _remove_legacy_nested_dir(out_root, stem)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return src.name, len(records)


_SC_BILARA_URL = "https://suttacentral.net/api/bilarasuttas/{uid}/{author}"


def _canonapi_file_path(uid: str, author: str) -> Path:
    auth = re.sub(r"[^\w\-]", "_", author.strip().lower())
    u = re.sub(r"[^\w.\-]", "_", (uid or "").strip().lower())
    return CANONAPI_DIR / auth / f"{u}.json"


def _try_load_canonapi_snapshot(uid: str, author: str) -> Dict[str, Any] | None:
    p = _canonapi_file_path(uid, author)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _translation_from_bilara_dict(data: Dict[str, Any]) -> str:
    tt = data.get("translation_text")
    ko = data.get("keys_order")
    if not isinstance(tt, dict):
        return ""
    if isinstance(ko, list) and ko:
        return "".join(str(tt.get(k, "")) for k in ko).strip()
    return " ".join(str(v) for v in tt.values()).strip()


def _write_canonapi_snapshot(uid: str, author: str, data: Dict[str, Any]) -> None:
    """Persist the raw bilara API JSON under ``canonapi/<author>/<uid>.json``."""
    uid = (uid or "").strip().lower()
    author_l = author.strip().lower()
    if not uid:
        return
    p = _canonapi_file_path(uid, author_l)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_bilara_translation_plain(
    uid: str,
    author: str = "sujato",
    *,
    pause_s: float = 0.0,
    local_only: bool = False,
) -> str:
    """
    Plain English from SuttaCentral bilara API, or the same text loaded from ``canonapi/`` if present.

    Order: (1) ``canonapi/<author>/<uid>.json`` if it exists; (2) unless ``local_only``, HTTP fetch
    and save snapshot. ``pause_s`` applies only before a network request.
    """
    uid = (uid or "").strip().lower()
    if not uid:
        return ""
    author_l = author.strip().lower()
    local = _try_load_canonapi_snapshot(uid, author_l)
    if local is not None:
        return _translation_from_bilara_dict(local)
    if local_only:
        return ""
    if pause_s > 0:
        time.sleep(pause_s)
    url = _SC_BILARA_URL.format(uid=uid, author=author_l)
    req = urllib.request.Request(url, headers={"User-Agent": "dama3-sutta-tagging/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""
    if not isinstance(data, dict):
        return ""
    _write_canonapi_snapshot(uid, author_l, data)
    return _translation_from_bilara_dict(data)


def enrich_segment_records(
    records: List[Dict[str, Any]],
    *,
    author: str = "sujato",
    pause_s: float = 0.2,
    uid_cache: Dict[str, str] | None = None,
    only_missing: bool = False,
    canon_local_only: bool = False,
) -> None:
    """
    For each row: set ``canon_content`` from SuttaCentral using ``canon_id`` (skips fetch when
    ``canon_id`` is empty). Skips when ``transcript_content`` does not mention ``transcript_id``
    (e.g. transition-only lines like ``the next sutta is``). ``transcript_content`` is unchanged.
    Pass ``uid_cache`` to reuse fetched text across multiple files (same dict is updated).
    If ``only_missing`` is True, rows that already have non-empty ``canon_content`` are not
    fetched again. Loads from ``canonapi/<author>/<uid>.json`` first when present; network only
    if missing and ``canon_local_only`` is False.
    """
    cache: Dict[str, str] = uid_cache if uid_cache is not None else {}
    for r in records:
        cid = (r.get("canon_id") or "").strip()
        tid = (r.get("transcript_id") or "").strip()
        body = (r.get("transcript_content") or "").strip()
        if not cid:
            r["canon_content"] = ""
            continue
        if tid and not transcript_content_supports_transcript_id(tid, body):
            r["canon_content"] = ""
            continue
        if only_missing and (r.get("canon_content") or "").strip():
            continue
        if cid not in cache:
            cache[cid] = fetch_bilara_translation_plain(
                cid,
                author,
                pause_s=pause_s,
                local_only=canon_local_only,
            )
        r["canon_content"] = cache[cid]


def enrich_json_files(
    paths: List[Path],
    *,
    author: str = "sujato",
    only_missing: bool = False,
    canon_local_only: bool = False,
) -> None:
    shared_cache: Dict[str, str] = {}
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            raise SystemExit(f"Expected JSON array in {path}")
        enrich_segment_records(
            data,
            author=author,
            uid_cache=shared_cache,
            only_missing=only_missing,
            canon_local_only=canon_local_only,
        )
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Enriched {path.name} ({len(data)} segment(s))")


# -----------------------------------------------------------------------------
# mapping quality, use flag, chains (books 3–6 when mapping ok), text census
# -----------------------------------------------------------------------------

_CHAIN_BOOKS = frozenset({3, 4, 5, 6})
_BOOK_7 = 7

# Default thresholds for mapping_quality from transcript↔canon agreement score [0,1].
_DEFAULT_MAP_OK_MIN = 0.20
_DEFAULT_MAP_SUSPECT_MIN = 0.10

_LEADING_TRANSCRIPT_IDS = re.compile(r"^[\s]*(?:\d+\.)+\d+\s+")


def _strip_diacritics(s: str) -> str:
    if not s:
        return s
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _normalize_words_alignment(s: str) -> str:
    s = _strip_diacritics(s).lower()
    s = re.sub(r"<j>", " ", s)
    s = re.sub(r"[^\w\s\u0100-\u024f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def transcript_opening(transcript: str, max_chars: int = 650) -> str:
    t = (transcript or "").strip()
    t = _LEADING_TRANSCRIPT_IDS.sub("", t, count=1)
    return t[:max_chars]


def canon_opening_for_sim(canon: str, max_chars: int = 700) -> str:
    c = (canon or "").strip()
    m_then = re.search(r"\bThen\b", c)
    if m_then:
        return c[m_then.start() : m_then.start() + max_chars]
    m = re.search(r'[“"]', c)
    if m:
        return c[m.start() : m.start() + max_chars]
    return c[:max_chars]


def _jaccard_words(a: str, b: str) -> float:
    wa = set(_normalize_words_alignment(a).split())
    wb = set(_normalize_words_alignment(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _keyword_overlap_ratio(transcript: str, canon: str) -> float:
    to = _normalize_words_alignment(transcript_opening(transcript))
    co = _normalize_words_alignment(canon_opening_for_sim(canon))
    words = [w for w in co.split() if len(w) >= 4][:40]
    if not words:
        return 0.0
    return sum(1 for w in words if w in to) / len(words)


def mapping_agreement_score(transcript: str, canon: str) -> float:
    """Weighted blend of Jaccard, sequence ratio, keyword overlap on openings (0..1)."""
    to_raw = transcript_opening(transcript)
    co_raw = canon_opening_for_sim(canon)
    n1, n2 = _normalize_words_alignment(to_raw), _normalize_words_alignment(co_raw)
    if not n1 or not n2:
        return 0.0
    seq = SequenceMatcher(None, n1, n2).ratio()
    jac = _jaccard_words(to_raw, co_raw)
    overlap = _keyword_overlap_ratio(transcript, canon)
    return 0.2 * jac + 0.2 * seq + 0.6 * overlap


def classify_mapping_quality(
    score: float,
    *,
    ok_min: float,
    suspect_min: float,
) -> Tuple[str, str]:
    """Return (mapping_quality, short reason). Qualities: ok | suspect | bad."""
    if score >= ok_min:
        return "ok", f"agreement_score={score:.3f}>={ok_min}"
    if score >= suspect_min:
        return (
            "suspect",
            f"agreement_score={score:.3f} in [{suspect_min},{ok_min}) (possible topic mismatch)",
        )
    return "bad", f"agreement_score={score:.3f}<{suspect_min} (likely wrong sutta or garbled text)"


def extract_chains_from_canon(canon: str) -> Tuple[List[List[str]], str]:
    """Heuristic chains from English canon (Sujato-style). Returns (chains, method_tag)."""
    t = (canon or "").replace("<j>", " ")
    if not t.strip():
        return [], "empty"

    if "Experiential confidence in the" in t:
        labels: List[str] = []
        for m in re.finditer(r"Experiential confidence in the\s+([^:]+):", t):
            head = m.group(1).strip()
            if re.match(r"(?i)buddha", head):
                labels.append("experiential confidence in the Buddha")
            elif re.match(r"(?i)teaching", head):
                labels.append("experiential confidence in the teaching")
            elif "Saṅgha" in head or re.match(r"(?i)sangha", head):
                labels.append("experiential confidence in the Saṅgha")
            else:
                labels.append(f"experiential confidence in the {head.split()[0]}")
        if len(labels) >= 3:
            return [labels[:3]], "experiential_confidence_three"

    if (
        "deeds are the field" in t
        and "consciousness is the seed" in t
        and "craving is the moisture" in t
    ):
        return [
            ["deeds", "consciousness", "craving", "regeneration into a new state of existence"],
        ], "field_seed_moisture"

    if re.search(
        r"higher ethics,\s*the higher mind,\s*and\s*the higher wisdom",
        t,
        re.IGNORECASE,
    ):
        return [["higher ethics", "higher mind", "higher wisdom"]], "three_higher_trainings"
    if re.search(
        r"training in the higher ethics,\s*the higher mind,\s*and\s*the higher wisdom",
        t,
        re.IGNORECASE,
    ):
        return [["higher ethics", "higher mind", "higher wisdom"]], "three_higher_trainings_training"

    if "gone for refuge to the Buddha" in t or "refuge to the Buddha, the teaching" in t:
        return [
            [
                "refuge to the Buddha, the teaching, and the Saṅgha",
                "ethical, of good character",
                "freely generous, open-handed, loving to let go",
                "praised by ascetics and brahmins; deities praise them",
            ],
        ], "fragrance_virtue"

    if "precepts and observances" in t and "fruitful" in t.lower():
        return [
            [
                "precepts and observances, lifestyles, and spiritual paths",
                "skillful qualities grow / unskillful decline",
                "fruitful",
            ],
        ], "precepts_fruitful"

    if "A fool is known by three things" in t and "hurtful deeds" in t:
        return [
            [
                "hurtful deeds by way of body, speech, and mind",
                "kind deeds by way of body, speech, and mind",
                "train: shun fool-qualities; follow astute qualities",
            ],
        ], "fool_astute_three"

    if "three kinds of fragrance" in t.lower() and "roots" in t and "heartwood" in t:
        return [["fragrance of roots", "heartwood", "flowers"]], "three_fragrances_wind"

    return [], "unmatched_needs_review"


def apply_alignment_and_chains_to_record(
    r: Dict[str, Any],
    *,
    map_ok_min: float,
    map_suspect_min: float,
) -> None:
    """Set ``mapping_quality``, ``use``, ``chains``, ``chain_extraction`` (chains only if mapping ok)."""
    for _k in tuple(r.keys()):
        if _k.startswith("transcript_canon_"):
            r.pop(_k, None)

    transcript = (r.get("transcript_content") or "").strip()
    canon = (r.get("canon_content") or "").strip()
    try:
        bk = int(r.get("book") or 0)
    except (TypeError, ValueError):
        bk = 0

    if not canon:
        r["use"] = "no"
        r["mapping_agreement_score"] = None
        r["mapping_quality"] = "bad"
        r["mapping_quality_reason"] = "empty_canon_text"
        r["chains"] = []
        r["chain_extraction"] = "empty_canon"
        return

    score = mapping_agreement_score(transcript, canon)
    mq, reason = classify_mapping_quality(score, ok_min=map_ok_min, suspect_min=map_suspect_min)
    r["mapping_agreement_score"] = round(score, 4)
    r["mapping_quality"] = mq
    r["mapping_quality_reason"] = reason
    r["use"] = "yes"

    if bk not in _CHAIN_BOOKS:
        r["chains"] = []
        r["chain_extraction"] = "skipped_book"
    elif mq != "ok":
        r["chains"] = []
        r["chain_extraction"] = "skipped_mapping_quality"
    else:
        chains, method = extract_chains_from_canon(canon)
        r["chains"] = chains
        r["chain_extraction"] = method


def score_and_chains_json_dir(
    out_dir: Path,
    *,
    map_ok_min: float = _DEFAULT_MAP_OK_MIN,
    map_suspect_min: float = _DEFAULT_MAP_SUSPECT_MIN,
    census_txt: Path | None = None,
    alignment_report: Path | None = None,
    mapping_review_book7: Path | None = None,
) -> None:
    paths = sorted(p for p in out_dir.glob("*.json") if p.is_file())
    total = 0
    for path in paths:
        try:
            raw_text = path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print(f"SKIP (invalid JSON): {path.name} — {e}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            apply_alignment_and_chains_to_record(
                row,
                map_ok_min=map_ok_min,
                map_suspect_min=map_suspect_min,
            )
            total += 1
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Annotated {path.name} ({len(data)} segment(s))")
    print(f"Done. {len(paths)} file(s), {total} segment row(s) annotated -> {out_dir}")
    if census_txt:
        write_census_txt(out_dir, census_txt)
        print(f"Wrote census -> {census_txt}")
    if alignment_report:
        write_alignment_report_tsv(out_dir, alignment_report)
        print(f"Wrote segment report -> {alignment_report}")
    if mapping_review_book7:
        write_book7_mapping_review_tsv(out_dir, mapping_review_book7)
        print(f"Wrote book 7 mapping review -> {mapping_review_book7}")


def write_alignment_report_tsv(out_dir: Path, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "json_file\tsegment_index\ttranscript_id\tcanon_id\tbook\tuse\tmapping_quality\tmapping_agreement_score\tchain_extraction",
    ]
    for path in sorted(p for p in out_dir.glob("*.json") if p.is_file()):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            lines.append(
                "\t".join(
                    [
                        path.name,
                        str(i),
                        str(row.get("transcript_id", "")),
                        str(row.get("canon_id", "")),
                        str(row.get("book", "")),
                        str(row.get("use", "")),
                        str(row.get("mapping_quality", "")),
                        str(row.get("mapping_agreement_score", "")),
                        str(row.get("chain_extraction", "")),
                    ]
                )
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tsv_one_line(s: str, max_len: int = 600) -> str:
    t = (s or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    if len(t) > max_len:
        return t[: max_len - 3] + "..."
    return t


def write_book7_mapping_review_tsv(json_root: Path, report_path: Path) -> None:
    """Suspect/bad rows in book 7 only: openings + mismatch reason for manual review."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "json_file\tsegment_index\ttranscript_id\tcanon_id\tmapping_quality\ttranscript_opening\tcanon_opening\tmismatch_reason",
    ]
    for path in sorted(p for p in json_root.glob("*.json") if p.is_file()):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            if _book_int(row) != _BOOK_7:
                continue
            mq = str(row.get("mapping_quality") or "")
            if mq not in ("suspect", "bad"):
                continue
            tid = (row.get("transcript_id") or "").strip()
            cid = (row.get("canon_id") or "").strip()
            tr = (row.get("transcript_content") or "").strip()
            cn = (row.get("canon_content") or "").strip()
            reason = str(row.get("mapping_quality_reason") or "")
            lines.append(
                "\t".join(
                    [
                        path.name,
                        str(i),
                        tid,
                        cid,
                        mq,
                        _tsv_one_line(transcript_opening(tr)),
                        _tsv_one_line(canon_opening_for_sim(cn)),
                        _tsv_one_line(reason, max_len=400),
                    ]
                )
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _book_int(r: Dict[str, Any]) -> int:
    try:
        return int(r.get("book") or 0)
    except (TypeError, ValueError):
        return 0


def write_census_txt(json_root: Path, txt_path: Path) -> None:
    segments: List[Dict[str, Any]] = []
    for jp in sorted(json_root.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict):
                segments.append(row)

    n_total = len(segments)
    n_cid = sum(1 for r in segments if (r.get("canon_id") or "").strip())
    n_empty_canon = sum(
        1 for r in segments if (r.get("canon_id") or "").strip() and not (r.get("canon_content") or "").strip()
    )
    n_use_yes = sum(1 for r in segments if r.get("use") == "yes")
    n_use_no = sum(1 for r in segments if r.get("use") == "no")
    n_b36 = sum(1 for r in segments if _book_int(r) in _CHAIN_BOOKS)

    mq_counts = Counter(str(r.get("mapping_quality") or "") for r in segments)
    mq_with_cid = Counter(
        str(r.get("mapping_quality") or "")
        for r in segments
        if (r.get("canon_id") or "").strip()
    )

    chain_nonempty = sum(
        1
        for r in segments
        if _book_int(r) in _CHAIN_BOOKS
        and r.get("chains")
        and isinstance(r.get("chains"), list)
        and any(r.get("chains"))
    )
    chain_unmatched = sum(
        1
        for r in segments
        if _book_int(r) in _CHAIN_BOOKS and r.get("chain_extraction") == "unmatched_needs_review"
    )

    chain_methods = Counter()
    for r in segments:
        if _book_int(r) in _CHAIN_BOOKS:
            chain_methods[str(r.get("chain_extraction") or "")] += 1

    lines_out = [
        f"total_segments: {n_total}",
        f"segments_with_canon_id: {n_cid}",
        f"empty_canon_with_canon_id: {n_empty_canon}",
        f"use_yes: {n_use_yes}",
        f"use_no: {n_use_no}",
        "",
        "mapping_quality_counts_all_segments:",
        f"  ok: {mq_counts.get('ok', 0)}",
        f"  suspect: {mq_counts.get('suspect', 0)}",
        f"  bad: {mq_counts.get('bad', 0)}",
        "",
        "mapping_quality_counts_segments_with_canon_id:",
        f"  ok: {mq_with_cid.get('ok', 0)}",
        f"  suspect: {mq_with_cid.get('suspect', 0)}",
        f"  bad: {mq_with_cid.get('bad', 0)}",
        "",
        f"segments_books_3_to_6: {n_b36}",
        f"chains_nonempty_books_3_6: {chain_nonempty}",
        f"chain_extraction_unmatched_needs_review_books_3_6: {chain_unmatched}",
        "",
        "chain_extraction_counts_books_3_to_6:",
    ]
    for method, cnt in sorted(chain_methods.items(), key=lambda x: (-x[1], x[0])):
        lines_out.append(f"  {method}: {cnt}")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def row_missing_canon_text(r: Dict[str, Any]) -> bool:
    """True when ``canon_id`` is set but ``canon_content`` is empty (no fetched text)."""
    cid = (r.get("canon_id") or "").strip()
    cc = (r.get("canon_content") or "").strip()
    return bool(cid) and not cc


def drop_rows_missing_canon_content(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if not row_missing_canon_text(r)]


def format_missing_canon_report(missing: List[tuple[str, int, str, str]]) -> str:
    """Build the standard missing-canon report text from collected rows."""
    pairs = sorted({(row[3], (row[2] or "").strip() or "(empty)") for row in missing})
    unique_ids = sorted({row[3] for row in missing})
    lines = [
        f"Total segments with canon_id but empty canon_content: {len(missing)}",
        "",
        "Unique (canon_id, transcript_id) pairs — tab-separated columns:",
        f"Count: {len(pairs)}",
        "canon_id\ttranscript_id",
        *[f"{cid}\t{tid}" for cid, tid in pairs],
        "",
        f"Distinct canon_id only (count {len(unique_ids)}):",
        *unique_ids,
        "",
        "--- Full segment list (tab-separated) ---",
        "json_file\tsegment_index\ttranscript_id\tcanon_id",
        *[f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}" for row in missing],
    ]
    return "\n".join(lines)


def write_missing_canon_report(report_path: Path, *, json_root: Path) -> tuple[int, int]:
    """
    Tab-separated report: unique (canon_id, transcript_id) pairs, distinct canon_ids,
    full per-row TSV. Returns (segment_rows_missing, distinct_canon_ids).
    """
    missing: List[tuple[str, int, str, str]] = []
    for jp in sorted(json_root.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for i, r in enumerate(data):
            if not row_missing_canon_text(r):
                continue
            cid = (r.get("canon_id") or "").strip()
            tid = (r.get("transcript_id") or "").strip()
            missing.append((jp.name, i, tid, cid))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_missing_canon_report(missing), encoding="utf-8")
    return len(missing), len({row[3] for row in missing})


def write_missing_canon_report_from_raw(
    report_path: Path,
    *,
    raw_dir: Path,
    pattern: str | None,
    author: str,
    canon_local_only: bool = False,
) -> tuple[int, int]:
    """
    Rebuild segments from each ``raw/*.txt``, enrich from SuttaCentral, then emit the same
    missing-canon report (for recovering the pre-drop list after ``canon_content`` rows were removed).
    """
    shared_cache: Dict[str, str] = {}
    missing: List[tuple[str, int, str, str]] = []
    files = sorted(p for p in raw_dir.glob("*.txt") if p.is_file())
    if pattern:
        files = [p for p in files if pattern in p.name]
    for src in files:
        text = src.read_text(encoding="utf-8", errors="replace")
        tagged = apply_sutta_tag_rules(text, filename=src.name)
        file_book = parse_book_number_from_filename(src.name)
        records = split_reference_records(tagged, file_book=file_book)
        fill_inferred_ids_for_empty_rows(records)
        enrich_segment_records(
            records,
            author=author,
            uid_cache=shared_cache,
            canon_local_only=canon_local_only,
        )
        label = f"{src.stem}.json"
        for i, r in enumerate(records):
            if not row_missing_canon_text(r):
                continue
            cid = (r.get("canon_id") or "").strip()
            tid = (r.get("transcript_id") or "").strip()
            missing.append((label, i, tid, cid))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_missing_canon_report(missing), encoding="utf-8")
    return len(missing), len({row[3] for row in missing})


def drop_missing_canon_rows_in_json_dir(out_dir: Path) -> tuple[int, int, int]:
    """
    Remove rows with ``canon_id`` but empty ``canon_content`` from each ``*.json``.
    Returns ``(files_written, segments_before, segments_after)``.
    """
    files_written = 0
    segments_before = 0
    segments_after = 0
    for jp in sorted(out_dir.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"Expected JSON array in {jp}")
        segments_before += len(data)
        filtered = drop_rows_missing_canon_content(data)
        segments_after += len(filtered)
        jp.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
        files_written += 1
    return files_written, segments_before, segments_after


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Write sutta_split/<stem>.json for each raw transcript (flat JSON only).",
    )
    ap.add_argument(
        "pattern",
        nargs="?",
        default=None,
        help="Optional substring filter on filename; omit for all .txt in raw/",
    )
    ap.add_argument("--raw-dir", type=Path, default=_PROJECT_ROOT / "raw")
    ap.add_argument("--out-dir", type=Path, default=_PROJECT_ROOT / "sutta_split")
    ap.add_argument(
        "--enrich-json",
        type=Path,
        nargs="+",
        metavar="JSON",
        help="Add canon_content from SuttaCentral (bilara); no raw rebuild",
    )
    ap.add_argument(
        "--enrich-all-json",
        action="store_true",
        help="Same as --enrich-json but for every *.json under --out-dir",
    )
    ap.add_argument(
        "--sc-author",
        default="sujato",
        metavar="UID",
        help="SuttaCentral translation author_uid for bilara (default: sujato)",
    )
    ap.add_argument(
        "--report-missing-canon",
        action="store_true",
        help="Write missing_canon_report.txt at project root (canon_id set, empty canon_content)",
    )
    ap.add_argument(
        "--drop-missing-canon-json",
        action="store_true",
        help="Remove segment rows with canon_id but empty canon_content from all *.json in --out-dir",
    )
    ap.add_argument(
        "--report-missing-canon-from-raw",
        action="store_true",
        help="Write missing_canon_report.txt from raw/*.txt + bilara fetch (slow; does not modify JSON)",
    )
    ap.add_argument(
        "--rebuild-all-json",
        action="store_true",
        help="With raw rebuild: process every raw/*.txt (ignore optional pattern filter)",
    )
    ap.add_argument(
        "--preserve-canon-on-rebuild",
        action="store_true",
        help="When rebuilding from raw, copy canon_content from existing JSON when transcript_id/canon_id match",
    )
    ap.add_argument(
        "--only-missing-canon",
        action="store_true",
        help="With --enrich-all-json / --enrich-json: fetch SuttaCentral only when canon_content is empty",
    )
    ap.add_argument(
        "--canon-local-only",
        action="store_true",
        help="With enrichment: use only canonapi/ snapshots, never call the bilara API (missing UIDs get empty text)",
    )
    ap.add_argument(
        "--score-and-chains",
        action="store_true",
        help="Score mapping_quality (transcript↔canon), set use, chains only when mapping ok (books 3–6), census, book 7 review TSV.",
    )
    ap.add_argument(
        "--map-ok-min",
        type=float,
        default=_DEFAULT_MAP_OK_MIN,
        metavar="S",
        help=f"Agreement score >= S → mapping_quality ok (default: {_DEFAULT_MAP_OK_MIN})",
    )
    ap.add_argument(
        "--map-suspect-min",
        type=float,
        default=_DEFAULT_MAP_SUSPECT_MIN,
        metavar="S",
        help=f"Agreement score >= S but < map-ok-min → suspect; below → bad (default: {_DEFAULT_MAP_SUSPECT_MIN})",
    )
    ap.add_argument(
        "--census-txt",
        type=Path,
        nargs="?",
        const=None,
        default=None,
        metavar="PATH",
        help="With --score-and-chains: text census path (default: output/census.txt when omitted)",
    )
    ap.add_argument(
        "--write-alignment-report",
        type=Path,
        default=None,
        metavar="TSV",
        help="With --score-and-chains: write tab-separated segment QA report",
    )
    ap.add_argument(
        "--mapping-review-book7",
        type=Path,
        nargs="?",
        const=None,
        default=None,
        metavar="PATH",
        help="With --score-and-chains: book 7 suspect/bad review TSV (default: metric/mapping_review_book7.tsv)",
    )
    args = ap.parse_args()

    if args.score_and_chains:
        root = args.out_dir.resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        if args.map_suspect_min >= args.map_ok_min:
            raise SystemExit("--map-suspect-min must be < --map-ok-min")
        census_path = (
            args.census_txt.resolve()
            if args.census_txt is not None
            else (_PROJECT_ROOT / "output" / "census.txt")
        )
        review_b7 = (
            args.mapping_review_book7.resolve()
            if args.mapping_review_book7 is not None
            else (_PROJECT_ROOT / "metric" / "mapping_review_book7.tsv")
        )
        score_and_chains_json_dir(
            root,
            map_ok_min=args.map_ok_min,
            map_suspect_min=args.map_suspect_min,
            census_txt=census_path,
            alignment_report=args.write_alignment_report.resolve() if args.write_alignment_report else None,
            mapping_review_book7=review_b7,
        )
        return

    if args.report_missing_canon_from_raw:
        raw_dir = args.raw_dir.resolve()
        if not raw_dir.is_dir():
            raise SystemExit(f"Not a directory: {raw_dir}")
        n_seg, n_uid = write_missing_canon_report_from_raw(
            _PROJECT_ROOT / "missing_canon_report.txt",
            raw_dir=raw_dir,
            pattern=args.pattern,
            author=args.sc_author,
            canon_local_only=args.canon_local_only,
        )
        rp = _PROJECT_ROOT / "missing_canon_report.txt"
        print(f"Wrote {rp} — {n_seg} segment row(s), {n_uid} distinct canon_id (rebuilt from raw + enrich)")
        return

    if args.report_missing_canon:
        root = args.out_dir.resolve()
        n_seg, n_uid = write_missing_canon_report(_PROJECT_ROOT / "missing_canon_report.txt", json_root=root)
        print(f"Wrote {_PROJECT_ROOT / 'missing_canon_report.txt'} — {n_seg} segment row(s), {n_uid} distinct canon_id")
        return

    if args.drop_missing_canon_json:
        root = args.out_dir.resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        fw, before, after = drop_missing_canon_rows_in_json_dir(root)
        dropped = before - after
        print(f"Dropped {dropped} segment row(s) ({before} -> {after}), {fw} file(s) -> {root}")
        return

    if args.enrich_all_json:
        root = args.out_dir.resolve()
        paths = sorted(p for p in root.glob("*.json") if p.is_file())
        if not paths:
            raise SystemExit(f"No .json files in {root}")
        enrich_json_files(
            paths,
            author=args.sc_author,
            only_missing=args.only_missing_canon,
            canon_local_only=args.canon_local_only,
        )
        print(f"Done. {len(paths)} file(s) enriched -> {root}")
        return

    if args.enrich_json:
        for p in args.enrich_json:
            rp = p.resolve()
            if not rp.is_file():
                raise SystemExit(f"Not a file: {p}")
        enrich_json_files(
            [p.resolve() for p in args.enrich_json],
            author=args.sc_author,
            only_missing=args.only_missing_canon,
            canon_local_only=args.canon_local_only,
        )
        return

    raw_dir = args.raw_dir.resolve()
    files = sorted(p for p in raw_dir.glob("*.txt") if p.is_file())
    if not args.rebuild_all_json and args.pattern:
        files = [p for p in files if args.pattern in p.name]
    if not files:
        raise SystemExit(
            f"No .txt files in {raw_dir}" + (f" matching {args.pattern!r}" if args.pattern else "")
        )

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    total_segs = 0
    for src in files:
        name, n = process_raw_file(src, out_root, preserve_canon=args.preserve_canon_on_rebuild)
        total_segs += n
        print(f"{name} -> {n} segment(s)")

    print(f"Done. {len(files)} file(s), {total_segs} segment row(s) -> {out_root}")


if __name__ == "__main__":
    main()
