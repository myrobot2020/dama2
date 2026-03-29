"""Parse per-lecture metadata from transcript filenames; split sutta segments by REFERENCE headers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

AN_REF_RE = re.compile(r"AN\s+(\d+(?:\.\d+)*)", re.IGNORECASE)


def parse_lecture_metadata(file_path: Path) -> Dict[str, Any]:
    """Fields stored on every Chroma chunk; use 0 / '' when unknown."""
    name = file_path.name
    stem = file_path.stem
    meta: Dict[str, Any] = {
        "lecture_ord": 0,
        "an_book": 0,
        "an_book_part": "",
        "graph_lecture_id": stem,
        "nikaya": "Anguttara",
        "commentary_role": "teacher_lecture",
        "commentary_on_nikaya": "Anguttara",
        "teacher": "",
    }
    mord = re.match(r"^(\d{3})_", name)
    if mord:
        meta["lecture_ord"] = int(mord.group(1))
    mbook = re.search(r"Book\s+(\d+)([A-Za-z])?", name, re.I)
    if mbook:
        meta["an_book"] = int(mbook.group(1))
        meta["an_book_part"] = (mbook.group(2) or "").strip()
    mteach = re.search(r"\sby\s+(.+?)(?:\.txt)?$", name, re.I)
    if mteach:
        meta["teacher"] = mteach.group(1).strip()[:200]
    return meta


def normalize_sutta_ref(header_line: str) -> str:
    """e.g. '### REFERENCE: AN 4.36 (Doṇa)' -> 'AN_4_36'."""
    m = AN_REF_RE.search(header_line)
    if not m:
        return ""
    return "AN_" + m.group(1).replace(".", "_")


def split_sutta_segments(text: str) -> List[Tuple[str, str]]:
    """
    Split on ### REFERENCE: boundaries.
    Returns list of (an_sutta_ref_normalized, segment_body_text).
    """
    if "### REFERENCE:" not in text:
        return [("", text)]

    parts = re.split(r"(?=(?:^|\n)### REFERENCE:)", text)
    out: List[Tuple[str, str]] = []
    for p in parts:
        p_st = p.strip()
        if not p_st:
            continue
        if p_st.startswith("### REFERENCE:"):
            nl = p_st.find("\n")
            hdr = p_st[:nl] if nl != -1 else p_st
            body = p_st[nl + 1 :].strip() if nl != -1 else ""
            ref = normalize_sutta_ref(hdr)
            out.append((ref, body))
        else:
            out.append(("", p_st))
    return out
