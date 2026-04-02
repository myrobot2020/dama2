from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import Counter, defaultdict

from openai import OpenAI
import urllib.error
import urllib.request
import time

# =========================
# CONFIG / SPLIT
# =========================

SPLIT_MARKERS: List[str] = [
    "that's the end of the sutta",
    "that is the end of the sutta",
    "the next sutta is",
    "now we come to another sutta",
    "another very interesting sutta",
    "the sutta continues",
]

MIN_ROW_CHARS = 300
MAX_UNITS_PER_ROW = 3
TOP_K_DEFAULT = 8

_TRIPLE_ID_RE = re.compile(r"(?<!\d)(\d{1,2}\.\d{1,3}\.\d{1,3})(?!\d)")

# Keep prompts bounded for faster Ollama generation + fewer timeouts.
MAX_UNIT_PROMPT_CHARS = 2000

# Default folder (under project root) to save debug artifacts.
DEFAULT_DEBUG_DIR = Path("debug") / "mapper"

# Retrieval heuristics (also used for reporting).
FALLBACK_TOP_SCORE_MIN = 0.22
FALLBACK_GAP_MIN = 0.03
SMALL_GAP_MAX = 0.02  # "gap is small" reporting threshold


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_apostrophes(text: str) -> str:
    # Normalize common unicode apostrophes to ASCII for robust matching.
    return (text or "").replace("’", "'").replace("‘", "'")


def _normalize_sutta_asr_errors_for_splitting(text: str) -> str:
    """
    The ASR transcripts sometimes mis-recognize 'sutta' as 'suta'/'utah'/'nexuta' etc.
    We normalize these so SPLIT_MARKERS can match reliably.
    """
    t = _normalize_apostrophes(text)
    t = re.sub(r"\bsuta\b", "sutta", t, flags=re.IGNORECASE)
    t = re.sub(r"\butah\b", "sutta", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(nexuta|nexus)\b", "next sutta", t, flags=re.IGNORECASE)
    t = re.sub(r"\bend of the so\b", "end of the sutta", t, flags=re.IGNORECASE)
    return t


def normalize_for_marker_search(text: str) -> str:
    return _normalize_ws(_normalize_sutta_asr_errors_for_splitting(text)).lower()


def normalize_for_similarity(text: str) -> str:
    t = _normalize_ws(_normalize_apostrophes(text)).lower()
    t = re.sub(r"[^\w\s\u0080-\u024f]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# =========================
# PYTHON DATA MODEL
# =========================


@dataclass
class MappingResult:
    teacher_id: str
    unit_id: str
    book: int
    raw_unit_text: str
    best_canon_ids: List[str] = field(default_factory=list)
    primary_canon_id: Optional[str] = None
    confidence: float = 0.0
    is_mixed: bool = False
    needs_review: bool = False
    evidence_phrases: List[str] = field(default_factory=list)
    reason: str = ""
    match_type: str = "unknown"
    status: str = "needs_review"


@dataclass
class RunStats:
    total_units: int = 0
    llm_chose_nonempty: int = 0
    llm_returned_unknown: int = 0
    fallback_used: int = 0
    accepted: int = 0
    accept_with_flag: int = 0
    needs_review: int = 0

    primary_freq: Counter[str] = field(default_factory=Counter)
    book_totals: Counter[int] = field(default_factory=Counter)
    book_needs_review: Counter[int] = field(default_factory=Counter)

    low_top_score_units: int = 0
    small_gap_units: int = 0

    # Keep small samples for investigation (not huge logs).
    sample_low_score_unit_ids: List[str] = field(default_factory=list)
    sample_small_gap_unit_ids: List[str] = field(default_factory=list)

    def add_low_score(self, unit_id: str) -> None:
        self.low_top_score_units += 1
        if len(self.sample_low_score_unit_ids) < 12:
            self.sample_low_score_unit_ids.append(unit_id)

    def add_small_gap(self, unit_id: str) -> None:
        self.small_gap_units += 1
        if len(self.sample_small_gap_unit_ids) < 12:
            self.sample_small_gap_unit_ids.append(unit_id)


def _jsonl_to_pretty_json(jsonl_path: Path, pretty_json_path: Path) -> int:
    """
    Convert JSONL (one object per line) to a pretty-printed JSON array.
    Returns number of records written.
    """
    rows: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    pretty_json_path.parent.mkdir(parents=True, exist_ok=True)
    pretty_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(rows)


# =========================
# TEACHER ROW EXTRACTION
# =========================


def extract_teacher_rows_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Metadata extraction: split the transcript text into "rows" by triple-id occurrences.

    IMPORTANT: This uses teacher_id only as metadata for splitting and output labeling.
    The LLM mapping must never infer canon ids from teacher numbering (enforced in prompts
    and runtime sanitization).
    """
    matches = list(_TRIPLE_ID_RE.finditer(text or ""))
    if not matches:
        return []

    rows: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        teacher_id = m.group(1)
        book = int(teacher_id.split(".")[0])
        row_text = (text[start:end] or "").strip()
        if not row_text:
            continue
        rows.append({"teacher_id": teacher_id, "book": book, "text": row_text})
    return rows


# =========================
# SPLIT INTO SEMANTIC UNITS
# =========================


def split_units_for_row(
    *,
    teacher_id: str,
    book: int,
    row_text: str,
    min_row_chars: int = MIN_ROW_CHARS,
    max_units_per_row: int = MAX_UNITS_PER_ROW,
) -> List[Dict[str, Any]]:
    """
    Split a transcript row into semantic commentary units using SPLIT_MARKERS.

    Rules implemented from the spec:
    - Split markers only when marker appears with enough surrounding text
    - Keep marker in earlier chunk (split at marker end)
    - If row is short, do not split
    - If a row seems to discuss two distinct suttas, allow up to max_units_per_row=3
    """
    row_text_flat = _normalize_ws(row_text)

    # If a row is short, do not split.
    if len(row_text_flat) < min_row_chars:
        return [{"unit_id": f"{teacher_id}#a", "text": row_text_flat, "book": book}]

    text_norm = normalize_for_marker_search(row_text_flat)
    text_len = len(text_norm)

    # Find split boundaries (marker end positions).
    boundaries: List[int] = []
    for marker in SPLIT_MARKERS:
        marker_norm = marker.lower()
        for m in re.finditer(re.escape(marker_norm), text_norm, flags=0):
            boundaries.append(m.end())  # marker kept in earlier chunk

    boundaries = sorted(set(boundaries))
    if not boundaries:
        return [{"unit_id": f"{teacher_id}#a", "text": text_norm, "book": book}]

    # Only split when enough surrounding text exists.
    min_before = 150
    min_after = 60

    chunks: List[str] = []
    last_idx = 0
    for b in boundaries:
        if b <= last_idx:
            continue
        before_len = b - last_idx
        after_len = text_len - b
        if before_len < min_before or after_len < min_after:
            continue
        chunk = text_norm[last_idx:b].strip()
        if chunk:
            chunks.append(chunk)
        last_idx = b
        if len(chunks) >= max(1, max_units_per_row - 1):
            break

    tail = text_norm[last_idx:].strip()
    if tail:
        chunks.append(tail)

    # Hard cap; merge remainder.
    if len(chunks) > max_units_per_row:
        chunks = chunks[: max_units_per_row - 1] + [" ".join(chunks[max_units_per_row - 1 :]).strip()]

    # Filter tiny chunks (but keep a fallback).
    chunks = [c for c in chunks if len(c) >= 50] or [text_norm]

    units: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks[:max_units_per_row]):
        units.append({"unit_id": f"{teacher_id}#{chr(ord('a') + i)}", "text": c, "book": book})
    return units


# =========================
# CANON LOAD + INDEX + CANDIDATES
# =========================


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "from",
    "as",
    "it",
    "this",
    "that",
    "these",
    "those",
    "are",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "my",
    "our",
    "us",
    "me",
    "your",
    "very",
    "then",
    "when",
    "what",
    "where",
    "which",
    "who",
    "why",
    "how",
    "also",
    "just",
    "only",
    "so",
    "thus",
    "listen",
    "apply",
    "mind",
    "buddha",
    "dhamma",
    "mendicants",
    "monks",
    "sutta",
}


def parse_book_from_canon_id(canon_id: str) -> int:
    m = re.match(r"an(\d+)\.", (canon_id or "").strip().lower())
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def extract_title_opening_keywords(canon_content: str, *, max_opening_chars: int = 300) -> Tuple[str, str, List[str]]:
    """
    Extract required candidate fields (title/opening/keywords) from `canon_content`.
    Heuristic only; retrieval is still content-similarity based.
    """
    t = canon_content or ""

    # Find first opening quote marker.
    quote_pos = None
    for q in ["“", '"']:
        p = t.find(q)
        if p != -1 and (quote_pos is None or p < quote_pos):
            quote_pos = p

    if quote_pos is None:
        title = _normalize_ws(t)[:120]
        opening = _normalize_ws(t)[:max_opening_chars]
        return title, opening, []

    before_quote = t[:quote_pos].strip().replace("\n", " ")
    after_quote = t[quote_pos:].strip()
    opening = after_quote[1 : 1 + max_opening_chars] if after_quote[:1] in {"“", '"'} else after_quote[:max_opening_chars]
    opening = _normalize_ws(opening)

    # Title heuristic: remove the leading "Numbered Discourses X.Y Z." prefix.
    prefix = before_quote
    prefix = re.sub(r"^\s*Numbered\s+Discourses\s+", "", prefix, flags=re.IGNORECASE)
    title = re.sub(r"^[\d.]+\s+\d+\.\s+", "", prefix).strip()
    if not title:
        title = before_quote[-120:].strip()
    title = title[:120]

    # Keyword extraction: token frequencies from title + opening.
    kw_src = f"{title} {opening}".lower()
    kw_tokens = re.findall(r"[a-z\u0080-\u024f']{3,}", kw_src)
    freq: Dict[str, int] = {}
    for tok in kw_tokens:
        if tok in STOPWORDS:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    sorted_kws = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    keywords = [k for k, _ in sorted_kws]
    return title, opening, keywords


def load_canon_corpus(canon_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load `canon_sutta.txt` where each line is a JSON object containing:
      { "canon_id": "...", "canon_content": "..." }
    """
    book_index: Dict[int, List[Dict[str, Any]]] = {}
    with canon_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            canon_id = str(data.get("canon_id") or "")
            canon_content = str(data.get("canon_content") or "")
            if not canon_id or not canon_content:
                continue

            book = parse_book_from_canon_id(canon_id)
            if book <= 0:
                continue

            title, opening, keywords = extract_title_opening_keywords(canon_content)
            book_index.setdefault(book, []).append(
                {"canon_id": canon_id, "title": title, "opening": opening, "keywords": keywords}
            )
    return book_index


def retrieve_candidates_from_canon(
    *,
    unit_text: str,
    canon_by_book: Dict[int, List[Dict[str, Any]]],
    book: int,
    k: int,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k candidate AN suttas from canon.
    Same-book first (required by spec).
    """
    unit_key = normalize_for_similarity(unit_text)[:320]
    same_book = list(canon_by_book.get(book, []) or [])
    if not same_book:
        return []

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for c in same_book:
        opening_key = normalize_for_similarity(c.get("opening") or "")[:320]
        score = SequenceMatcher(None, unit_key, opening_key).ratio()
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def retrieve_candidates_with_scores(
    *,
    unit_text: str,
    canon_by_book: Dict[int, List[Dict[str, Any]]],
    book: int,
    k: int,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Same as retrieve_candidates_from_canon, but returns (score, candidate)."""
    unit_key = normalize_for_similarity(unit_text)[:320]
    same_book = list(canon_by_book.get(book, []) or [])
    if not same_book:
        return []
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for c in same_book:
        opening_key = normalize_for_similarity(c.get("opening") or "")[:320]
        score = SequenceMatcher(None, unit_key, opening_key).ratio()
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def build_candidate_block(candidates: Sequence[Dict[str, Any]]) -> str:
    """
    Build candidate block exactly in the required enumerated format.
    """
    out_parts: List[str] = []
    for i, c in enumerate(candidates, start=1):
        kw_list = c.get("keywords") or []
        kw_text = ", ".join(str(x) for x in kw_list)
        out_parts.append(
            f"{i}. canon_id: {c.get('canon_id','')}\n"
            f"title: {c.get('title','')}\n"
            f"opening: {c.get('opening','')}\n"
            f"keywords: {kw_text}\n"
        )
    return "".join(out_parts).rstrip()


# =========================
# LLM CALL (TWO-STEP)
# =========================


SYSTEM_PROMPT = (
    "You are a precise Buddhist text alignment engine. "
    "You map teacher commentary segments to Anguttara Nikaya canon ids using content only. "
    "Never infer from teacher numbering. "
    "Never fabricate certainty. "
    "Return valid JSON only."
)

ULTRA_INSTRUCTION = "The teacher’s numbering is not canonical and must not be used as evidence."


def load_api_key() -> str:
    env = os.environ.get("OPENAI_API_KEY")
    if env and env.strip():
        return env.strip()
    key_path = Path(__file__).resolve().parent.parent / "creds" / "openaikey.txt"
    return key_path.read_text(encoding="utf-8").strip()


def _ollama_chat(
    *,
    host: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float = 0.0,
    timeout_s: float = 180.0,
) -> str:
    """
    Call Ollama's /api/chat and return assistant content.
    Expected Ollama JSON: { "message": { "content": "..." }, ... }.
    """
    url = host.rstrip("/") + "/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
    parsed = json.loads(body)
    msg = parsed.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Ollama response missing content: {parsed}")
    return content


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    sub = raw[start : end + 1]
    obj2 = json.loads(sub)
    if not isinstance(obj2, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj2


def clean_llm_text_for_json_parse(text: str) -> str:
    """
    Save the exact raw response, but parse from a minimally-cleaned version.
    Removes common markdown code fences and trims whitespace.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    return raw


def sanitize_unit_text(unit_text: str) -> str:
    # Remove triple-id patterns so unit text doesn’t carry teacher-id evidence.
    return re.sub(r"(?<!\d)\d{1,2}\.\d{1,3}\.\d{1,3}(?!\d)", "[id]", unit_text or "")


def call_llm_two_step(
    *,
    client: OpenAI,
    model: str,
    teacher_id: str,
    unit_id: str,
    book: int,
    unit_text: str,
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    unit_text_sanitized = sanitize_unit_text(unit_text)
    if unit_text_sanitized:
        unit_text_sanitized = unit_text_sanitized[:MAX_UNIT_PROMPT_CHARS]
    candidate_block = build_candidate_block(candidates)

    # Step A: Extract anchors.
    anchors_prompt = f"""
{ULTRA_INSTRUCTION}

The teacher’s numbering is not canonical and must not be used as evidence.

You are analyzing a teacher commentary segment.

Task:
Extract the key doctrinal anchor phrases that would identify the underlying sutta.

Rules:
- Use only phrases actually supported by the text.
- Prefer distinctive doctrinal motifs.
- Return JSON only.

JSON schema:
{{
  "topic_summary": "",
  "anchor_phrases": [],
  "possible_motifs": []
}}

Teacher segment:
teacher_id: {teacher_id}
unit_id: {unit_id}
book: {book}

Teacher segment text:
{unit_text_sanitized}

Return JSON only. No markdown. No extra text.
""".strip()

    try:
        a_resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": anchors_prompt},
            ],
        )
        anchors_text = a_resp.choices[0].message.content or ""
        anchors_json = parse_json_object(anchors_text)
    except Exception as e:
        return {
            "best_canon_ids": [],
            "primary_canon_id": None,
            "confidence": 0.0,
            "is_mixed": False,
            "needs_review": True,
            "evidence_phrases": [],
            "reason": f"LLM Step A failed: {e}",
            "match_type": "unknown",
        }

    # Step B: Map anchors to candidates (strict schema).
    mapping_prompt = f"""
You must choose from the provided candidate canon ids.

Rules:
- The teacher numbering is not canonical and must not be used as evidence.
- Use only the segment text and candidate entries.
- You must either:
  A) choose one or more canon ids from the candidate list, or
  B) return match_type="unknown" and needs_review=true.
- If match_type is "exact", "probable", or "mixed", then best_canon_ids must be non-empty.
- Every canon id in best_canon_ids must be copied exactly from the candidate list.
- Do not invent canon ids.
- Output JSON only.

Allowed candidate ids:
{json.dumps([str(c.get("canon_id") or "") for c in candidates], ensure_ascii=False)}

Segment:
{unit_text_sanitized}

Candidates:
{candidate_block}

Return exactly:
{{
  "best_canon_ids": [],
  "primary_canon_id": null,
  "confidence": 0.0,
  "is_mixed": false,
  "needs_review": false,
  "evidence_phrases": [],
  "reason": "",
  "match_type": "unknown"
}}
""".strip()

    try:
        b_resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": mapping_prompt},
            ],
        )
        mapping_text_raw = b_resp.choices[0].message.content or ""
        mapping_text_clean = clean_llm_text_for_json_parse(mapping_text_raw)
        parsed = parse_json_object(mapping_text_clean)
        if str(parsed.get("match_type") or "unknown") in {"exact", "probable", "mixed"} and not parsed.get(
            "best_canon_ids"
        ):
            parsed["match_type"] = "unknown"
            parsed["needs_review"] = True
            prev = str(parsed.get("reason") or "").strip()
            tail = "Empty best_canon_ids invalidated."
            parsed["reason"] = (prev + " " + tail).strip() if prev else tail
        return parsed
    except Exception as e:
        return {
            "best_canon_ids": [],
            "primary_canon_id": None,
            "confidence": 0.0,
            "is_mixed": False,
            "needs_review": True,
            "evidence_phrases": [],
            "reason": f"LLM Step B failed: {e}",
            "match_type": "unknown",
        }


def call_llm_two_step_ollama(
    *,
    host: str,
    model: str,
    teacher_id: str,
    unit_id: str,
    book: int,
    unit_text: str,
    candidates: Sequence[Dict[str, Any]],
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    """
    Ollama backend version of the strict 2-step mapping.
    """
    unit_text_sanitized = sanitize_unit_text(unit_text)
    if unit_text_sanitized:
        unit_text_sanitized = unit_text_sanitized[:MAX_UNIT_PROMPT_CHARS]
    candidate_block = build_candidate_block(candidates)

    anchors_prompt = f"""
{ULTRA_INSTRUCTION}

The teacher’s numbering is not canonical and must not be used as evidence.

You are analyzing a teacher commentary segment.

Task:
Extract the key doctrinal anchor phrases that would identify the underlying sutta.

Rules:
- Use only phrases actually supported by the text.
- Prefer distinctive doctrinal motifs.
- Return JSON only.

JSON schema:
{{
  "topic_summary": "",
  "anchor_phrases": [],
  "possible_motifs": []
}}

Teacher segment:
teacher_id: {teacher_id}
unit_id: {unit_id}
book: {book}

Teacher segment text:
{unit_text_sanitized}

Return JSON only. No markdown. No extra text.
""".strip()

    try:
        anchors_text = _ollama_chat(
            host=host,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": anchors_prompt},
            ],
            temperature=0.0,
            timeout_s=timeout_s,
        )
        anchors_json = parse_json_object(anchors_text)
    except Exception as e:
        return {
            "best_canon_ids": [],
            "primary_canon_id": None,
            "confidence": 0.0,
            "is_mixed": False,
            "needs_review": True,
            "evidence_phrases": [],
            "reason": f"LLM Step A failed: {e}",
            "match_type": "unknown",
        }

    allowed_candidate_ids = [
        str(c.get("canon_id") or "") for c in candidates if str(c.get("canon_id") or "").strip()
    ]

    mapping_prompt = f"""
You must choose from the provided candidate canon ids.

Rules:
- The teacher numbering is not canonical and must not be used as evidence.
- Use only the segment text and candidate entries.
- You must either:
  A) choose one or more canon ids from the candidate list, or
  B) return match_type="unknown" and needs_review=true.
- If match_type is "exact", "probable", or "mixed", then best_canon_ids must be non-empty.
- Every canon id in best_canon_ids must be copied exactly from the candidate list.
- Do not invent canon ids.
- Output JSON only.

Allowed candidate ids:
{json.dumps(allowed_candidate_ids, ensure_ascii=False)}

Segment:
{unit_text_sanitized}

Candidates:
{candidate_block}

Return exactly:
{{
  "best_canon_ids": [],
  "primary_canon_id": null,
  "confidence": 0.0,
  "is_mixed": false,
  "needs_review": false,
  "evidence_phrases": [],
  "reason": "",
  "match_type": "unknown"
}}
""".strip()

    debug_bundle: Dict[str, Any] = {
        "unit_id": unit_id,
        "teacher_id": teacher_id,
        "book": book,
        "raw_unit_text": unit_text,
        "candidate_block": candidate_block,
        "step_b_prompt": mapping_prompt,
    }

    try:
        mapping_text_raw = _ollama_chat(
            host=host,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": mapping_prompt},
            ],
            temperature=0.0,
            timeout_s=timeout_s,
        )
        mapping_text_clean = clean_llm_text_for_json_parse(mapping_text_raw)
        debug_bundle["step_b_response_raw"] = mapping_text_raw
        debug_bundle["step_b_response_clean"] = mapping_text_clean
        parsed = parse_json_object(mapping_text_clean)

        if str(parsed.get("match_type") or "unknown") in {"exact", "probable", "mixed"} and not parsed.get(
            "best_canon_ids"
        ):
            parsed["match_type"] = "unknown"
            parsed["needs_review"] = True
            prev = str(parsed.get("reason") or "").strip()
            tail = "Empty best_canon_ids invalidated."
            parsed["reason"] = (prev + " " + tail).strip() if prev else tail

        debug_bundle["step_b_parsed_json"] = parsed
        return {"__debug__": debug_bundle, **parsed}
    except Exception as e:
        debug_bundle["step_b_error"] = repr(e)
        return {
            "__debug__": debug_bundle,
            "best_canon_ids": [],
            "primary_canon_id": None,
            "confidence": 0.0,
            "is_mixed": False,
            "needs_review": True,
            "evidence_phrases": [],
            "reason": f"LLM Step B failed: {e}",
            "match_type": "unknown",
        }


# =========================
# VALIDATION + STATUS (HARD RULES)
# =========================


def _content_word_count(phrase: str) -> int:
    words = re.findall(r"[A-Za-z\u0080-\u024f']{3,}", (phrase or "").lower())
    words = [w for w in words if w not in STOPWORDS]
    return len(words)


def evidence_phrases_too_generic(evidence_phrases: Sequence[str]) -> bool:
    if not evidence_phrases:
        return True
    counts = [_content_word_count(p) for p in evidence_phrases if p]
    if not counts:
        return True
    # If the strongest evidence phrase is too short, treat as generic.
    return max(counts) < 2


def validate_and_assign_status(mapping: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    best = mapping.get("best_canon_ids") or []
    if not isinstance(best, list):
        best = []

    primary = mapping.get("primary_canon_id")

    conf_raw = mapping.get("confidence", 0.0)
    try:
        confidence = float(conf_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    is_mixed = bool(mapping.get("is_mixed", False))
    needs_review_llm = bool(mapping.get("needs_review", False))
    match_type = str(mapping.get("match_type", "unknown") or "unknown")
    if match_type not in {"exact", "probable", "mixed", "unknown"}:
        match_type = "unknown"

    evidence = mapping.get("evidence_phrases") or []
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(x) for x in evidence if x]

    evidence_generic = evidence_phrases_too_generic(evidence)

    # If the model provides multiple best ids but leaves primary null,
    # deterministically choose the first best id. This preserves "primary"
    # semantics while avoiding a purely-model-formatting omission.
    if primary is None and best:
        primary = best[0]

    # Force review if any "also force review" hard rules apply.
    force_review = False
    if primary is None:
        force_review = True
    if not best:
        force_review = True
    if match_type == "unknown":
        force_review = True
    if is_mixed is True:
        force_review = True
    if evidence_generic:
        force_review = True

    if force_review:
        needs_review = True
        status = "needs_review"
    else:
        # accept automatically rules
        if confidence >= 0.90 and not needs_review_llm:
            needs_review = False
            status = "accept"
        elif confidence >= 0.75 and match_type in {"exact", "probable"}:
            needs_review = True
            status = "accept_with_flag"
        else:
            needs_review = True
            status = "needs_review"

    out_fields = {
        "best_canon_ids": [str(x) for x in best],
        "primary_canon_id": primary if (primary is None or isinstance(primary, str)) else str(primary),
        "confidence": confidence,
        "is_mixed": is_mixed,
        "needs_review": needs_review,
        "evidence_phrases": evidence,
        "reason": str(mapping.get("reason") or ""),
        "match_type": match_type,
    }
    return out_fields, status


# =========================
# JSONL OUTPUT (RESUME)
# =========================


def load_processed_unit_ids(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    processed: set[str] = set()
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "unit_id" in obj:
                processed.add(str(obj["unit_id"]))
    return processed


def load_processed_unit_ids_from_pretty_json(pretty_json: Path) -> set[str]:
    """
    Resume support when only a pretty JSON array exists.
    """
    if not pretty_json.exists():
        return set()
    try:
        data = json.loads(pretty_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(data, list):
        return set()
    out: set[str] = set()
    for obj in data:
        if isinstance(obj, dict) and "unit_id" in obj:
            out.add(str(obj["unit_id"]))
    return out


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_debug_artifact(debug_dir: Path, unit_id: str, payload: Dict[str, Any]) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\-#]", "_", unit_id)
    out = debug_dir / f"{safe}.stepb_debug.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# =========================
# MAIN PIPELINE
# =========================


def iter_text_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.glob("*.txt")):
        if p.is_file():
            yield p


def process_file(
    *,
    file_path: Path,
    canon_by_book: Dict[int, List[Dict[str, Any]]],
    output_jsonl: Path,
    client: OpenAI | None,
    backend: str,
    ollama_host: str,
    ollama_model: str,
    llm_timeout_s: float,
    model: str,
    top_k: int,
    min_row_chars: int,
    max_units_per_row: int,
    processed_unit_ids: set[str],
    debug_stepb_one_unit: bool = False,
    debug_dir: Path = DEFAULT_DEBUG_DIR,
    stats: RunStats | None = None,
) -> int:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    teacher_rows = extract_teacher_rows_from_text(text)
    if not teacher_rows:
        return 0

    n_written = 0

    for row in teacher_rows:
        teacher_id = str(row["teacher_id"])
        book = int(row["book"])
        row_text = str(row["text"])

        units = split_units_for_row(
            teacher_id=teacher_id,
            book=book,
            row_text=row_text,
            min_row_chars=min_row_chars,
            max_units_per_row=max_units_per_row,
        )

        for unit in units:
            unit_id = str(unit["unit_id"])
            if (not debug_stepb_one_unit) and unit_id in processed_unit_ids:
                continue

            raw_unit_text = str(unit["text"])

            scored = retrieve_candidates_with_scores(
                unit_text=raw_unit_text,
                canon_by_book=canon_by_book,
                book=book,
                k=top_k,
            )
            candidates = [c for _, c in scored]

            if not candidates or ((not debug_stepb_one_unit) and len(candidates) < 5):
                mapping_json = {
                    "best_canon_ids": [],
                    "primary_canon_id": None,
                    "confidence": 0.0,
                    "is_mixed": False,
                    "needs_review": True,
                    "evidence_phrases": [],
                    "reason": "No sufficient candidates retrieved from canon.",
                    "match_type": "unknown",
                }
            else:
                llm_best_nonempty = False
                llm_returned_unknown = False
                fallback_used = False

                if backend == "openai":
                    mapping_json = call_llm_two_step(
                        client=client,
                        model=model,
                        teacher_id=teacher_id,
                        unit_id=unit_id,
                        book=book,
                        unit_text=raw_unit_text,
                        candidates=candidates[:top_k],
                    )
                elif backend == "ollama":
                    mapping_json = call_llm_two_step_ollama(
                        host=ollama_host,
                        model=ollama_model,
                        teacher_id=teacher_id,
                        unit_id=unit_id,
                        book=book,
                        unit_text=raw_unit_text,
                        candidates=candidates[:top_k],
                        timeout_s=llm_timeout_s,
                    )
                else:
                    raise SystemExit(f"Unknown backend: {backend}")

            # For the "no candidates" branch, ensure these flags exist.
            if not candidates or ((not debug_stepb_one_unit) and len(candidates) < 5):
                llm_best_nonempty = False
                llm_returned_unknown = True
                fallback_used = False

            # Heuristic fallback (only when model refuses to choose):
            # If Step B returns unknown/empty but retrieval has a clearly best candidate,
            # pick it as "probable" so the pipeline can progress.
            try:
                mt = str(mapping_json.get("match_type") or "unknown")
            except Exception:
                mt = "unknown"
            best_list = mapping_json.get("best_canon_ids") if isinstance(mapping_json, dict) else None
            best_empty = (not best_list) if isinstance(best_list, list) else True

            if isinstance(mapping_json, dict):
                llm_best_nonempty = bool(mapping_json.get("best_canon_ids"))
                llm_returned_unknown = (str(mapping_json.get("match_type") or "unknown") == "unknown")

            if mt == "unknown" and best_empty and scored:
                top_score, top_cand = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else 0.0
                gap = float(top_score) - float(second_score)
                top_cid = str(top_cand.get("canon_id") or "")

                if top_cid and float(top_score) >= FALLBACK_TOP_SCORE_MIN and gap >= FALLBACK_GAP_MIN:
                    fallback_used = True
                    mapping_json = {
                        **(mapping_json if isinstance(mapping_json, dict) else {}),
                        "best_canon_ids": [top_cid],
                        "primary_canon_id": top_cid,
                        "confidence": round(min(0.85, 0.60 + float(top_score)), 3),
                        "is_mixed": False,
                        "needs_review": False,
                        "evidence_phrases": [],
                        "reason": f"Heuristic fallback: top candidate by retrieval (score={float(top_score):.3f}, gap={gap:.3f}).",
                        "match_type": "probable",
                    }

            # Debug: save the exact Step B prompt/response for the first encountered unit.
            if debug_stepb_one_unit:
                dbg = {
                    "unit_id": unit_id,
                    "teacher_id": teacher_id,
                    "book": book,
                    "raw_unit_text": raw_unit_text,
                    "retrieval_top": [
                        {"score": float(s), "canon_id": str(c.get("canon_id") or ""), "title": str(c.get("title") or "")}
                        for s, c in scored
                    ],
                    "mapping_json_raw": mapping_json,
                }
                out_path = write_debug_artifact(debug_dir, unit_id, dbg)
                print(f"[debug] wrote {out_path}")
                return 0

            validated_fields, status = validate_and_assign_status(mapping_json)

            # ---- stats accumulation (post-validate) ----
            if stats is not None:
                stats.total_units += 1
                if llm_best_nonempty:
                    stats.llm_chose_nonempty += 1
                if llm_returned_unknown:
                    stats.llm_returned_unknown += 1
                if fallback_used:
                    stats.fallback_used += 1

                stats.book_totals[book] += 1
                if status == "accept":
                    stats.accepted += 1
                elif status == "accept_with_flag":
                    stats.accept_with_flag += 1
                else:
                    stats.needs_review += 1
                    stats.book_needs_review[book] += 1

                pcid = validated_fields.get("primary_canon_id")
                if isinstance(pcid, str) and pcid.strip():
                    stats.primary_freq[pcid.strip()] += 1

                if scored:
                    top_score = float(scored[0][0])
                    gap = float(scored[0][0]) - float(scored[1][0]) if len(scored) > 1 else 1.0
                    if top_score < FALLBACK_TOP_SCORE_MIN:
                        stats.add_low_score(unit_id)
                    if gap <= SMALL_GAP_MAX:
                        stats.add_small_gap(unit_id)

            out_obj: Dict[str, Any] = {
                "teacher_id": teacher_id,
                "unit_id": unit_id,
                "book": book,
                "raw_unit_text": raw_unit_text,
                "best_canon_ids": validated_fields["best_canon_ids"],
                "primary_canon_id": validated_fields["primary_canon_id"],
                "confidence": validated_fields["confidence"],
                "is_mixed": validated_fields["is_mixed"],
                "needs_review": validated_fields["needs_review"],
                "evidence_phrases": validated_fields["evidence_phrases"],
                "reason": validated_fields["reason"],
                "match_type": validated_fields["match_type"],
                "status": status,
            }

            append_jsonl(output_jsonl, out_obj)
            processed_unit_ids.add(unit_id)
            n_written += 1

    return n_written


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic AN canon mapper (semantic units -> constrained LLM).")
    ap.add_argument("--input-dir", type=Path, default=Path("sample_transciprt"))
    ap.add_argument("--canon-file", type=Path, default=Path("canon_sutta.txt"))
    ap.add_argument("--output-jsonl", type=Path, default=Path("output/semantic_mapping.jsonl"))
    ap.add_argument("--top-k", type=int, default=TOP_K_DEFAULT, help="Retrieve top candidates (must be 5..10).")
    ap.add_argument("--min-row-chars", type=int, default=MIN_ROW_CHARS)
    ap.add_argument("--max-units-per-row", type=int, default=MAX_UNITS_PER_ROW)
    ap.add_argument("--backend", type=str, choices=["openai", "ollama"], default="ollama")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name (when --backend=openai).")
    ap.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", type=str, default="qwen2.5:0.5b-instruct")
    ap.add_argument("--llm-timeout-s", type=float, default=180.0)
    ap.add_argument("--debug-stepb-one-unit", action="store_true", help="Write Step B debug artifact for first unit then exit.")
    ap.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    ap.add_argument(
        "--keep-jsonl",
        action="store_true",
        help="Keep the .jsonl file after writing the pretty JSON companion (default deletes JSONL).",
    )
    args = ap.parse_args()

    if args.debug_stepb_one_unit:
        if args.top_k < 1 or args.top_k > 10:
            raise SystemExit("--top-k must be between 1 and 10 in debug mode.")
    else:
        if args.top_k < 5 or args.top_k > 10:
            raise SystemExit("--top-k must be between 5 and 10 (inclusive) per spec.")

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {args.input_dir}")
    if not args.canon_file.is_file():
        raise SystemExit(f"Canon file not found: {args.canon_file}")

    canon_by_book = load_canon_corpus(args.canon_file)
    if not canon_by_book:
        raise SystemExit("Loaded 0 canon entries; check canon_sutta.txt format.")

    output_jsonl = args.output_jsonl.resolve()
    # If we are in pretty-only mode and JSONL doesn't exist, resume from the pretty JSON.
    pretty_path_for_resume = output_jsonl.with_suffix("")
    pretty_path_for_resume = pretty_path_for_resume.with_name(pretty_path_for_resume.name + ".pretty").with_suffix(".json")
    processed_unit_ids = load_processed_unit_ids(output_jsonl) | load_processed_unit_ids_from_pretty_json(pretty_path_for_resume)

    client: OpenAI | None = None
    if args.backend == "openai":
        client = OpenAI(api_key=load_api_key())

    total = 0
    n_files = 0
    run_stats = RunStats()
    for f in iter_text_files(args.input_dir):
        n_files += 1
        n = process_file(
            file_path=f,
            canon_by_book=canon_by_book,
            output_jsonl=output_jsonl,
            client=client,
            backend=args.backend,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
            llm_timeout_s=args.llm_timeout_s,
            model=args.model,
            top_k=args.top_k,
            min_row_chars=args.min_row_chars,
            max_units_per_row=args.max_units_per_row,
            processed_unit_ids=processed_unit_ids,
            debug_stepb_one_unit=args.debug_stepb_one_unit,
            debug_dir=args.debug_dir,
            stats=run_stats,
        )
        total += n
        print(f"{f.name}: wrote {n} unit(s)")

    print(f"Done. Files={n_files}, new_units_written={total}, output={output_jsonl}")

    # Write wrapped/pretty JSON array for easy viewing.
    # By default, delete the JSONL afterwards so there are no duplicates.
    try:
        base = output_jsonl
        pretty_path = base.with_suffix("")  # drop .jsonl
        pretty_path = pretty_path.with_name(pretty_path.name + ".pretty").with_suffix(".json")
        n_pretty = _jsonl_to_pretty_json(output_jsonl, pretty_path)
        print(f"Wrote pretty JSON -> {pretty_path} ({n_pretty} records)")
        if (not args.keep_jsonl) and output_jsonl.exists():
            output_jsonl.unlink(missing_ok=True)
            print(f"Deleted JSONL (pretty-only mode) -> {output_jsonl}")
    except Exception as e:
        print(f"WARN: Failed to write pretty JSON companion: {e}")

    # ---- Summary counts requested ----
    print("\n=== Summary counts ===")
    print(f"total_units: {run_stats.total_units}")
    print(f"llm_chose_nonempty: {run_stats.llm_chose_nonempty}")
    print(f"llm_returned_unknown: {run_stats.llm_returned_unknown}")
    print(f"fallback_used: {run_stats.fallback_used}")
    print(f"accepted: {run_stats.accepted}")
    print(f"accept_with_flag: {run_stats.accept_with_flag}")
    print(f"needs_review: {run_stats.needs_review}")

    print("\n=== primary_canon_id frequency (top 15) ===")
    for cid, cnt in run_stats.primary_freq.most_common(15):
        print(f"{cid}\t{cnt}")

    print("\n=== top books by review rate (top 10) ===")
    book_rates: List[Tuple[float, int, int, int]] = []
    for b, tot in run_stats.book_totals.items():
        nr = run_stats.book_needs_review.get(b, 0)
        rate = (nr / tot) if tot else 0.0
        book_rates.append((rate, b, nr, tot))
    for rate, b, nr, tot in sorted(book_rates, key=lambda x: (-x[0], x[1]))[:10]:
        print(f"book={b}\treview_rate={rate:.3f}\tneeds_review={nr}\ttotal={tot}")

    print("\n=== retrieval weakness ===")
    print(f'units_where_top_candidate_score < {FALLBACK_TOP_SCORE_MIN}: {run_stats.low_top_score_units}')
    if run_stats.sample_low_score_unit_ids:
        print("sample_low_score_unit_ids:\t" + ", ".join(run_stats.sample_low_score_unit_ids))
    print(f'units_where_score_gap_is_small (<= {SMALL_GAP_MAX}): {run_stats.small_gap_units}')
    if run_stats.sample_small_gap_unit_ids:
        print("sample_small_gap_unit_ids:\t" + ", ".join(run_stats.sample_small_gap_unit_ids))


if __name__ == "__main__":
    main()

