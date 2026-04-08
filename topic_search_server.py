"""
DAMA — Anguttara Nikāya (AN books 4–11) topic UI + LLM chat. FastAPI + Uvicorn.
Run: uvicorn topic_search_server:app --host 127.0.0.1 --port 8020

Data: all processed transcript/an*.json files merged (numeric book order).

LLM priority (first match wins):
  1) DAMA_LLM_BASE_URL — explicit Ollama/custom OpenAI-compatible base
  2) Local Ollama at http://127.0.0.1:11434/v1 — unless DAMA_USE_OLLAMA=0
  3) Groq: GROQ_API_KEY or creds/groqkey.txt
  4) OpenAI: OPENAI_API_KEY or creds/openaikey.txt (429 → Groq fallback if configured)

Topic search uses only local JSON — no API keys. Ask/chat needs local Ollama (free) or Groq/OpenAI on the host.

Models: DAMA_CHAT_MODEL (Ollama default qwen2.5:0.5b-instruct), DAMA_GROQ_MODEL, OpenAI default gpt-4o-mini.
For reliable context-only answers, prefer a larger local model (e.g. 7B+) via DAMA_CHAT_MODEL, or Groq/OpenAI.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent

# Dataset resolution
_DATA_DIR_DEFAULT = ROOT / "processed transcript"
_DATA_DIR = Path(os.environ.get("DAMA_DATA_LOCAL_DIR", "").strip() or _DATA_DIR_DEFAULT)
_DAMA_DATA_GCS_URI = os.environ.get("DAMA_DATA_GCS_URI", "").strip()
_DATA_CACHE_DIR = Path(os.environ.get("DAMA_DATA_CACHE_DIR", "").strip() or "/tmp/dama_data")


def _an_json_paths(data_dir: Path) -> List[Path]:
    paths = list(data_dir.glob("an*.json"))

    def sort_key(p: Path) -> tuple[int, str]:
        m = re.match(r"^an(\d+)$", p.stem, re.IGNORECASE)
        if m:
            return (int(m.group(1)), p.name.lower())
        return (999, p.name.lower())

    return sorted(paths, key=sort_key)
_CREDS_OPENAI_PATH = ROOT / "creds" / "openaikey.txt"
_CREDS_GROQ_PATH = ROOT / "creds" / "groqkey.txt"
TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))

_OLLAMA_LOCAL_URL = "http://127.0.0.1:11434/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
_OLLAMA_DEFAULT_MODEL = "qwen2.5:0.5b-instruct"

WEIGHT_CHAIN = 3
WEIGHT_COMMENTARY = 1
WEIGHT_SUTTA = 1
MAX_COLUMNS_DEFAULT = 3
MAX_CONTEXT_CHARS = 12000

# Structure / phrase bonuses (additive) so category and chain items beat scattered commentary hits
_BONUS_CATEGORY_EQ = 130.0
_BONUS_CATEGORY_PHRASE = 70.0
_BONUS_QUERY_IN_CHAIN_BLOB = 85.0
_BONUS_ITEM_EQ = 110.0
_BONUS_ITEM_PHRASE = 50.0
_BONUS_SUTTA_PHRASE_LONG = 95.0
_BONUS_SUTTA_PHRASE_MED = 72.0
_BONUS_SUTTA_LEAD_IN = 320.0
_SUTTA_LEAD_IN_CHARS = 260
_BONUS_COMM_PHRASE = 38.0
_NO_CHAIN_SCORE_FACTOR = 0.82
# Scale extracted-chain matches so exact category/items beat long generic commentary hits.
_CHAIN_ANCHOR_SCALE = 1000.0

MIN_CHAT_REPLY_CHARS = 42
_CHAT_FALLBACK_REPLY = (
    "I could not form an answer from the excerpts shown. "
    "Try another result column (i), rephrase your question, or ask about wording that appears in the sutta or commentary."
)

_DEFAULT_CHAT_MODEL = "gpt-4o-mini"

_SUGGESTIONS_MARKER = "SUGGESTIONS_JSON:"
_MAX_SUGGESTIONS = 5
_MAX_SUGGESTION_CHARS = 320


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _load_json_array(path: Path) -> List[dict[str, Any]]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _load_all_rows() -> List[dict[str, Any]]:
    """All an<N>.json in configured data dir, book order; skip missing/unreadable."""
    rows: List[dict[str, Any]] = []
    for path in _an_json_paths(_DATA_DIR):
        rows.extend(_load_json_array(path))
    return rows


_ROWS_CACHE: Optional[List[dict[str, Any]]] = None


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """
    Parse gs://bucket/prefix[/] into (bucket, prefix_without_leading_slash).
    """
    u = uri.strip()
    if not u.startswith("gs://"):
        raise ValueError("DAMA_DATA_GCS_URI must start with gs://")
    p = urlparse(u)
    bucket = p.netloc
    prefix = (p.path or "").lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def _maybe_fetch_gcs_dataset() -> None:
    """
    If DAMA_DATA_GCS_URI is set, download an*.json to a local cache dir and switch _DATA_DIR to it.
    Safe to call multiple times.
    """
    global _DATA_DIR
    if not _DAMA_DATA_GCS_URI:
        return
    try:
        from google.cloud import storage  # type: ignore
    except Exception:
        # Running without google-cloud-storage installed; fall back to local dir.
        return
    bucket_name, prefix = _parse_gcs_uri(_DAMA_DATA_GCS_URI)
    local_dir = _DATA_CACHE_DIR / "processed_transcript"
    local_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    for blob in blobs:
        name = blob.name or ""
        base = name.split("/")[-1]
        if not re.match(r"^an\d+\.json$", base, flags=re.IGNORECASE):
            continue
        dest = local_dir / base
        blob.download_to_filename(str(dest))
    _DATA_DIR = local_dir


def _rows() -> List[dict[str, Any]]:
    global _ROWS_CACHE
    if _ROWS_CACHE is None:
        _maybe_fetch_gcs_dataset()
        _ROWS_CACHE = _load_all_rows()
    return _ROWS_CACHE


def _has_chain(row: dict[str, Any]) -> bool:
    c = row.get("chain")
    if not isinstance(c, dict):
        return False
    items = c.get("items")
    return isinstance(items, list) and len(items) > 0


def _book_from_sutta_id(sid: Any) -> Optional[int]:
    if sid is None:
        return None
    m = re.match(r"^(\d+)\.", str(sid).strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _column_title(chain: dict[str, Any]) -> str:
    cat = chain.get("category")
    if cat:
        return str(cat).replace("_", " ").strip()
    items = chain.get("items")
    if isinstance(items, list) and items:
        s = str(items[0]).strip()
        return s[:80] + ("…" if len(s) > 80 else "")
    return "Chain"


def _tokens(q: str) -> List[str]:
    parts = re.split(r"\s+", (q or "").strip())
    return [p for p in parts if len(p) >= 2]


def _count_occurrences(hay: str, tok: str) -> int:
    if not tok or not hay:
        return 0
    return len(re.findall(re.escape(tok), hay, flags=re.IGNORECASE))


def _hyphen_loose(norm: str) -> str:
    """Map hyphenated phrases to space form for user queries without hyphens."""
    return norm.replace("-", " ").strip()


def _category_equals_query(row: dict[str, Any], q_norm: str) -> bool:
    if not q_norm:
        return False
    chain = row.get("chain")
    if not isinstance(chain, dict):
        return False
    cat = chain.get("category")
    if not cat:
        return False
    cat_norm = _normalize(str(cat))
    return cat_norm == q_norm or _hyphen_loose(cat_norm) == _hyphen_loose(q_norm)


def _chain_phrase_bonus(row: dict[str, Any], q_norm: str) -> float:
    """Bonuses from chain category/items/blob (rank above loose commentary token overlap)."""
    if not q_norm or len(q_norm) < 2:
        return 0.0
    q_loose = _hyphen_loose(q_norm)
    bonus = 0.0
    chain = row.get("chain")
    cat_norm = ""
    items_joined = ""
    if isinstance(chain, dict):
        cat = chain.get("category")
        if cat:
            cat_norm = _normalize(str(cat))
        items = chain.get("items")
        if isinstance(items, list):
            for it in items:
                items_joined += " " + _normalize(str(it))
    chain_blob = _normalize(f"{cat_norm} {items_joined}")
    chain_loose = _hyphen_loose(chain_blob)

    if cat_norm:
        cat_loose = _hyphen_loose(cat_norm)
        if cat_norm == q_norm or cat_loose == q_loose:
            bonus += _BONUS_CATEGORY_EQ
        elif len(q_norm) >= 5 and (
            q_norm in cat_norm
            or cat_norm in q_norm
            or q_loose in cat_loose
            or cat_loose in q_loose
        ):
            bonus += _BONUS_CATEGORY_PHRASE

    if len(q_norm) >= 6 and chain_blob and (
        q_norm in chain_blob or q_loose in chain_loose
    ):
        bonus += _BONUS_QUERY_IN_CHAIN_BLOB

    if isinstance(chain, dict):
        items = chain.get("items")
        if isinstance(items, list):
            for it in items:
                inn = _normalize(str(it))
                if not inn:
                    continue
                inl = _hyphen_loose(inn)
                if inn == q_norm or inl == q_loose:
                    bonus += _BONUS_ITEM_EQ
                elif len(q_norm) >= 5 and (q_norm in inn or q_loose in inl):
                    bonus += _BONUS_ITEM_PHRASE
                elif len(inn) >= 8 and (inn in q_norm or inl in q_loose):
                    bonus += _BONUS_ITEM_PHRASE * 0.75

    return bonus


def _sutta_lead_has_query_phrase(row: dict[str, Any], q_norm: str) -> bool:
    """True if normalized query (multi-word) appears in the opening of the reading text."""
    if len(q_norm) < 12:
        return False
    raw_sut = str(row.get("sutta") or "")
    lead = _normalize(raw_sut[:_SUTTA_LEAD_IN_CHARS])
    q_loose = _hyphen_loose(q_norm)
    leadl = _hyphen_loose(lead)
    return q_norm in lead or q_loose in leadl


def _text_phrase_bonus(row: dict[str, Any], q_norm: str) -> float:
    """Substring phrase hits in sutta/commentary (secondary to chain structure)."""
    if not q_norm or len(q_norm) < 2:
        return 0.0
    q_loose = _hyphen_loose(q_norm)
    raw_sut = str(row.get("sutta") or "")
    sut = _normalize(raw_sut)
    comm = _normalize(str(row.get("commentary") or ""))
    sutl = _hyphen_loose(sut)
    coml = _hyphen_loose(comm)
    bonus = 0.0
    if _sutta_lead_has_query_phrase(row, q_norm):
        bonus += _BONUS_SUTTA_LEAD_IN
    elif len(q_norm) >= 10 and (q_norm in sut or q_loose in sutl):
        bonus += _BONUS_SUTTA_PHRASE_LONG
    elif len(q_norm) >= 6 and (q_norm in sut or q_loose in sutl):
        bonus += _BONUS_SUTTA_PHRASE_MED
    if len(q_norm) >= 10 and (q_norm in comm or q_loose in coml):
        bonus += _BONUS_COMM_PHRASE
    return bonus


def score_row(row: dict[str, Any], toks: List[str], query: str) -> float:
    chain = row.get("chain") or {}
    items = chain.get("items") if isinstance(chain, dict) else None
    chain_blob = _normalize(" ".join(items) if isinstance(items, list) else "")
    comm = _normalize(str(row.get("commentary") or ""))
    sut = _normalize(str(row.get("sutta") or ""))
    token_total = 0.0
    for t in toks:
        nt = _normalize(t)
        if len(nt) < 2:
            continue
        token_total += WEIGHT_CHAIN * _count_occurrences(chain_blob, nt)
        token_total += WEIGHT_COMMENTARY * _count_occurrences(comm, nt)
        token_total += WEIGHT_SUTTA * _count_occurrences(sut, nt)
    q_norm = _normalize(query)
    chain_b = _chain_phrase_bonus(row, q_norm)
    text_b = _text_phrase_bonus(row, q_norm)
    lex = text_b + token_total
    if not _has_chain(row) and lex > 0:
        if not _sutta_lead_has_query_phrase(row, q_norm):
            lex *= _NO_CHAIN_SCORE_FACTOR
    return chain_b * _CHAIN_ANCHOR_SCALE + lex


def build_bot_summary(query: str, top_rows: List[dict[str, Any]]) -> str:
    q_disp = (query or "").strip() or "(topic)"
    categories: List[str] = []
    item_tags: List[str] = []
    for row in top_rows:
        c = row.get("chain")
        if not isinstance(c, dict):
            continue
        cat = c.get("category")
        if cat and str(cat) not in categories:
            categories.append(str(cat))
        items = c.get("items")
        if isinstance(items, list):
            for it in items:
                s = str(it).strip()
                if s and s.lower() not in [x.lower() for x in item_tags]:
                    item_tags.append(s)
                if len(item_tags) >= 10:
                    break
        if len(categories) >= 5:
            break
    if categories:
        tail = ", ".join(categories[:6])
    elif item_tags:
        tail = ", ".join(item_tags[:8])
    else:
        tail = "no chain tags"
    return f"BOT: {q_disp} ({tail})"


def row_to_hit(row: dict[str, Any], _score: float) -> dict[str, Any]:
    chain = row.get("chain")
    if chain is not None and not isinstance(chain, dict):
        chain = None
    title = _column_title(chain) if isinstance(chain, dict) else "Chain"
    sid = row.get("sutta_id")
    return {
        "sutta_id": sid,
        "title": title,
        "book": _book_from_sutta_id(sid),
        "chain": chain,
        "sutta": str(row.get("sutta") or ""),
        "commentary": str(row.get("commentary") or ""),
    }


def _sutta_id_key(row: dict[str, Any]) -> str:
    return str(row.get("sutta_id") or "")


def _pick_top_scored_hits(
    scored: List[Tuple[float, dict[str, Any]]],
    *,
    max_columns: int,
) -> List[Tuple[float, dict[str, Any]]]:
    """
    Take top unique sutta_id rows by score (includes segments without chains so
    sutta-only matches e.g. transference of merit can surface).
    """
    chosen: List[Tuple[float, dict[str, Any]]] = []
    chosen_ids: set[str] = set()
    for s, row in scored:
        if len(chosen) >= max_columns:
            break
        sid = _sutta_id_key(row)
        if sid in chosen_ids:
            continue
        chosen.append((s, row))
        chosen_ids.add(sid)
    return chosen


def run_search(
    q: str,
    *,
    max_columns: int = MAX_COLUMNS_DEFAULT,
) -> dict[str, Any]:
    toks = _tokens(q)
    bot_summary = f"BOT: {(q or '').strip() or '(topic)'} (enter a topic)"
    if not toks:
        return {
            "query": q,
            "tokens": toks,
            "bot_summary": bot_summary,
            "top": [],
        }
    scored: List[Tuple[float, dict[str, Any]]] = []
    for row in _rows():
        s = score_row(row, toks, q)
        if s > 0:
            scored.append((s, row))
    qn = _normalize(q)
    scored.sort(
        key=lambda x: (
            0 if _category_equals_query(x[1], qn) else 1,
            -x[0],
        )
    )

    hits_rows = _pick_top_scored_hits(scored, max_columns=max_columns)

    if not scored:
        bot_summary = f"BOT: {(q or '').strip()} (no matches)"
    elif not hits_rows:
        bot_summary = f"BOT: {(q or '').strip()} (no matches)"
    else:
        bot_summary = build_bot_summary(q, [r for _, r in hits_rows])

    hits = [row_to_hit(row, s) for s, row in hits_rows]
    return {
        "query": q,
        "tokens": toks,
        "bot_summary": bot_summary,
        "top": hits,
    }


def _weak_chat_reply(reply: str) -> bool:
    return len((reply or "").strip()) < MIN_CHAT_REPLY_CHARS


def _split_reply_and_suggestions(raw: str) -> tuple[str, list[str]]:
    """Parse trailing SUGGESTIONS_JSON: [...]; return visible reply and suggestion strings."""
    raw_st = (raw or "").strip()
    if _SUGGESTIONS_MARKER not in raw_st:
        return raw_st, []
    idx = raw_st.rfind(_SUGGESTIONS_MARKER)
    reply = raw_st[:idx].rstrip()
    tail = raw_st[idx + len(_SUGGESTIONS_MARKER) :].strip()
    if not tail:
        return reply, []
    try:
        parsed = json.loads(tail)
    except json.JSONDecodeError:
        # If the model didn't follow the contract (marker present but no valid JSON),
        # keep the original text so we don't silently drop visible content.
        return raw_st, []
    if not isinstance(parsed, list):
        return raw_st, []
    out: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            s = item.strip()
            if s and len(s) <= _MAX_SUGGESTION_CHARS:
                out.append(s)
        if len(out) >= _MAX_SUGGESTIONS:
            break
    return reply, out


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatContext(BaseModel):
    sutta_id: Optional[str] = None
    sutta: str = ""
    commentary: str = ""


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    context: Optional[ChatContext] = None
    top_context: Optional[List[ChatContext]] = None


class SearchStartersRequest(BaseModel):
    query: str = ""
    sutta_ids: List[str] = Field(default_factory=list)
    prior_user_messages: List[str] = Field(default_factory=list)


def _parse_starters_json(raw: str) -> list[str]:
    t = (raw or "").strip()
    if not t:
        return []
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []
    qs = obj.get("questions")
    if not isinstance(qs, list):
        return []
    out: list[str] = []
    for item in qs:
        if isinstance(item, str):
            s = item.strip()
            if s and len(s) <= _MAX_SUGGESTION_CHARS:
                out.append(s)
        if len(out) >= _MAX_SUGGESTIONS:
            break
    return out


def _search_starters_user_payload(req: SearchStartersRequest) -> str:
    lines = [
        f"Search query: {(req.query or '').strip() or '(empty)'}",
        f"Top sutta ids shown: {', '.join(str(x).strip() for x in req.sutta_ids[:8] if str(x).strip()) or '(none)'}",
    ]
    prior = [p.strip() for p in req.prior_user_messages if isinstance(p, str) and p.strip()][-5:]
    if prior:
        lines.append("Earlier user messages (for continuity):")
        for i, p in enumerate(prior, 1):
            lines.append(f"  {i}. {p[:500]}")
    return "\n".join(lines)


def _search_starters_system_prompt() -> str:
    return (
        "You help users explore Anguttara Nikāya teacher commentary material after a topic search.\n"
        "Output JSON only, one object, exactly this shape:\n"
        '{"questions":["short question 1","short question 2","short question 3"]}\n'
        "Use 3 or 4 questions. Each string is a natural follow-up the user might ask about their "
        "search query and the listed sutta ids; use prior user messages for continuity when provided.\n"
        "No markdown fences, no extra keys, no prose outside the JSON."
    )


def _chat_system_prompt(
    context: Optional[ChatContext],
    top_context: Optional[List[ChatContext]],
    *,
    strict_retry: bool = False,
) -> str:
    blocks: List[str] = []
    if strict_retry:
        blocks.append(
            "RETRY — Your previous answer was empty or too short. REQUIRED: write at least three sentences "
            "that only paraphrase or quote the excerpts below (name sutta_id). No filler. "
            "Then one short question. Then the FINAL line only must be the SUGGESTIONS_JSON line as specified below."
        )
    blocks.extend(
        [
        "You are DAMA, a helpful assistant for Anguttara Nikāya study materials (Theravāda teacher reading + commentary excerpts; books 4–11 in this app).",
        "Answer using ONLY the provided sutta reading and teacher commentary excerpts. Do not use outside knowledge.",
        "Do not invent etymologies, Sanskrit, or facts not supported by those excerpts. Do not bring in Zen, Mahāyāna, or other schools unless the excerpt itself does.",
        "If the excerpts do not support an answer, say briefly that you do not find it in this material.",
        "Short follow-up questions from the user (e.g. \"can anyone do it?\") refer to the previous topic in the conversation—answer in that Dhamma thread, not about being an AI.",
        "Never reply with \"As an AI language model\" or similar meta-refusals; stay in character as DAMA for this study app.",
        "Cite sutta_id when relevant. Be concise.",
        "End your visible answer (before the final line below) with one short engaging question for the user, still grounded in the excerpts or their question.",
        "After that, on the FINAL line only (no text after it), output exactly:",
        f'{_SUGGESTIONS_MARKER} ["short question 1", "short question 2", "short question 3"]',
        "Use 2–4 strings in the JSON array, grounded in the material and conversation. No markdown, no extra keys.",
        ]
    )
    if context and (context.sutta or context.commentary):
        sid = context.sutta_id or "?"
        blocks.append(
            f"--- Primary segment {sid} ---\nSUTTA (reading):\n{_truncate(context.sutta, MAX_CONTEXT_CHARS)}\n\nCOMMENTARY:\n{_truncate(context.commentary, MAX_CONTEXT_CHARS)}"
        )
    if top_context:
        for i, tc in enumerate(top_context[:3], 1):
            if not (tc.sutta or tc.commentary):
                continue
            sid = tc.sutta_id or "?"
            blocks.append(
                f"--- Related segment {i} ({sid}) ---\nSUTTA:\n{_truncate(tc.sutta, MAX_CONTEXT_CHARS // 2)}\n\nCOMMENTARY:\n{_truncate(tc.commentary, MAX_CONTEXT_CHARS // 2)}"
            )
    return "\n\n".join(blocks)


def _read_first_line_cred(path: Path) -> str:
    if not path.is_file():
        return ""
    for part in path.read_text(encoding="utf-8", errors="replace").splitlines():
        k = part.strip()
        if k and not k.startswith("#"):
            return k
    return ""


def _openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip() or _read_first_line_cred(
        _CREDS_OPENAI_PATH
    )


def _groq_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "").strip() or _read_first_line_cred(
        _CREDS_GROQ_PATH
    )


def _normalize_openai_base(url: str) -> str:
    u = url.rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1" if "/v1" not in u else u
    return u


def _resolve_llm() -> tuple[Optional[OpenAI], str, str]:
    """
    Returns (client, model_name, provider) where provider is
    ollama | groq | openai | none.
    """
    base = os.environ.get("DAMA_LLM_BASE_URL", "").strip()
    if base:
        key = os.environ.get("DAMA_LLM_API_KEY", "").strip() or "ollama"
        model = (
            os.environ.get("DAMA_CHAT_MODEL", "").strip() or _OLLAMA_DEFAULT_MODEL
        )
        return OpenAI(base_url=_normalize_openai_base(base), api_key=key), model, "ollama"

    if os.environ.get("DAMA_USE_OLLAMA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        model = (
            os.environ.get("DAMA_CHAT_MODEL", "").strip() or _OLLAMA_DEFAULT_MODEL
        )
        return (
            OpenAI(base_url=_OLLAMA_LOCAL_URL, api_key="ollama"),
            model,
            "ollama",
        )

    gq = _groq_api_key()
    if gq:
        model = (
            os.environ.get("DAMA_GROQ_MODEL", "").strip()
            or os.environ.get("DAMA_CHAT_MODEL", "").strip()
            or _DEFAULT_GROQ_MODEL
        )
        return OpenAI(base_url=GROQ_BASE_URL, api_key=gq), model, "groq"

    oa = _openai_api_key()
    if oa:
        model = (
            os.environ.get("DAMA_CHAT_MODEL", "").strip() or _DEFAULT_CHAT_MODEL
        )
        return OpenAI(api_key=oa), model, "openai"

    return None, "", "none"


def _ollama_reachable() -> bool:
    """True if something responds on default Ollama port (avoids claiming chat works when Ollama is off)."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/tags", timeout=0.75
        ) as r:
            return 200 <= getattr(r, "status", 200) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _groq_client_and_model() -> tuple[Optional[OpenAI], str]:
    gq = _groq_api_key()
    if not gq:
        return None, ""
    model = (
        os.environ.get("DAMA_GROQ_MODEL", "").strip()
        or _DEFAULT_GROQ_MODEL
    )
    return OpenAI(base_url=GROQ_BASE_URL, api_key=gq), model


app = FastAPI(title="DAMA — AN 4–11")


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    client, model, provider = _resolve_llm()
    chat_ok = client is not None
    chat_disabled = os.environ.get("DAMA_DISABLE_CHAT", "").strip() in ("1", "true", "yes")
    if chat_disabled:
        chat_ok = False
    custom_base = os.environ.get("DAMA_LLM_BASE_URL", "").strip()
    if chat_ok and provider == "ollama" and not custom_base:
        chat_ok = _ollama_reachable()
    if os.environ.get("DAMA_USE_VERTEX", "").strip() in ("1", "true", "yes"):
        # Vertex chat does not use the OpenAI-compatible client.
        chat_ok = False if chat_disabled else True
        provider = "vertex"
        model = os.environ.get("DAMA_VERTEX_MODEL", "").strip() or "gemini-2.5-flash"
    return {
        "chat_available": chat_ok,
        "provider": provider,
        "model": model,
    }


@app.get("/api/search")
def api_search(
    q: str = Query("", description="Topic keywords"),
    max_columns: int = Query(MAX_COLUMNS_DEFAULT, ge=1, le=10),
) -> JSONResponse:
    data = run_search(q, max_columns=max_columns)
    return JSONResponse(content=data)


@app.post("/api/search_starters")
def api_search_starters(req: SearchStartersRequest) -> JSONResponse:
    client, model, provider = _resolve_llm()
    if client is None:
        return JSONResponse(content={"questions": []})
    custom_base = os.environ.get("DAMA_LLM_BASE_URL", "").strip()
    if provider == "ollama" and not custom_base and not _ollama_reachable():
        return JSONResponse(content={"questions": []})
    system = _search_starters_system_prompt()
    user = _search_starters_user_payload(req)
    oa_messages: List[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    used_model = model
    try:
        out = _chat_completion(client, model, oa_messages, temperature=0.25)
    except Exception as e:
        err_s = f"{e!s}"
        quota_like = (
            "429" in err_s
            or "insufficient_quota" in err_s
            or "rate_limit" in err_s.lower()
        )
        if provider == "openai" and quota_like:
            gcli, gmodel = _groq_client_and_model()
            if gcli is not None:
                try:
                    out = _chat_completion(gcli, gmodel, oa_messages, temperature=0.25)
                    used_model = gmodel
                except Exception:
                    return JSONResponse(content={"questions": []})
            else:
                return JSONResponse(content={"questions": []})
        else:
            return JSONResponse(content={"questions": []})
    text = (out.choices[0].message.content or "").strip()
    questions = _parse_starters_json(text)
    return JSONResponse(content={"questions": questions, "model": used_model})


def _chat_completion(
    client: OpenAI, model: str, oa_messages: List[dict[str, str]], temperature: float = 0.3
) -> Any:
    return client.chat.completions.create(
        model=model, messages=oa_messages, temperature=temperature
    )


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> JSONResponse:
    if os.environ.get("DAMA_DISABLE_CHAT", "").strip() in ("1", "true", "yes"):
        raise HTTPException(
            status_code=503,
            detail="Chat disabled due to budget limit. Topic search remains available.",
        )
    if os.environ.get("DAMA_USE_VERTEX", "").strip() in ("1", "true", "yes"):
        # Vertex AI chat path (Gemini).
        # NOTE: Implemented as a minimal provider; keeps response schema unchanged.
        try:
            import vertexai  # type: ignore
            from vertexai.generative_models import GenerativeModel  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Vertex AI not available: {e!s}")

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip() or os.environ.get(
            "GCP_PROJECT", ""
        ).strip()
        location = os.environ.get("GOOGLE_CLOUD_REGION", "").strip() or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", ""
        ).strip() or "us-central1"
        if not project:
            raise HTTPException(
                status_code=503,
                detail="Vertex AI enabled but GOOGLE_CLOUD_PROJECT not set.",
            )
        model_name = os.environ.get("DAMA_VERTEX_MODEL", "").strip() or "gemini-2.5-flash"
        max_output = int(os.environ.get("DAMA_MAX_OUTPUT_TOKENS", "").strip() or "384")

        ctx = req.context
        top = req.top_context
        sys = _chat_system_prompt(ctx, top, strict_retry=False)
        # Flatten to a single prompt: Gemini expects a single user content string in simplest form.
        convo: List[str] = [f"[SYSTEM]\n{sys}"]
        for m in req.messages:
            if m.role in ("user", "assistant") and m.content.strip():
                convo.append(f"[{m.role.upper()}]\n{m.content.strip()}")
        prompt = "\n\n".join(convo)

        try:
            vertexai.init(project=project, location=location)
            gm = GenerativeModel(model_name)
            resp = gm.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": max_output,
                },
            )
            text = (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Vertex AI error: {e!s}") from e

        reply, suggestions = _split_reply_and_suggestions(text)
        if _weak_chat_reply(reply):
            # One retry with stricter prompt.
            sys2 = _chat_system_prompt(ctx, top, strict_retry=True)
            convo2 = [f"[SYSTEM]\n{sys2}"]
            for m in req.messages:
                if m.role in ("user", "assistant") and m.content.strip():
                    convo2.append(f"[{m.role.upper()}]\n{m.content.strip()}")
            prompt2 = "\n\n".join(convo2)
            try:
                resp2 = gm.generate_content(
                    prompt2,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": max_output,
                    },
                )
                text2 = (getattr(resp2, "text", None) or "").strip()
                r2, s2 = _split_reply_and_suggestions(text2)
                if not _weak_chat_reply(r2):
                    reply, suggestions = r2, (s2 if s2 else suggestions)
                elif len(r2.strip()) > len((reply or "").strip()):
                    reply, suggestions = r2, (s2 if s2 else suggestions)
            except Exception:
                pass
        if not (reply or "").strip():
            reply = _CHAT_FALLBACK_REPLY
        return JSONResponse(
            content={
                "reply": reply,
                "suggestions": suggestions,
                "model": model_name,
                "provider": "vertex",
            }
        )

    client, model, provider = _resolve_llm()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No LLM: start Ollama (default), or set DAMA_USE_OLLAMA=0 and add "
                "Groq/OpenAI keys — see topic_search_server.py docstring."
            ),
        )
    ctx = req.context
    top = req.top_context

    def _oa_messages(strict_retry: bool) -> List[dict[str, str]]:
        sys = _chat_system_prompt(ctx, top, strict_retry=strict_retry)
        oa: List[dict[str, str]] = [{"role": "system", "content": sys}]
        for m in req.messages:
            if m.role in ("user", "assistant") and m.content.strip():
                oa.append({"role": m.role, "content": m.content.strip()})
        return oa

    oa_messages = _oa_messages(False)
    if len(oa_messages) < 2:
        raise HTTPException(status_code=400, detail="Need at least one user message.")
    used_model = model
    used_provider = provider
    used_client = client
    try:
        out = _chat_completion(client, model, oa_messages)
    except Exception as e:
        err_s = f"{e!s}"
        quota_like = (
            "429" in err_s
            or "insufficient_quota" in err_s
            or "rate_limit" in err_s.lower()
        )
        if provider == "openai" and quota_like:
            gcli, gmodel = _groq_client_and_model()
            if gcli is not None:
                try:
                    out = _chat_completion(gcli, gmodel, oa_messages)
                    used_model = gmodel
                    used_provider = "groq_fallback"
                    used_client = gcli
                except Exception as e2:
                    raise HTTPException(
                        status_code=502,
                        detail=f"OpenAI failed ({e!s}); Groq fallback failed: {e2!s}",
                    ) from e2
            else:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"LLM error: {e!s} "
                        "(add creds/groqkey.txt for free Groq, or use Ollama — see docstring)"
                    ),
                ) from e
        else:
            raise HTTPException(status_code=502, detail=f"LLM error: {e!s}") from e
    text = (out.choices[0].message.content or "").strip()
    reply, suggestions = _split_reply_and_suggestions(text)
    if _weak_chat_reply(reply):
        try:
            out2 = _chat_completion(
                used_client, used_model, _oa_messages(True), temperature=0.2
            )
            text2 = (out2.choices[0].message.content or "").strip()
            r2, s2 = _split_reply_and_suggestions(text2)
            if not _weak_chat_reply(r2):
                reply, suggestions = r2, (s2 if s2 else suggestions)
            elif len(r2.strip()) > len((reply or "").strip()):
                reply, suggestions = r2, (s2 if s2 else suggestions)
        except Exception:
            pass
    if not (reply or "").strip():
        reply = _CHAT_FALLBACK_REPLY
    return JSONResponse(
        content={
            "reply": reply,
            "suggestions": suggestions,
            "model": used_model,
            "provider": used_provider,
        }
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    # Starlette's Jinja2Templates.TemplateResponse signature is:
    #   TemplateResponse(request, name, context=None, ...)
    return TEMPLATES.TemplateResponse(request, "topic_search.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "rows": str(len(_rows()))}
