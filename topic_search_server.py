"""
DAMA — Anguttara Nikāya Books 7–8 topic UI + LLM chat. FastAPI + Uvicorn.
Run: uvicorn topic_search_server:app --host 127.0.0.1 --port 8020

Data: processed transcript/an7.json + an8.json (merged).

LLM priority (first match wins):
  1) DAMA_LLM_BASE_URL — explicit Ollama/custom OpenAI-compatible base
  2) Local Ollama at http://127.0.0.1:11434/v1 — unless DAMA_USE_OLLAMA=0
  3) Groq: GROQ_API_KEY or creds/groqkey.txt
  4) OpenAI: OPENAI_API_KEY or creds/openaikey.txt (429 → Groq fallback if configured)

Models: DAMA_CHAT_MODEL (Ollama default llama3.2), DAMA_GROQ_MODEL, OpenAI default gpt-4o-mini.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
_DATA_DIR = ROOT / "processed transcript"
_PATH_AN7 = _DATA_DIR / "an7.json"
_PATH_AN8 = _DATA_DIR / "an8.json"
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

_DEFAULT_CHAT_MODEL = "gpt-4o-mini"


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
    """Book 7 then book 8; skip missing files."""
    rows = _load_json_array(_PATH_AN7)
    rows.extend(_load_json_array(_PATH_AN8))
    return rows


ROWS: List[dict[str, Any]] = _load_all_rows()


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


def score_row(row: dict[str, Any], toks: List[str]) -> float:
    chain = row.get("chain") or {}
    items = chain.get("items") if isinstance(chain, dict) else None
    chain_blob = _normalize(" ".join(items) if isinstance(items, list) else "")
    comm = _normalize(str(row.get("commentary") or ""))
    sut = _normalize(str(row.get("sutta") or ""))
    total = 0.0
    for t in toks:
        nt = _normalize(t)
        if len(nt) < 2:
            continue
        total += WEIGHT_CHAIN * _count_occurrences(chain_blob, nt)
        total += WEIGHT_COMMENTARY * _count_occurrences(comm, nt)
        total += WEIGHT_SUTTA * _count_occurrences(sut, nt)
    return total


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


def _pick_stratified_chain_hits(
    scored: List[Tuple[float, dict[str, Any]]],
    *,
    max_columns: int,
) -> List[Tuple[float, dict[str, Any]]]:
    """
    Prefer one top-scoring hit per book (AN7 vs AN8), then fill remaining
    slots by global score. Avoids showing three AN8 rows when AN7 also matches.
    """
    scored_chain = [(s, r) for s, r in scored if _has_chain(r)]
    if not scored_chain:
        return []
    chosen: List[Tuple[float, dict[str, Any]]] = []
    chosen_ids: set[str] = set()

    for book in (7, 8):
        if len(chosen) >= max_columns:
            break
        for s, row in scored_chain:
            if _book_from_sutta_id(row.get("sutta_id")) != book:
                continue
            sid = _sutta_id_key(row)
            if sid in chosen_ids:
                continue
            chosen.append((s, row))
            chosen_ids.add(sid)
            break

    for s, row in scored_chain:
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
    for row in ROWS:
        s = score_row(row, toks)
        if s > 0:
            scored.append((s, row))
    scored.sort(key=lambda x: -x[0])

    hits_rows = _pick_stratified_chain_hits(scored, max_columns=max_columns)

    if not scored:
        bot_summary = f"BOT: {(q or '').strip()} (no matches)"
    elif not hits_rows:
        bot_summary = f"BOT: {(q or '').strip()} (no segments with extracted chains)"
    else:
        bot_summary = build_bot_summary(q, [r for _, r in hits_rows])

    hits = [row_to_hit(row, s) for s, row in hits_rows]
    return {
        "query": q,
        "tokens": toks,
        "bot_summary": bot_summary,
        "top": hits,
    }


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


def _chat_system_prompt(
    context: Optional[ChatContext],
    top_context: Optional[List[ChatContext]],
) -> str:
    blocks: List[str] = [
        "You are DAMA, a helpful assistant for Anguttara Nikāya Books 7 and 8 study materials (teacher reading + commentary excerpts).",
        "Answer using ONLY the provided sutta reading and teacher commentary excerpts.",
        "If the answer is not in the context, say you do not know from this material.",
        "Cite sutta_id when relevant. Be concise.",
    ]
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


def _groq_client_and_model() -> tuple[Optional[OpenAI], str]:
    gq = _groq_api_key()
    if not gq:
        return None, ""
    model = (
        os.environ.get("DAMA_GROQ_MODEL", "").strip()
        or _DEFAULT_GROQ_MODEL
    )
    return OpenAI(base_url=GROQ_BASE_URL, api_key=gq), model


app = FastAPI(title="DAMA — AN 7–8")


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    client, model, provider = _resolve_llm()
    return {
        "chat_available": client is not None,
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


def _chat_completion(
    client: OpenAI, model: str, oa_messages: List[dict[str, str]]
) -> Any:
    return client.chat.completions.create(
        model=model, messages=oa_messages, temperature=0.4
    )


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> JSONResponse:
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
    system = _chat_system_prompt(ctx, top)
    oa_messages: List[dict[str, str]] = [{"role": "system", "content": system}]
    for m in req.messages:
        if m.role in ("user", "assistant") and m.content.strip():
            oa_messages.append({"role": m.role, "content": m.content.strip()})
    if len(oa_messages) < 2:
        raise HTTPException(status_code=400, detail="Need at least one user message.")
    used_model = model
    used_provider = provider
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
    return JSONResponse(
        content={"reply": text, "model": used_model, "provider": used_provider}
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return TEMPLATES.TemplateResponse(
        "topic_search.html",
        {
            "request": request,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "rows": str(len(ROWS))}
