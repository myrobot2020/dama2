import json
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import requests

import an1_build_index as an1_build
import an1_vertex_core as vx
from an1_build_index import (
    _commentary_body,
    _commentary_id,
    _extract_records_fallback,
    _parse_json_lenient,
)


BASE_DIR = Path(__file__).resolve().parent
NOTES_DIR = BASE_DIR / "sutta_notes"
AN1_PATH = an1_build.AN1_PATH
PERSIST_DIR = an1_build.PERSIST_DIR
COLLECTION_NAME = an1_build.COLLECTION_NAME

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:instruct"

# Bumped when RAG/LLM behavior changes (shown in GET /api/index_status so you know the server reloaded).
AN1_APP_BUILD = "2026-04-11-vertex-bundle-mode"

DEBUG_LOG_PATH = BASE_DIR / "debug-655121.log"


def _dbg(hypothesis_id: str, location: str, message: str, data: Optional[Dict[str, Any]] = None, run_id: str = "pre-fix") -> None:
    try:
        payload = {
            "sessionId": "655121",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(__import__("time").time() * 1000),
        }
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=20)
    use_llm: bool = Field(default=True)


class Chunk(BaseModel):
    source: str = ""
    suttaid: str = ""
    commentary_id: str = ""
    kind: str = ""
    distance: Optional[float] = None
    text: str


class QueryResponse(BaseModel):
    chunks: List[Chunk]
    answer: str = ""
    used_llm: bool = False


class BuildResponse(BaseModel):
    ok: bool
    collection_count: int


class ItemSummary(BaseModel):
    suttaid: str
    title: str = ""
    has_commentary: bool = False


class ItemDetail(BaseModel):
    suttaid: str
    sutta: str
    commentry: str = ""
    commentary_id: str = ""


_lock = threading.RLock()
_embed_model: Optional[Any] = None
_reranker: Optional[Any] = None
_items_cache: Optional[List[ItemDetail]] = None


def _get_collection() -> Any:
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


def _get_embed_model() -> Any:
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _dbg("H5", "an1_app.py:_get_embed_model", "Loading embedding model", {"model": "all-MiniLM-L6-v2"})
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_reranker() -> Any:
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _dbg("H5", "an1_app.py:_get_reranker", "Loading reranker", {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2"})
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


_STOPWORDS = frozenset(
    "a an and are as at be but by do for from get got has have he her him his how "
    "if in into is it its just me my no nor not now of on or our out own she so "
    "some than that the their them then there these they this those through to too "
    "very was we were what when where which who will with would you your".split()
)


def _tokenize_query(q: str) -> List[str]:
    tokens = [
        t for t in re.findall(r"[a-zA-Z0-9']+", (q or "").lower())
        if len(t) >= 3 and t not in _STOPWORDS
    ]
    return tokens[:12]


def _chunk_from_doc(doc: Any, meta: Any, dist: Optional[float] = None) -> Chunk:
    src = ""
    sid = ""
    cid = ""
    kind = ""
    if isinstance(meta, dict):
        src = str(meta.get("source") or "")
        sid = str(meta.get("suttaid") or "")
        cid = str(meta.get("commentary_id") or "")
        kind = str(meta.get("kind") or "")
    return Chunk(
        source=src,
        suttaid=sid,
        commentary_id=cid,
        kind=kind,
        distance=float(dist) if dist is not None else None,
        text=str(doc),
    )


def _retrieval_boost_phrases(query: str) -> List[str]:
    """Multi-word or rare tokens that token search can miss on long questions (phrase > 60 chars)."""
    ql = (query or "").lower()
    out: List[str] = []
    if "thorough attention" in ql:
        out.append("thorough attention")
    if "yoniso" in ql:
        out.append("Yoniso")
    if "manasikara" in ql:
        out.append("Manasikara")
    if "feature of beauty" in ql:
        out.append("feature of beauty")
    if "subha-nimitta" in ql:
        out.append("Subha-nimitta")
    elif "nimitta" in ql and "subha" in ql:
        out.append("Subha-nimitta")
    # Sense-sphere: AN 1.1.2 text uses "scent"; users often say "smell".
    if any(x in ql for x in ("smell", "scent", "odor", "fragrance")):
        out.append("scent")
    for w in ("sound", "touch", "taste"):
        if w in ql:
            out.append(w)
    if "oyster" in ql:
        out.append("oysters")
    if "pool" in ql and "water" in ql:
        out.append("pool of water")
    if "water" in ql:
        out.append("pool of water")
    return list(dict.fromkeys([p for p in out if p]))


def _dedupe_chunks_keep_order(chunks: List[Chunk]) -> List[Chunk]:
    seen: set = set()
    out: List[Chunk] = []
    for c in chunks:
        key = (c.source, c.suttaid, (c.kind or "").lower(), c.text[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _inject_pair_both_ways(top: List[Chunk], pool: List[Chunk], cap: int = 14) -> List[Chunk]:
    """For each suttaid present, pull in missing sutta or commentary chunk from pool when available."""
    merged = _dedupe_chunks_keep_order(list(top))
    for _ in range(2):
        sids = {c.suttaid for c in merged if c.suttaid}
        have = {(c.suttaid, (c.kind or "").lower()) for c in merged if c.suttaid}
        for sid in sids:
            for kind_need in ("commentary", "sutta"):
                if (sid, kind_need) in have:
                    continue
                for c in pool:
                    if c.suttaid == sid and (c.kind or "").lower() == kind_need:
                        merged.append(c)
                        break
        merged = _dedupe_chunks_keep_order(merged)
    return merged[:cap]


def _lexical_score(query: str, chunk_text: str) -> int:
    q = (query or "").strip().lower()
    if not q or not chunk_text:
        return 0
    text = str(chunk_text).lower()
    score = 0
    if len(q) <= 60 and q in text:
        score += 100
    tokens = _tokenize_query(q)
    if tokens:
        matched = sum(1 for t in tokens if t in text)
        ratio = matched / len(tokens)
        if ratio >= 0.6:
            score += int(ratio * 20)
    return score


def _retrieve(embed_model: Any, collection: Any, query: str, k: int) -> List[Chunk]:
    q_emb = embed_model.encode([query])[0].tolist()
    n_candidates = min(max(k * 10, 50), 200)
    results: Dict[str, Any] = collection.query(
        query_embeddings=[q_emb],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out: List[Chunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(_chunk_from_doc(doc, meta, float(dist) if dist is not None else None))

    phrase = (query or "").strip().lower()
    terms = []
    if phrase and len(phrase) <= 60 and len(phrase.split()) >= 2:
        terms.append(phrase)
    terms.extend(_tokenize_query(query))
    terms = list(dict.fromkeys([t for t in terms if t]))
    boost = _retrieval_boost_phrases(query)
    merged_terms = list(dict.fromkeys([*boost, *terms]))[:10]

    seen_keys = set((c.source, c.suttaid, c.text[:200]) for c in out)
    for t in merged_terms:
        try:
            got = collection.get(
                where_document={"$contains": t},
                include=["documents", "metadatas"],
                limit=min(200, max(50, k * 20)),
            )
        except Exception:
            continue
        k_docs = (got or {}).get("documents", []) or []
        k_metas = (got or {}).get("metadatas", []) or []
        for doc, meta in zip(k_docs, k_metas):
            ch = _chunk_from_doc(doc, meta, None)
            key = (ch.source, ch.suttaid, ch.text[:200])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(ch)

    out.sort(
        key=lambda c: (
            -_lexical_score(query, c.text),
            c.distance if c.distance is not None else 1e9,
        )
    )
    candidates = out[:max(k * 4, 20)]

    if len(candidates) > k:
        try:
            reranker = _get_reranker()
            pairs = [[query, c.text] for c in candidates]
            scores = reranker.predict(pairs)
            scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
            candidates = [c for c, _ in scored]
        except Exception:
            pass

    top = _inject_pair_both_ways(candidates[:k], out, cap=14)

    has_commentary = any((c.kind or "").lower() == "commentary" for c in top)
    if not has_commentary:
        for c in out:
            if (c.kind or "").lower() == "commentary":
                if top:
                    top = top[: max(0, k - 1)] + [c]
                else:
                    top = [c]
                break

    has_sutta = any((c.kind or "").lower() == "sutta" for c in top)
    if not has_sutta:
        for c in out:
            if (c.kind or "").lower() == "sutta":
                if top:
                    top = top[: max(0, k - 1)] + [c]
                else:
                    top = [c]
                break

    return _dedupe_chunks_keep_order(top)[:14]


def _retrieve_vertex(bundle: Dict[str, Any], query: str, k: int) -> List[Chunk]:
    """Same ranking heuristics as _retrieve, but vector search + lexical scan over a Vertex bundle (no Chroma)."""
    rows_all = vx.bundle_chunk_rows(bundle)
    if not rows_all:
        return []

    q_emb = vx.embed_texts_vertex([query])[0]
    out: List[Chunk] = []
    for row in rows_all:
        emb = row.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        try:
            emb_f = [float(x) for x in emb]
        except (TypeError, ValueError):
            continue
        dist = vx.cosine_distance(q_emb, emb_f)
        out.append(
            _chunk_from_doc(
                str(row.get("text") or ""),
                {
                    "source": row.get("source") or "",
                    "suttaid": row.get("suttaid") or "",
                    "commentary_id": row.get("commentary_id") or "",
                    "kind": row.get("kind") or "",
                },
                float(dist),
            )
        )

    phrase = (query or "").strip().lower()
    terms = []
    if phrase and len(phrase) <= 60 and len(phrase.split()) >= 2:
        terms.append(phrase)
    terms.extend(_tokenize_query(query))
    terms = list(dict.fromkeys([t for t in terms if t]))
    boost = _retrieval_boost_phrases(query)
    merged_terms = list(dict.fromkeys([*boost, *terms]))[:10]

    seen_keys = set((c.source, c.suttaid, c.text[:200]) for c in out)
    for t in merged_terms:
        tl = t.lower()
        for row in rows_all:
            doc = str(row.get("text") or "")
            if tl not in doc.lower():
                continue
            ch = _chunk_from_doc(
                doc,
                {
                    "source": row.get("source") or "",
                    "suttaid": row.get("suttaid") or "",
                    "commentary_id": row.get("commentary_id") or "",
                    "kind": row.get("kind") or "",
                },
                None,
            )
            key = (ch.source, ch.suttaid, ch.text[:200])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(ch)

    out.sort(
        key=lambda c: (
            -_lexical_score(query, c.text),
            c.distance if c.distance is not None else 1e9,
        )
    )
    candidates = out[: max(k * 4, 20)]

    top = _inject_pair_both_ways(candidates[:k], out, cap=14)

    has_commentary = any((c.kind or "").lower() == "commentary" for c in top)
    if not has_commentary:
        for c in out:
            if (c.kind or "").lower() == "commentary":
                if top:
                    top = top[: max(0, k - 1)] + [c]
                else:
                    top = [c]
                break

    has_sutta = any((c.kind or "").lower() == "sutta" for c in top)
    if not has_sutta:
        for c in out:
            if (c.kind or "").lower() == "sutta":
                if top:
                    top = top[: max(0, k - 1)] + [c]
                else:
                    top = [c]
                break

    return _dedupe_chunks_keep_order(top)[:14]


def _ollama_chat(messages: list, temperature: float = 0, num_ctx: int = 4096, timeout: int = 600) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text[:500]}")
    return resp.json().get("message", {}).get("content", "")


def _sanitize_chat_answer_light(text: str) -> str:
    """Remove legacy 'Excerpt N' echoes; keep (AN …) and (cAN …) intact."""
    if not (text or "").strip():
        return text or ""
    s = re.sub(r"\b[Ee]xcerpt\s*\d+\b", "", text)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def _chunks_for_llm_paired_by_suttaid(chunks: List[Chunk], max_n: int = 10) -> List[Chunk]:
    """
    Walk suttaids in retrieval order; for each id emit sutta chunk(s) then commentary chunk(s)
    so the model sees (AN …) text before (cAN …) for the same suttaid.
    """
    chunks = _dedupe_chunks_keep_order(chunks)
    if not chunks:
        return []
    by_sid: Dict[str, List[Chunk]] = {}
    order: List[str] = []
    for c in chunks:
        sid = (c.suttaid or "").strip()
        if not sid:
            continue
        if sid not in by_sid:
            by_sid[sid] = []
            order.append(sid)
        by_sid[sid].append(c)

    out: List[Chunk] = []
    for sid in order:
        grp = by_sid[sid]
        comm = [x for x in grp if (x.kind or "").lower() == "commentary"]
        sut = [x for x in grp if (x.kind or "").lower() == "sutta"]
        for x in sut[:1] + comm[:2]:
            if len(out) >= max_n:
                return out
            out.append(x)

    seen = {(x.source, x.suttaid, (x.kind or "").lower(), x.text[:80]) for x in out}
    for c in chunks:
        key = (c.source, c.suttaid, (c.kind or "").lower(), c.text[:80])
        if key in seen or len(out) >= max_n:
            continue
        seen.add(key)
        out.append(c)
    return out[:max_n]


def _format_llm_passage(idx: int, c: Chunk) -> str:
    kind = (c.kind or "").lower()
    if kind == "sutta":
        hint = (
            "QUOTE_RULE: For (AN …) you may ONLY copy text from the SUTTA: section below. "
            "This block has no TEACHER COMMENTARY — do not call commentary ideas a 'sutta quote'.\n"
        )
    elif kind == "commentary":
        hint = (
            "QUOTE_RULE: For (cAN …) you may ONLY copy text from the TEACHER COMMENTARY: section below. "
            "Do not attribute this wording to 'the sutta' or (AN …).\n"
        )
    else:
        hint = "QUOTE_RULE: Match quotes to the correct SUTTA vs TEACHER COMMENTARY section.\n"
    return (
        f"[{idx} | suttaid: {c.suttaid or '-'} | commentary_id: {c.commentary_id or '-'} | kind: {c.kind or '-'}]\n"
        f"{hint}"
        f"{c.text}"
    )


def _llm_system_and_user_blocks(query: str, balanced: List[Chunk]) -> Tuple[str, str]:
    numbered = "\n\n".join(_format_llm_passage(i + 1, c) for i, c in enumerate(balanced))
    system = (
        "You are a friendly, conversational assistant answering from AN1 sutta and teacher commentary "
        "passages below.\n"
        "- kind 'commentary' = teacher notes (text after TEACHER COMMENTARY:). Cite with (cAN …) using "
        "commentary_id from that passage’s header.\n"
        "- kind 'sutta' = sutta text (after SUTTA:). Cite with (AN …) using suttaid from that passage’s header.\n\n"
        "STRICT RULES:\n"
        "1. PASSAGES are grouped by suttaid: sutta for that id appears before teacher commentary for the "
        "same id. Prefer quoting sutta and teacher notes from the SAME suttaid in one answer. Do NOT pair "
        "(AN 1.2) with (cAN 1.5.6) unless you write one clear sentence explaining why two discourses are "
        "needed; otherwise stay within one suttaid pair.\n"
        "2. Structure: for your main example, FIRST a verbatim SUTTA quote + (AN …), THEN a verbatim "
        "TEACHER COMMENTARY quote + (cAN …) for the same suttaid when both are in PASSAGES. You may say "
        "e.g. 'According to the commentary…' before the teacher quote.\n"
        "3. Never put (cAN …) on sutta quotes or (AN …) on commentary quotes.\n"
        "4. HARD: Any phrase inside quotation marks before (AN …) must be copied verbatim from a "
        "passage block whose kind is 'sutta' (text under SUTTA: only). Never call TEACHER COMMENTARY "
        "wording 'the sutta' or tag it (AN …). Likewise, quoted text before (cAN …) must come only from "
        "TEACHER COMMENTARY: in a kind 'commentary' block.\n"
        "5. If PASSAGES include a sutta about water, pools, oysters, etc., do not claim excerpts omit water.\n"
        "6. If only commentary or only sutta appears for the relevant suttaid, say what is missing briefly.\n"
        "7. Do not write 'Excerpt' / internal passage numbers from headers.\n"
        "8. If the question uses 'this' / 'it' without a clear topic, ask ONE short clarifying question.\n"
        "9. If nothing in PASSAGES answers the question, say: "
        "\"The provided excerpts do not contain information to answer this question.\"\n"
        "10. Do NOT guess or use outside knowledge."
    )
    user = (
        f"Question: {query}\n\nPASSAGES (grouped by suttaid: sutta for an id, then teacher notes for that id; "
        f"headers for you only — do not echo them):\n{numbered}\n\n"
        "Answer: use one suttaid’s sutta quote + (AN …) first (words must appear under SUTTA: in a "
        "kind:sutta passage), then teacher commentary quote + (cAN …) (words under TEACHER COMMENTARY: "
        "in kind:commentary). Avoid unrelated cross-citations; never call teacher notes the sutta."
    )
    return system, user


def _call_llm_vertex(query: str, chunks: List[Chunk]) -> str:
    balanced = _chunks_for_llm_paired_by_suttaid(chunks, max_n=10)
    if not balanced:
        return "No passages were retrieved for this question. Try Rebuild or rephrase."
    system, user = _llm_system_and_user_blocks(query, balanced)
    return _sanitize_chat_answer_light(vx.gemini_generate(system, user, temperature=0.2))


def _call_llm(query: str, chunks: List[Chunk]) -> str:
    if vx.an1_vertex_enabled():
        return _call_llm_vertex(query, chunks)

    balanced = _chunks_for_llm_paired_by_suttaid(chunks, max_n=10)
    if not balanced:
        return "No passages were retrieved for this question. Try Rebuild or rephrase."
    system, user = _llm_system_and_user_blocks(query, balanced)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _sanitize_chat_answer_light(_ollama_chat(messages, num_ctx=8192, timeout=300))


def _load_items() -> List[ItemDetail]:
    global _items_cache
    if _items_cache is not None:
        return _items_cache
    if not AN1_PATH.exists():
        _dbg("H1", "an1_app.py:_load_items", "AN1_PATH missing", {"an1_path": str(AN1_PATH)})
        _items_cache = []
        return _items_cache
    raw = AN1_PATH.read_text(encoding="utf-8", errors="ignore")
    try:
        data = _parse_json_lenient(raw)
    except Exception as e:
        _dbg(
            "H1",
            "an1_app.py:_load_items",
            "JSON parse failed",
            {"error": str(e), "bytes": len(raw), "prefix": raw[:200]},
        )
        try:
            data = _extract_records_fallback(raw)
            _dbg("H1", "an1_app.py:_load_items", "Fallback extracted records", {"count": len(data)})
        except Exception as e2:
            _dbg("H1", "an1_app.py:_load_items", "Fallback parse failed", {"error": str(e2)})
            _items_cache = []
            return _items_cache
    items: List[ItemDetail] = []
    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            sid = str(obj.get("suttaid") or "").strip()
            sutta = str(obj.get("sutta") or "").strip()
            comm = _commentary_body(obj)
            cid = _commentary_id(obj)
            if not sid or not sutta:
                continue
            items.append(ItemDetail(suttaid=sid, sutta=sutta, commentry=comm, commentary_id=cid))
    else:
        _dbg("H1", "an1_app.py:_load_items", "Unexpected JSON top-level type", {"type": str(type(data))})
    _items_cache = items
    _dbg("H1", "an1_app.py:_load_items", "Loaded items", {"count": len(items)})
    return _items_cache


def _item_title(item: ItemDetail) -> str:
    # lightweight label for the left nav
    s = (item.sutta or "").strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s[:72] + ("…" if len(s) > 72 else "")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if vx.an1_vertex_enabled():
        try:
            vx.ensure_bundle_loaded(PERSIST_DIR)
        except Exception as e:
            _dbg("H2", "an1_app.py:lifespan", "Vertex bundle preload failed (will retry on request)", {"error": str(e)})
    yield


app = FastAPI(title="Dama AN1 RAG", lifespan=_lifespan)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Dama — AN1 RAG (local)</title>
    <style>
      :root {
        --bg: #0b0f19;
        --panel: #0f1626;
        --panel2: #0c1220;
        --text: #e9eefc;
        --muted: #a7b3d6;
        --border: rgba(255,255,255,.10);
        --border2: rgba(255,255,255,.07);
        --accent: #ffcc33;
        --accent2: #7c5cff;
        --good: #41d17a;
        --bad: #ff5c7a;
      }
      * { box-sizing: border-box; }
      html, body { height: 100%; }
      body {
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
        color: var(--text);
        background: radial-gradient(1200px 700px at 20% 0%, #1c2a50 0%, var(--bg) 58%) fixed;
      }
      .topbar {
        height: 54px;
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 14px;
        border-bottom: 1px solid var(--border);
        background: rgba(0,0,0,.22);
        backdrop-filter: blur(6px);
      }
      .brand { display:flex; flex-direction: column; gap: 2px; }
      .brand .name { font-size: 13px; letter-spacing: .2px; }
      .brand .sub { font-size: 11px; color: var(--muted); }
      .rightmeta { display:flex; gap:10px; align-items:center; color: var(--muted); font-size: 12px; }
      .pill { padding: 6px 10px; border: 1px solid var(--border); border-radius: 999px; background: rgba(255,255,255,.04); }
      .layout { height: calc(100% - 54px); display: grid; grid-template-columns: 280px 1fr 420px; gap: 12px; padding: 12px; }
      .pane { border: 1px solid var(--border); border-radius: 14px; background: rgba(15,22,38,.78); backdrop-filter: blur(6px); overflow: hidden; display:flex; flex-direction: column; min-height: 0; }
      .pane .hdr { padding: 8px 10px; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--border2); display:flex; align-items:center; justify-content: space-between; gap: 10px;}
      .pane .body { padding: 8px 10px; min-height: 0; overflow: auto; }
      .searchRow { display:flex; gap: 8px; align-items:center; }
      input[type="text"], textarea {
        width: 100%;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(0,0,0,.22);
        color: var(--text);
        padding: 8px 10px;
        outline: none;
      }
      button {
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,.06);
        color: var(--text);
        padding: 8px 10px;
        cursor: pointer;
      }
      button.primary { background: linear-gradient(180deg, rgba(124,92,255,.9), rgba(124,92,255,.6)); border-color: rgba(124,92,255,.7); }
      button.warn { background: linear-gradient(180deg, rgba(255,204,51,.40), rgba(255,204,51,.18)); border-color: rgba(255,204,51,.35); }
      button:disabled { opacity: .55; cursor: not-allowed; }
      .list { display:flex; flex-direction: column; gap: 8px; }
      .card {
        border: 1px solid var(--border2);
        background: rgba(0,0,0,.16);
        border-radius: 12px;
        padding: 10px 10px;
        cursor: pointer;
      }
      .card:hover { border-color: rgba(255,255,255,.18); }
      .card .row1 { display:flex; justify-content: space-between; gap: 10px; align-items:center; }
      .sid { font-size: 12px; color: var(--text); font-weight: 700; }
      .tag { font-size: 11px; color: var(--muted); border: 1px solid var(--border2); padding: 3px 8px; border-radius: 999px; }
      .title { margin-top: 6px; font-size: 12px; color: #dbe6ff; line-height: 1.25; }
      .sectionTitle { font-size: 12px; color: var(--muted); margin: 10px 0 6px; }
      .reading h2 { margin: 0; font-size: 12px; color: var(--muted); letter-spacing: .6px; }
      .reading .txt { margin-top: 8px; white-space: pre-wrap; word-break: break-word; color: #dbe6ff; font-size: 13px; line-height: 1.42; }
      .reading .split { height: 1px; background: var(--border2); margin: 12px 0; }
      .muted { color: var(--muted); font-size: 12px; }
      .chatArea { display:flex; flex-direction: column; gap: 8px; min-height: 0; }
      .chatLog { flex: 1; min-height: 0; overflow: auto; padding-bottom: 2px; }
      textarea { resize: none; min-height: 44px; max-height: 44px; }
      .chatLog { display:flex; flex-direction: column; gap: 10px; }
      .msg { display:flex; }
      .bubble {
        max-width: 92%;
        border-radius: 14px;
        border: 1px solid var(--border2);
        padding: 10px 10px;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 13px;
        line-height: 1.42;
        background: rgba(0,0,0,.14);
      }
      .msg.user { justify-content: flex-end; }
      .msg.user .bubble { background: rgba(124,92,255,.18); border-color: rgba(124,92,255,.35); }
      .msg.asst { justify-content: flex-start; }
      .msg.asst .bubble { background: rgba(0,0,0,.14); }
      .msg.asst .bubble small.cite { font-size: 0.76em; color: var(--muted); font-weight: 500; }
      .quote {
        border: 1px solid var(--border2);
        background: rgba(0,0,0,.12);
        border-radius: 12px;
        padding: 10px;
      }
      .quote .meta { display:flex; gap:10px; flex-wrap: wrap; align-items:center; color: var(--muted); font-size: 11px; margin-bottom: 6px; }
      .quote .meta b { color: var(--text); font-weight: 700; }
      .quote pre { margin: 0; white-space: pre-wrap; word-break: break-word; color: #dbe6ff; font-size: 12.5px; line-height: 1.35; }
      .copy { margin-left: auto; padding: 5px 9px; border-radius: 10px; border: 1px solid var(--border2); background: rgba(255,255,255,.04); cursor: pointer; color: var(--muted); }
      .small { font-size: 11px; }
      .row { display:flex; gap: 8px; align-items:center; flex-wrap: wrap; }
      .spacer { flex: 1; }
      .toggle { display:flex; align-items:center; gap: 8px; }
      .kInput { width: 76px; }
      .kInput { height: 30px; padding: 6px 8px; border-radius: 10px; }
      .tightBtn { height: 30px; padding: 6px 10px; border-radius: 10px; }
      .hdrLabel { font-size: 12px; color: var(--muted); }
      .msgFooter { margin-top: 6px; display:flex; align-items:center; gap: 8px; color: var(--muted); font-size: 11px; }
      .thumb { padding: 4px 8px; border-radius: 999px; border: 1px solid var(--border2); background: rgba(255,255,255,.04); cursor: pointer; color: var(--muted); }
      .thumb.active { border-color: rgba(255,204,51,.45); color: #ffe08a; background: rgba(255,204,51,.10); }
      .composer { display:flex; gap: 8px; align-items: center; }
      .composer textarea { flex: 1; }
      .composer .right { display:flex; gap: 8px; align-items:center; }
      .composer .right label { display:flex; align-items:center; gap: 6px; }
      .pane.rightPane .body { display:flex; flex-direction: column; min-height: 0; }
      @media (max-width: 1200px) { .layout { grid-template-columns: 260px 1fr; } .rightPane { grid-column: 1 / -1; } }
    </style>
  </head>
  <body>
    <div class="topbar">
      <div class="brand">
        <div class="name" id="brandName">Dama — AN1</div>
        <div class="sub">Index: <span id="idxStatus">unknown</span> · Source: processed scipts2/an1.json</div>
      </div>
      <div class="rightmeta">
        <span class="pill">UI: 3‑pane</span>
        <span class="pill small">API: /api/query</span>
      </div>
    </div>

    <div class="layout">
      <div class="pane">
        <div class="hdr">
          <div>Search</div>
          <div class="row">
            <button id="rebuild" class="warn">Rebuild</button>
          </div>
        </div>
        <div class="body">
          <div class="searchRow">
            <input id="filter" type="text" placeholder="Filter suttas…" />
            <button id="reload">Load</button>
          </div>
          <div class="sectionTitle">Suttas</div>
          <div id="items" class="list"></div>
          <div id="leftStatus" class="muted" style="margin-top:10px;"></div>
        </div>
      </div>

      <div class="pane reading">
        <div class="hdr">
          <div>SUTTA READING</div>
          <div class="muted" id="readingMeta"></div>
        </div>
        <div class="body">
          <div id="readingEmpty" class="muted">Select a sutta on the left.</div>
          <div id="readingBody" style="display:none;">
            <h2>SUTTA</h2>
            <div id="suttaText" class="txt"></div>
            <div class="split"></div>
            <h2>TEACHER COMMENTARY</h2>
            <div id="commText" class="txt"></div>
          </div>
        </div>
      </div>

      <div class="pane rightPane">
        <div class="hdr">
          <div class="hdrLabel">Chat</div>
          <div class="muted small">Enter=send · Shift+Enter=newline</div>
        </div>
        <div class="body">
          <div class="chatArea">
            <div id="chatLog" class="chatLog"></div>
            <div class="composer">
              <textarea id="q" placeholder="Ask about AN1…"></textarea>
              <div class="right">
                <button id="ask" class="primary tightBtn">Send</button>
              </div>
            </div>
            <div id="chatStatus" class="muted small"></div>
            <div id="quotes"></div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const itemsEl = document.getElementById('items');
      const leftStatusEl = document.getElementById('leftStatus');
      const idxStatusEl = document.getElementById('idxStatus');
      const filterEl = document.getElementById('filter');
      const reloadBtn = document.getElementById('reload');
      const rebuildBtn = document.getElementById('rebuild');
      const askBtn = document.getElementById('ask');
      const qEl = document.getElementById('q');
      const chatStatusEl = document.getElementById('chatStatus');
      const quotesEl = document.getElementById('quotes');
      const chatLogEl = document.getElementById('chatLog');

      const readingEmptyEl = document.getElementById('readingEmpty');
      const readingBodyEl = document.getElementById('readingBody');
      const readingMetaEl = document.getElementById('readingMeta');
      const suttaTextEl = document.getElementById('suttaText');
      const commTextEl = document.getElementById('commText');

      function esc(s) { return (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
      function formatAsstAnswer(raw) {
        let s = esc(raw ?? '');
        s = s.replace(/\\((cAN[^)]{0,160})\\)/gi, '<small class="cite">($1)</small>');
        s = s.replace(/\\((AN[^)]{0,160})\\)/g, '<small class="cite">($1)</small>');
        return s;
      }
      function setLeftStatus(s) { leftStatusEl.textContent = s || ''; }
      function setChatStatus(s) { chatStatusEl.textContent = s || ''; }
      function setIdxStatus(s) { idxStatusEl.textContent = s || ''; }

      let allItems = [];
      let selectedId = null;
      let chat = [];

      function appendMsg(role, text, meta) {
        const id = 'm_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2,8);
        chat.push({ id, role, text, meta: meta || null, rating: 0 });
        const wrap = document.createElement('div');
        wrap.className = 'msg ' + (role === 'user' ? 'user' : 'asst');
        const b = document.createElement('div');
        b.className = 'bubble';
        if (role === 'user') {
          b.textContent = text || '';
        } else {
          b.innerHTML = formatAsstAnswer(text || '');
        }
        wrap.appendChild(b);
        if (role !== 'user') {
          const footer = document.createElement('div');
          footer.className = 'msgFooter';
          const spacer = document.createElement('div');
          spacer.style.flex = '1';
          footer.appendChild(spacer);

          const up = document.createElement('button');
          up.className = 'thumb';
          up.textContent = '👍';
          const down = document.createElement('button');
          down.className = 'thumb';
          down.textContent = '👎';

          function setRating(r) {
            chat = chat.map(m => (m.id === id ? { ...m, rating: r } : m));
            up.classList.toggle('active', r === 1);
            down.classList.toggle('active', r === -1);
            // best-effort feedback; ignore errors
            fetch('/api/feedback', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message_id: id, rating: r, question: (meta && meta.question) || '', answer: text || '', sources: (meta && meta.sources) || [] })
            }).catch(()=>{});
          }
          up.onclick = () => setRating(1);
          down.onclick = () => setRating(-1);
          footer.appendChild(up);
          footer.appendChild(down);

          b.appendChild(footer);
        }
        chatLogEl.appendChild(wrap);
        chatLogEl.scrollTop = chatLogEl.scrollHeight;
      }

      async function fetchJson(url, opts) {
        const res = await fetch(url, opts);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || JSON.stringify(data) || res.statusText);
        return data;
      }

      function renderItems() {
        // items are filtered server-side by q for full-text search
        const items = allItems;
        itemsEl.innerHTML = '';
        if (!items.length) {
          itemsEl.innerHTML = '<div class="muted">No matches.</div>';
          return;
        }
        for (const it of items) {
          const div = document.createElement('div');
          div.className = 'card';
          div.innerHTML =
            '<div class="row1">' +
              '<div class="sid">' + esc(it.suttaid) + '</div>' +
              (it.has_commentary ? '<div class="tag">commentary</div>' : '<div class="tag">sutta</div>') +
            '</div>' +
            '<div class="title">' + esc(it.title || '') + '</div>';
          div.onclick = () => selectItem(it.suttaid);
          itemsEl.appendChild(div);
        }
      }

      async function loadItems() {
        setLeftStatus('Loading…');
        try {
          const q = (filterEl.value || '').trim();
          const url = q ? ('/api/items?q=' + encodeURIComponent(q)) : '/api/items';
          const data = await fetchJson(url);
          allItems = data.items || [];
          setLeftStatus('Loaded ' + allItems.length + ' item(s).');
          renderItems();
          await refreshIndexStatus();
        } catch (e) {
          setLeftStatus('Error: ' + (e?.message || e));
        }
      }

      async function refreshIndexStatus() {
        try {
          const data = await fetchJson('/api/index_status');
          const mode = (data.mode || '').toString();
          const brandEl = document.getElementById('brandName');
          if (brandEl) {
            brandEl.textContent = mode === 'vertex' ? 'Dama — AN1 (Vertex)' : 'Dama — AN1 (local)';
          }
          const line = data.exists
            ? ('ready (' + data.count + ' chunks' + (mode ? ', ' + mode : '') + ')')
            : ('missing (rebuild)' + (mode ? ' · ' + mode : ''));
          setIdxStatus(line);
          const tip = [
            data.build,
            data.mode,
            data.embedding_model,
            data.bundle_gcs,
            data.bundle_path,
            data.error,
            data.an1_path,
            data.persist_dir
          ].filter(Boolean).join('\\n');
          idxStatusEl.title = tip || '';
        } catch {
          setIdxStatus('unknown');
          idxStatusEl.title = '';
        }
      }

      async function selectItem(suttaid) {
        selectedId = suttaid;
        readingMetaEl.textContent = 'suttaid: ' + suttaid;
        readingEmptyEl.style.display = 'none';
        readingBodyEl.style.display = 'block';
        suttaTextEl.textContent = '';
        commTextEl.textContent = '';
        try {
          const data = await fetchJson('/api/item/' + encodeURIComponent(suttaid));
          const cid = (data.commentary_id || '').toString().trim();
          readingMetaEl.textContent = cid ? ('suttaid: ' + suttaid + ' · commentary_id: ' + cid) : ('suttaid: ' + suttaid);
          suttaTextEl.textContent = data.sutta || '';
          commTextEl.textContent = data.commentry || '(no commentary)';
        } catch (e) {
          suttaTextEl.textContent = 'Error: ' + (e?.message || e);
          commTextEl.textContent = '';
        }
      }

      async function rebuild() {
        rebuildBtn.disabled = true;
        reloadBtn.disabled = true;
        setLeftStatus('Rebuilding index…');
        try {
          const data = await fetchJson('/api/build', { method: 'POST' });
          setLeftStatus('Index rebuilt. collection_count=' + data.collection_count);
        } catch (e) {
          setLeftStatus('Rebuild error: ' + (e?.message || e));
        }
        await refreshIndexStatus();
        rebuildBtn.disabled = false;
        reloadBtn.disabled = false;
      }

      async function ask() {
        const question = (qEl.value || '').trim();
        if (!question) return;

        askBtn.disabled = true;
        rebuildBtn.disabled = true;
        reloadBtn.disabled = true;
        quotesEl.innerHTML = '';
        setChatStatus('Searching…');
        appendMsg('user', question);
        qEl.value = '';

        try {
          const data = await fetchJson('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
          });
          setChatStatus('Done.');
          const chunks = data.chunks || [];
          const sidKinds = [];
          for (const c of chunks) {
            const sid = (c.suttaid || '').toString().trim();
            const kind = (c.kind || '').toString().trim() || 'unknown';
            const cid = (c.commentary_id || '').toString().trim();
            if (!sid) continue;
            const key = sid + ':' + kind + (cid ? ':' + cid : '');
            if (!sidKinds.includes(key)) sidKinds.push(key);
          }
          appendMsg('asst', (data.answer || '(no answer)'), { question, sources: sidKinds });

          // Chat mode: don't display retrieved passages; show only a compact sources list.
          quotesEl.innerHTML = '';
        } catch (e) {
          setChatStatus('Error: ' + (e?.message || e));
        }

        askBtn.disabled = false;
        rebuildBtn.disabled = false;
        reloadBtn.disabled = false;
      }

      reloadBtn.addEventListener('click', loadItems);
      rebuildBtn.addEventListener('click', rebuild);
      askBtn.addEventListener('click', ask);
      let _filterTimer = null;
      filterEl.addEventListener('input', () => {
        if (_filterTimer) clearTimeout(_filterTimer);
        _filterTimer = setTimeout(loadItems, 120);
      });
      qEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          ask();
        }
      });

      loadItems();
    </script>
  </body>
</html>
"""


@app.get("/api/items")
def api_items(q: str = Query(default="")) -> JSONResponse:
    items = _load_items()
    qn = (q or "").strip().lower()
    filtered: List[ItemDetail] = []
    if qn:
        for i in items:
            blob = f"{i.suttaid}\n{i.sutta}\n{i.commentry}\n{i.commentary_id}".lower()
            if qn in blob:
                filtered.append(i)
    else:
        filtered = items

    out = [
        ItemSummary(
            suttaid=i.suttaid,
            title=_item_title(i),
            has_commentary=bool((i.commentry or "").strip()),
        ).model_dump()
        for i in filtered
    ]
    _dbg("H4", "an1_app.py:api_items", "Serve items", {"count": len(out), "q": (q or "")[:80]})
    return JSONResponse({"items": out})


@app.get("/api/item/{suttaid}", response_model=ItemDetail)
def api_item(suttaid: str) -> ItemDetail:
    items = _load_items()
    for it in items:
        if it.suttaid == suttaid:
            return it
    raise HTTPException(status_code=404, detail=f"Unknown suttaid: {suttaid}")


@app.get("/api/index_status")
def api_index_status() -> JSONResponse:
    if vx.an1_vertex_enabled():
        try:
            b = vx.ensure_bundle_loaded(PERSIST_DIR)
            n = len(vx.bundle_chunk_rows(b))
            meta = vx.bundle_meta_for_status(b)
            return JSONResponse(
                {
                    "exists": n > 0,
                    "count": n,
                    "mode": "vertex",
                    "persist_dir": str(PERSIST_DIR.resolve()),
                    "an1_path": str(AN1_PATH.resolve()),
                    "build": AN1_APP_BUILD,
                    "bundle_gcs": os.environ.get("AN1_VERTEX_BUNDLE_GCS_URI", "").strip(),
                    "bundle_path": os.environ.get("AN1_VERTEX_BUNDLE_PATH", "").strip(),
                    "embedding_model": meta.get("embedding_model") or "",
                    "bundle_format": meta.get("format") or "",
                }
            )
        except Exception as e:
            return JSONResponse(
                {
                    "exists": False,
                    "count": 0,
                    "mode": "vertex",
                    "persist_dir": str(PERSIST_DIR.resolve()),
                    "an1_path": str(AN1_PATH.resolve()),
                    "build": AN1_APP_BUILD,
                    "error": str(e),
                }
            )

    exists = PERSIST_DIR.exists()
    count = 0
    if exists:
        try:
            col = _get_collection()
            count = int(col.count())
        except Exception:
            count = 0
    return JSONResponse(
        {
            "exists": bool(exists and count > 0),
            "count": count,
            "mode": "local_chroma",
            "persist_dir": str(PERSIST_DIR.resolve()),
            "an1_path": str(AN1_PATH.resolve()),
            "build": AN1_APP_BUILD,
        }
    )


@app.post("/api/build", response_model=BuildResponse)
def api_build() -> BuildResponse:
    global _items_cache
    with _lock:
        _dbg("H2", "an1_app.py:api_build", "Build requested")
        try:
            if vx.an1_vertex_enabled():
                from an1_build_vertex_bundle import write_bundle

                out = PERSIST_DIR / "an1_vertex_bundle.json"
                gcs_uri = os.environ.get("AN1_VERTEX_BUNDLE_GCS_URI", "").strip()
                write_bundle(out, upload_gs_uri=gcs_uri)
                vx.invalidate_bundle_cache()
                b = vx.ensure_bundle_loaded(PERSIST_DIR)
                _items_cache = None
                return BuildResponse(ok=True, collection_count=len(vx.bundle_chunk_rows(b)))

            an1_build.main()
            # JSON list is cached separately from Chroma; clear so reading pane / items match an1.json after rebuild.
            _items_cache = None
            col = _get_collection()
            return BuildResponse(ok=True, collection_count=int(col.count()))
        except Exception as e:
            _dbg("H2", "an1_app.py:api_build", "Build failed", {"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
def api_query(req: QueryRequest) -> QueryResponse:
    with _lock:
        # UI no longer shows controls; keep API-compatible, but default to LLM answers.
        req.use_llm = True
        req.k = 8
        _dbg("H4", "an1_app.py:api_query", "Query", {"k": req.k, "use_llm": req.use_llm, "question_len": len(req.question)})

        if vx.an1_vertex_enabled():
            try:
                bundle = vx.ensure_bundle_loaded(PERSIST_DIR)
            except Exception as e:
                _dbg("H2", "an1_app.py:api_query", "Vertex bundle missing", {"error": str(e)})
                raise HTTPException(
                    status_code=400,
                    detail=f"Vertex bundle not available: {e}. Rebuild (POST /api/build) or set AN1_VERTEX_BUNDLE_GCS_URI.",
                ) from e
            try:
                chunks = _retrieve_vertex(bundle, req.question, req.k)
            except Exception as e:
                _dbg("H2", "an1_app.py:api_query", "Vertex retrieval failed", {"error": str(e)})
                raise HTTPException(status_code=502, detail=f"Vertex retrieval failed: {e}") from e

            used_llm = False
            answer = ""
            if req.use_llm:
                used_llm = True
                llm_chunks = chunks[:10]
                try:
                    _dbg(
                        "H3",
                        "an1_app.py:api_query",
                        "Calling Vertex Gemini",
                        {"chunks": len(llm_chunks), "model": vx.chat_model_name()},
                    )
                    answer = _call_llm(req.question, llm_chunks)
                except Exception as e:
                    _dbg("H3", "an1_app.py:api_query", "Vertex LLM failed", {"error": str(e)})
                    raise HTTPException(status_code=502, detail=f"Vertex LLM call failed: {e}") from e

            return QueryResponse(chunks=chunks, answer=answer, used_llm=used_llm)

        if not PERSIST_DIR.exists():
            _dbg("H2", "an1_app.py:api_query", "Index missing", {"persist_dir": str(PERSIST_DIR)})
            raise HTTPException(status_code=400, detail="Index not found. Rebuild first.")

        try:
            col = _get_collection()
        except Exception as e:
            _dbg("H2", "an1_app.py:api_query", "Open collection failed", {"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Failed to open collection: {e}")

        embed_model = _get_embed_model()
        chunks = _retrieve(embed_model, col, req.question, req.k)

        used_llm = False
        answer = ""
        if req.use_llm:
            used_llm = True
            llm_chunks = chunks[:10]
            try:
                _dbg("H3", "an1_app.py:api_query", "Calling Ollama", {"chunks": len(llm_chunks), "model": OLLAMA_MODEL})
                answer = _call_llm(req.question, llm_chunks)
            except Exception as e:
                _dbg("H3", "an1_app.py:api_query", "Ollama failed", {"error": str(e)})
                raise HTTPException(status_code=502, detail=f"Ollama LLM call failed: {e}")

        return QueryResponse(chunks=chunks, answer=answer, used_llm=used_llm)


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1)
    rating: int = Field(ge=-1, le=1)
    question: str = ""
    answer: str = ""
    sources: List[str] = []


@app.post("/api/feedback")
def api_feedback(req: FeedbackRequest) -> JSONResponse:
    _dbg(
        "H4",
        "an1_app.py:api_feedback",
        "Feedback",
        {"message_id": req.message_id, "rating": req.rating, "sources": req.sources[:20]},
    )
    return JSONResponse({"ok": True})

