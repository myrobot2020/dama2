import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
import requests

import an1_build_index as an1_build
import an1_vertex_core as vx
from an1_build_index import (
    _commentary_body,
    _commentary_id,
    _extract_records_fallback,
    _normalize_suttaid_an2,
    _parse_json_lenient,
)


BASE_DIR = Path(__file__).resolve().parent
GLOBAL_CONVERSATIONS_DIR = BASE_DIR / "global_chat_history"
CONVERSATIONS_INDEX = "conversations_index.json"
AN1_PATH = an1_build.AN1_PATH
PERSIST_DIR = an1_build.PERSIST_DIR
COLLECTION_NAME = an1_build.COLLECTION_NAME
AN2_PATH = an1_build.AN2_PATH
PERSIST_AN2_DIR = an1_build.PERSIST_AN2_DIR
COLLECTION_AN2 = an1_build.COLLECTION_AN2
GONG_MP3_PATH = BASE_DIR / "freesound_community-gong-79191.mp3"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:instruct"

# Bumped when RAG/LLM behavior changes (shown in GET /api/index_status so you know the server reloaded).
AN1_APP_BUILD = "2026-04-12-gcp-parity-check"

# Short reply for greetings / smalltalk — the citation-only LLM prompt is wrong for this (it still sees random retrieval).
_CHAT_ONLY_REPLY = (
    "Hi there. Ask anything about the AN1 or AN2 teachings (suttas, commentary, or themes), "
    "or click an (AN …) or (cAN …) citation in an answer to open the full text in Reference."
)

_CHAT_SUBSTANTIVE_HINT = re.compile(
    r"\b(?:"
    r"what|why|when|where|who|whom|which|"
    r"how\s+(?:do|does|did|can|could|should|would|is|are|was|were|many|much|long|far|often|come|work)"
    r"|explain|meaning|define|sutta|suttas|dhamma|buddha|"
    r"noble|eightfold|path|meditation|jhana|karma|rebirth|nibbana|nirvana|teach|teaching|"
    r"compare|difference|according|chapter|verse|quote|paraphrase|discourse|canon"
    r"|an\d+\s*\.|c?an\s*\d"
    r")\b",
    re.I,
)

_CHAT_GREETING_PHRASES = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "howdy",
        "greetings",
        "hi there",
        "hey there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "ty",
        "ok",
        "okay",
        "ok thanks",
        "okay thanks",
        "thanks a lot",
        "thank you so much",
        "tysm",
        "bye",
        "goodbye",
        "see you",
        "cya",
        "later",
        "whats up",
        "what's up",
        "wassup",
        "how are you",
        "how are u",
        "how r u",
        "nice to meet you",
        "morning",
        "evening",
        "afternoon",
    }
)


def _normalize_chat_probe(q: str) -> str:
    t = (q or "").strip().lower()
    t = re.sub(r"[!?.]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _is_chat_only_message(question: str) -> bool:
    q = (question or "").strip()
    if not q or len(q) > 140:
        return False
    if re.search(r"\d", q):
        return False
    low = q.lower()
    if "?" in q and not re.search(r"how\s+are\s+you", low):
        return False
    if _CHAT_SUBSTANTIVE_HINT.search(low):
        return False
    norm = _normalize_chat_probe(q)
    if norm in _CHAT_GREETING_PHRASES:
        return True
    if len(norm) <= 10:
        tokens = re.findall(r"[a-z']+", norm)
        if not tokens:
            return False
        small = frozenset(
            {
                "hi",
                "hey",
                "hello",
                "yo",
                "sup",
                "there",
                "you",
                "thanks",
                "thank",
                "thx",
                "ty",
                "ok",
                "okay",
                "yes",
                "no",
                "hm",
                "hmm",
                "cool",
                "nice",
                "great",
                "good",
                "well",
                "morning",
                "afternoon",
                "evening",
                "day",
                "bye",
                "welcome",
                "back",
                "again",
                "a",
                "the",
                "to",
                "so",
                "very",
                "just",
                "im",
                "i",
                "am",
                "are",
                "is",
                "it",
                "me",
                "my",
                "we",
            }
        )
        if len(tokens) <= 5 and all(t in small for t in tokens):
            return True
    return False

DEBUG_LOG_PATH = BASE_DIR / "debug-655121.log"


def _is_uuid_str(s: str) -> bool:
    try:
        uuid.UUID(str(s).strip())
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _conversation_jsonl_path(conversation_id: str) -> Path:
    cid = str(conversation_id or "").strip().lower()
    if not _is_uuid_str(cid):
        raise HTTPException(status_code=400, detail="invalid conversation id")
    GLOBAL_CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    p = (GLOBAL_CONVERSATIONS_DIR / f"{cid}.jsonl").resolve()
    base = GLOBAL_CONVERSATIONS_DIR.resolve()
    if p.parent != base:
        raise HTTPException(status_code=400, detail="invalid path")
    return p


def _conversations_index_path() -> Path:
    GLOBAL_CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return GLOBAL_CONVERSATIONS_DIR / CONVERSATIONS_INDEX


def _read_conversation_index() -> List[Dict[str, Any]]:
    path = _conversations_index_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_conversation_index(rows: List[Dict[str, Any]]) -> None:
    path = _conversations_index_path()
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _conversation_has_valid_turn(path: Path) -> bool:
    if not path.is_file():
        return False
    for m in _read_jsonl_messages(path):
        q = str(m.get("q") or "").strip()
        a = str(m.get("a") or "").strip()
        if q and a and a != "(no answer)":
            return True
    return False


def _prune_empty_conversations() -> None:
    """Drop index rows / jsonl with no stored Q+A; merge duplicate index ids; re-write index."""
    GLOBAL_CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    idx_rows = _read_conversation_index()
    by_cid: Dict[str, List[Dict[str, Any]]] = {}
    for row in idx_rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        cid = str(row["id"]).strip().lower()
        if not _is_uuid_str(cid):
            continue
        by_cid.setdefault(cid, []).append(row)

    all_cids = set(by_cid.keys())
    if GLOBAL_CONVERSATIONS_DIR.is_dir():
        for p in GLOBAL_CONVERSATIONS_DIR.glob("*.jsonl"):
            stem = p.stem.strip().lower()
            if _is_uuid_str(stem):
                all_cids.add(stem)

    out: List[Dict[str, Any]] = []
    for cid in all_cids:
        try:
            path = _conversation_jsonl_path(cid)
        except HTTPException:
            continue
        if not _conversation_has_valid_turn(path):
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
            continue

        msgs = _read_jsonl_messages(path)
        msg_ts = max((int(m.get("ts") or 0) for m in msgs), default=0)
        mtime = int(path.stat().st_mtime * 1000) if path.is_file() else 0
        best_ts = max(msg_ts, mtime)
        for r in by_cid.get(cid, []):
            best_ts = max(best_ts, int(r.get("updated_ts") or 0))

        title = "Chat"
        for r in by_cid.get(cid, []):
            t = str(r.get("title") or "").strip()
            if t and t != "New chat":
                title = t
                break
        if title == "Chat":
            for r in by_cid.get(cid, []):
                t = str(r.get("title") or "").strip()
                if t:
                    title = t
                    break

        out.append({"id": str(uuid.UUID(cid)), "title": title, "updated_ts": best_ts})

    out.sort(key=lambda r: -int(r.get("updated_ts") or 0))
    _write_conversation_index(out)


def _list_conversations_merged() -> List[Dict[str, Any]]:
    """Index only (pruned); sorted by updated_ts desc."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in _read_conversation_index():
        if not isinstance(row, dict) or not row.get("id"):
            continue
        cid = str(row["id"]).strip().lower()
        if not _is_uuid_str(cid):
            continue
        by_id[cid] = {
            "id": cid,
            "title": str(row.get("title") or "Chat").strip() or "Chat",
            "updated_ts": int(row.get("updated_ts") or 0),
        }
    return sorted(by_id.values(), key=lambda x: -int(x.get("updated_ts") or 0))


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
    book: str = Field(
        default="all",
        description="Legacy: an1/an2/all. Local Chroma always searches AN1+AN2 together.",
    )

    @field_validator("book")
    @classmethod
    def _norm_query_book(cls, v: str) -> str:
        b = (v or "all").strip().lower()
        return b if b in ("an1", "an2", "all") else "all"


class Chunk(BaseModel):
    source: str = ""
    suttaid: str = ""
    commentary_id: str = ""
    kind: str = ""
    book: str = ""
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
    chain: Optional[Dict[str, Any]] = None


class ChatHistoryAppend(BaseModel):
    question: str = ""
    answer: str = ""
    latency_ms: int = Field(default=0, ge=0, le=3_600_000)


class ConversationCreate(BaseModel):
    title: str = Field(default="", max_length=240)


_lock = threading.RLock()
_embed_model: Optional[Any] = None
_reranker: Optional[Any] = None
_items_cache: Dict[str, List[ItemDetail]] = {}


def _norm_book(book: Optional[str]) -> str:
    b = (book or "an1").strip().lower()
    return b if b in ("an1", "an2") else "an1"


def _persist_dir_for_book(book: str) -> Path:
    return PERSIST_AN2_DIR if book == "an2" else PERSIST_DIR


def _collection_name_for_book(book: str) -> str:
    return COLLECTION_AN2 if book == "an2" else COLLECTION_NAME


def _get_collection(book: str = "an1") -> Any:
    import chromadb
    from chromadb.config import Settings

    b = _norm_book(book)
    client = chromadb.PersistentClient(
        path=str(_persist_dir_for_book(b)),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(_collection_name_for_book(b))


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
    book = ""
    if isinstance(meta, dict):
        src = str(meta.get("source") or "")
        sid = str(meta.get("suttaid") or "")
        cid = str(meta.get("commentary_id") or "")
        kind = str(meta.get("kind") or "")
        book = str(meta.get("book") or "").strip().lower()
        if book not in ("an1", "an2") and src:
            book = "an2" if "an2" in src.lower() else ""
        if book not in ("an1", "an2"):
            book = ""
    return Chunk(
        source=src,
        suttaid=sid,
        commentary_id=cid,
        kind=kind,
        book=book,
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


def _retrieve_merged_local(embed_model: Any, query: str, k: int) -> List[Chunk]:
    """Retrieve from AN1 + AN2 Chroma with roughly equal budget, then merge rank + pair-inject."""
    k1 = max(1, (k + 1) // 2)
    k2 = max(1, k - k1)
    out1: List[Chunk] = []
    out2: List[Chunk] = []
    if PERSIST_DIR.exists():
        try:
            col1 = _get_collection("an1")
            out1 = [c.model_copy(update={"book": "an1"}) for c in _retrieve(embed_model, col1, query, k1)]
        except Exception:
            pass
    if PERSIST_AN2_DIR.exists():
        try:
            col2 = _get_collection("an2")
            out2 = [c.model_copy(update={"book": "an2"}) for c in _retrieve(embed_model, col2, query, k2)]
        except Exception:
            pass
    merged = _dedupe_chunks_keep_order(out1 + out2)
    merged.sort(
        key=lambda c: (
            -_lexical_score(query, c.text),
            c.distance if c.distance is not None else 1e9,
        )
    )
    pool = merged
    seeded = _vertex_quota_pick(merged[: max(k * 4, 20)], k)
    top = _inject_pair_both_ways(seeded, pool, cap=14)
    return _dedupe_chunks_keep_order(top)[:14]


def _vertex_bundle_row_meta(row: Dict[str, Any]) -> Dict[str, str]:
    book = str(row.get("book") or "").strip().lower()
    if book not in ("an1", "an2"):
        src = str(row.get("source") or "")
        book = "an2" if "an2" in src.lower() else "an1"
    return {
        "source": str(row.get("source") or ""),
        "suttaid": str(row.get("suttaid") or ""),
        "commentary_id": str(row.get("commentary_id") or ""),
        "kind": str(row.get("kind") or ""),
        "book": book,
    }


def _vertex_quota_pick(sorted_chunks: List[Chunk], k: int) -> List[Chunk]:
    """Roughly equal picks per book (an1/an2), in global score order, then fill to k."""
    if k <= 0 or not sorted_chunks:
        return []
    books_present = sorted({((c.book or "an1").lower()) for c in sorted_chunks if (c.book or "an1").lower() in ("an1", "an2")})
    books_present = [b for b in books_present if b in ("an1", "an2")]
    if not books_present:
        books_present = ["an1"]
    n = len(books_present)
    base, rem = divmod(k, n)
    quota: Dict[str, int] = {b: base for b in books_present}
    for i in range(rem):
        quota[books_present[i]] += 1

    def _key(c: Chunk) -> Tuple[str, str, str, str, str]:
        return (
            c.source,
            c.suttaid,
            (c.kind or "").lower(),
            (c.book or "an1").lower(),
            c.text[:180],
        )

    picked: List[Chunk] = []
    seen: set = set()
    for c in sorted_chunks:
        if len(picked) >= k:
            break
        b = (c.book or "an1").lower()
        if b not in quota:
            b = books_present[0]
        ky = _key(c)
        if ky in seen:
            continue
        if quota.get(b, 0) <= 0:
            continue
        picked.append(c)
        seen.add(ky)
        quota[b] -= 1
    if len(picked) < k:
        for c in sorted_chunks:
            if len(picked) >= k:
                break
            ky = _key(c)
            if ky in seen:
                continue
            picked.append(c)
            seen.add(ky)
    return picked[:k]


def _retrieve_vertex(bundle: Dict[str, Any], query: str, k: int) -> List[Chunk]:
    """Same ranking heuristics as _retrieve, but vector search + lexical scan over a Vertex bundle (no Chroma)."""
    rows_all = vx.bundle_chunk_rows(bundle)
    if not rows_all:
        return []

    q_emb = vx.embed_texts_vertex([query])[0]
    out: List[Chunk] = []
    for row in rows_all:
        if not isinstance(row, dict):
            continue
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
                _vertex_bundle_row_meta(row),
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

    seen_keys = set((c.source, c.suttaid, c.book, c.text[:200]) for c in out)
    for t in merged_terms:
        tl = t.lower()
        for row in rows_all:
            if not isinstance(row, dict):
                continue
            doc = str(row.get("text") or "")
            if tl not in doc.lower():
                continue
            ch = _chunk_from_doc(doc, _vertex_bundle_row_meta(row), None)
            key = (ch.source, ch.suttaid, ch.book, ch.text[:200])
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

    seeded = _vertex_quota_pick(candidates, k)
    top = _inject_pair_both_ways(seeded, out, cap=14)

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


def _chunks_for_llm_paired_by_suttaid(
    chunks: List[Chunk], max_n: int = 10, *, include_chain: bool = False
) -> List[Chunk]:
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
        chn = [x for x in grp if (x.kind or "").lower() == "chain"]
        seq = sut[:1] + comm[:2]
        if include_chain and chn:
            seq = seq + chn[:1]
        for x in seq:
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
            "CITE_RULE: (AN …) must reflect this SUTTA text; paraphrase in chat, optional ≤25-word quote only if needed. "
            "This block has no TEACHER COMMENTARY — do not call commentary ideas a 'sutta quote'.\n"
        )
    elif kind == "commentary":
        hint = (
            "CITE_RULE: (cAN …) must reflect this TEACHER COMMENTARY; paraphrase in chat, optional ≤25-word quote only if needed. "
            "Do not attribute this wording to 'the sutta' or (AN …).\n"
        )
    elif kind == "chain":
        hint = (
            "CITE_RULE: kind 'chain' is a short list of conceptual links (not Buddha-worded scripture). "
            "Do not put CHAIN lines inside quotation marks as if they were sutta text. "
            "You may summarize the pairing in your own words and still cite (AN …) only for actual sutta content.\n"
        )
    else:
        hint = "CITE_RULE: Match (AN …)/(cAN …) to the correct SUTTA vs TEACHER COMMENTARY section.\n"
    bk = (getattr(c, "book", None) or "").strip()
    book_part = (f"book: {bk} | ") if bk else ""
    return (
        f"[{idx} | {book_part}suttaid: {c.suttaid or '-'} | commentary_id: {c.commentary_id or '-'} | kind: {c.kind or '-'}]\n"
        f"{hint}"
        f"{c.text}"
    )


def _llm_system_and_user_blocks(query: str, balanced: List[Chunk], *, book: str = "an1") -> Tuple[str, str]:
    numbered = "\n\n".join(_format_llm_passage(i + 1, c) for i, c in enumerate(balanced))
    if _norm_book(book) == "an2":
        system = (
            "You are a friendly, conversational assistant answering from AN2 (Numerical Discourses, book of twos) "
            "sutta text, teacher commentary, and optional CHAIN passages below.\n"
            "- kind 'commentary' = teacher notes (after TEACHER COMMENTARY:). Cite with (cAN …) using commentary_id.\n"
            "- kind 'sutta' = sutta text (after SUTTA:). Cite with (AN …) using suttaid from the header.\n"
            "- kind 'chain' = a short enumerated list of the two linked themes for that discourse (study aid). "
            "It is NOT verbatim scripture — do not quote CHAIN lines as the Buddha’s words.\n\n"
            "DISPLAY: The chat must stay short. Do NOT paste long sutta or commentary blocks — the app has "
            "clickable (AN …)/(cAN …) citations and a Reference panel with full text. Paraphrase clearly in a few "
            "sentences; optional at most one short phrase in quotation marks per cited id (roughly ≤25 words) only "
            "if essential; otherwise no block quotes.\n\n"
            "STRICT RULES:\n"
            "1. Prefer staying within ONE suttaid: sutta, commentary, and chain for the same id when present.\n"
            "2. Ground every (AN …) in the sutta passage for that id and every (cAN …) in the teacher commentary "
            "for that id, but explain in your own words; do not reproduce full discourses in the answer.\n"
            "3. Never put (cAN …) on sutta-only ideas or (AN …) on commentary-only ideas.\n"
            "4. You may describe the CHAIN (the two linked ideas) in plain language without pretending it is a sutta citation.\n"
            "5. If nothing in PASSAGES answers the question, say: "
            "\"The provided excerpts do not contain information to answer this question.\"\n"
            "6. Do NOT guess or use outside knowledge."
        )
        user = (
            f"Question: {query}\n\nPASSAGES (suttaid order; headers for you only):\n{numbered}\n\n"
            "Answer concisely with (AN …)/(cAN …) citations; use CHAIN only as conceptual context, not as quoted scripture."
        )
        return system, user

    system = (
        "You are a friendly, conversational assistant answering from AN1 sutta and teacher commentary "
        "passages below.\n"
        "- kind 'commentary' = teacher notes (text after TEACHER COMMENTARY:). Cite with (cAN …) using "
        "commentary_id from that passage’s header.\n"
        "- kind 'sutta' = sutta text (after SUTTA:). Cite with (AN …) using suttaid from that passage’s header.\n"
        "- kind 'chain' = a short enumerated pair of linked themes for that discourse (study aid). "
        "It is NOT verbatim scripture — do not quote CHAIN lines as the Buddha’s words.\n\n"
        "DISPLAY: The chat must stay short. Do NOT paste long sutta or commentary blocks — the app has "
        "clickable (AN …)/(cAN …) citations and a Reference panel with full text. Paraphrase the teaching in "
        "a few clear sentences; optional at most one short phrase in quotation marks per cited id "
        "(roughly ≤25 words) only if essential; otherwise no block quotes.\n\n"
        "STRICT RULES:\n"
        "1. PASSAGES are grouped by suttaid: sutta for that id appears before teacher commentary for the "
        "same id. Prefer sutta, teacher notes, and chain from the SAME suttaid in one answer. Do NOT pair "
        "(AN 1.2) with (cAN 1.5.6) unless you write one clear sentence explaining why two discourses are "
        "needed; otherwise stay within one suttaid pair.\n"
        "2. For your main example, explain the sutta point in your own words and add (AN …); then summarize "
        "the teacher’s angle and add (cAN …) when commentary is in PASSAGES — both grounded in that suttaid.\n"
        "3. Never put (cAN …) on sutta-only ideas or (AN …) on commentary-only ideas.\n"
        "4. You may describe the CHAIN (the two linked ideas) in plain language without pretending it is a sutta citation.\n"
        "5. If you use quotation marks before (AN …), the quoted words must appear verbatim under SUTTA: in a "
        "kind 'sutta' block for that id (keep the quote very short). If you use quotation marks before (cAN …), "
        "they must be verbatim from TEACHER COMMENTARY: in a kind 'commentary' block. Never call teacher wording "
        "the sutta or tag it (AN …).\n"
        "6. If PASSAGES include a sutta about water, pools, oysters, etc., do not claim excerpts omit water.\n"
        "7. If only commentary or only sutta appears for the relevant suttaid, say what is missing briefly.\n"
        "8. Do not write 'Excerpt' / internal passage numbers from headers.\n"
        "9. If the question uses 'this' / 'it' without a clear topic, ask ONE short clarifying question.\n"
        "10. If nothing in PASSAGES answers the question, say: "
        "\"The provided excerpts do not contain information to answer this question.\"\n"
        "11. Do NOT guess or use outside knowledge."
    )
    user = (
        f"Question: {query}\n\nPASSAGES (grouped by suttaid: sutta for an id, then teacher notes for that id; "
        f"headers for you only — do not echo them):\n{numbered}\n\n"
        "Answer concisely with (AN …)/(cAN …) citations; use CHAIN only as conceptual context, not as quoted scripture; "
        "avoid unrelated cross-citations; never call teacher notes the sutta."
    )
    return system, user


def _llm_vertex_mixed_blocks(query: str, balanced: List[Chunk]) -> Tuple[str, str]:
    numbered = "\n\n".join(_format_llm_passage(i + 1, c) for i, c in enumerate(balanced))
    system = (
        "You are a friendly, conversational assistant answering from AN1 and/or AN2 (Anguttara Nikāya) "
        "passages below. Each block header may include book: an1 or an2.\n"
        "- kind 'sutta' / 'commentary': cite with (AN …) and (cAN …); ground each citation in the matching passage.\n"
        "- kind 'chain': conceptual link labels (study aid), not scripture; never quote CHAIN lines as the Buddha’s words.\n\n"
        "DISPLAY: Keep the chat answer short. Do NOT paste long sutta or commentary blocks — citations open full "
        "text in the Reference panel. Paraphrase; optional at most one short quoted phrase per cited id "
        "(roughly ≤25 words) only if essential.\n\n"
        "STRICT RULES:\n"
        "1. Prefer one suttaid (and one book) for your main example when possible.\n"
        "2. Never put (cAN …) on sutta-only ideas or (AN …) on commentary-only ideas.\n"
        "3. If nothing in PASSAGES answers the question, say: "
        "\"The provided excerpts do not contain information to answer this question.\"\n"
        "4. Do NOT guess or use outside knowledge."
    )
    user = (
        f"Question: {query}\n\nPASSAGES (headers for you only):\n{numbered}\n\n"
        "Answer concisely with correct (AN …)/(cAN …) citations; use CHAIN only as non-quoted context when present."
    )
    return system, user


def _call_llm_vertex(query: str, chunks: List[Chunk]) -> str:
    include_chain = any((c.kind or "").lower() == "chain" for c in chunks)
    balanced = _chunks_for_llm_paired_by_suttaid(chunks, max_n=10, include_chain=include_chain)
    if not balanced:
        return "No passages were retrieved for this question. Try Rebuild or rephrase."
    has_an2 = any((c.book or "").lower() == "an2" for c in balanced)
    if has_an2:
        system, user = _llm_vertex_mixed_blocks(query, balanced)
    else:
        system, user = _llm_system_and_user_blocks(query, balanced, book="an1")
    return _sanitize_chat_answer_light(vx.gemini_generate(system, user, temperature=0.2))


def _call_llm(query: str, chunks: List[Chunk], *, book: str = "an1") -> str:
    if vx.an1_vertex_enabled():
        return _call_llm_vertex(query, chunks)

    include_chain = any((c.kind or "").lower() == "chain" for c in chunks)
    balanced = _chunks_for_llm_paired_by_suttaid(
        chunks, max_n=10, include_chain=include_chain
    )
    if not balanced:
        return "No passages were retrieved for this question. Try Rebuild or rephrase."
    has_an2 = any((c.book or "").lower() == "an2" for c in balanced)
    if has_an2:
        system, user = _llm_vertex_mixed_blocks(query, balanced)
    else:
        b = _norm_book(book)
        system, user = _llm_system_and_user_blocks(query, balanced, book=b if b == "an2" else "an1")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _sanitize_chat_answer_light(_ollama_chat(messages, num_ctx=8192, timeout=300))


def _merged_item_details() -> List[ItemDetail]:
    seen: Set[str] = set()
    out: List[ItemDetail] = []
    for it in _load_items("an1"):
        out.append(it)
        seen.add(it.suttaid)
    for it in _load_items("an2"):
        if it.suttaid not in seen:
            out.append(it)
            seen.add(it.suttaid)
    return out


def _find_item_by_suttaid(suttaid: str) -> Optional[ItemDetail]:
    sid = (suttaid or "").strip()
    if not sid:
        return None
    for it in _load_items("an1"):
        if it.suttaid == sid:
            return it
    for it in _load_items("an2"):
        if it.suttaid == sid:
            return it
    return None


def _find_item_by_commentary_id(commentary_id: str) -> Optional[ItemDetail]:
    want = (commentary_id or "").strip()
    if not want:
        return None
    for it in _load_items("an1"):
        if (it.commentary_id or "").strip() == want:
            return it
    for it in _load_items("an2"):
        if (it.commentary_id or "").strip() == want:
            return it
    return None


def _invalidate_items_cache(book: Optional[str] = None) -> None:
    global _items_cache
    if book is None:
        _items_cache.clear()
    else:
        _items_cache.pop(_norm_book(book), None)


def _load_items(book: str = "an1") -> List[ItemDetail]:
    global _items_cache
    b = _norm_book(book)
    if b in _items_cache:
        return _items_cache[b]

    path = AN2_PATH if b == "an2" else AN1_PATH
    if not path.exists():
        _dbg("H1", "an1_app.py:_load_items", "JSON missing", {"book": b, "path": str(path)})
        _items_cache[b] = []
        return _items_cache[b]

    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        data = _parse_json_lenient(raw)
    except Exception as e:
        _dbg(
            "H1",
            "an1_app.py:_load_items",
            "JSON parse failed",
            {"book": b, "error": str(e), "bytes": len(raw), "prefix": raw[:200]},
        )
        try:
            data = _extract_records_fallback(raw)
            _dbg("H1", "an1_app.py:_load_items", "Fallback extracted records", {"count": len(data)})
        except Exception as e2:
            _dbg("H1", "an1_app.py:_load_items", "Fallback parse failed", {"error": str(e2)})
            _items_cache[b] = []
            return _items_cache[b]

    items: List[ItemDetail] = []
    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            if b == "an2":
                sid = _normalize_suttaid_an2(obj.get("sutta_id") or obj.get("suttaid"))
                sutta = str(obj.get("sutta") or "").strip()
                comm = _commentary_body(obj)
                cid = ("c" + sid) if sid else ""
                ch = obj.get("chain")
                chain_dict = ch if isinstance(ch, dict) else None
                if not sid or not sutta:
                    continue
                items.append(
                    ItemDetail(
                        suttaid=sid,
                        sutta=sutta,
                        commentry=comm,
                        commentary_id=cid,
                        chain=chain_dict,
                    )
                )
            else:
                sid = str(obj.get("suttaid") or "").strip()
                sutta = str(obj.get("sutta") or "").strip()
                comm = _commentary_body(obj)
                cid = _commentary_id(obj)
                ch = obj.get("chain")
                chain_dict = ch if isinstance(ch, dict) else None
                if not sid or not sutta:
                    continue
                items.append(
                    ItemDetail(
                        suttaid=sid,
                        sutta=sutta,
                        commentry=comm,
                        commentary_id=cid,
                        chain=chain_dict,
                    )
                )
    else:
        _dbg("H1", "an1_app.py:_load_items", "Unexpected JSON top-level type", {"type": str(type(data))})

    _items_cache[b] = items
    _dbg("H1", "an1_app.py:_load_items", "Loaded items", {"book": b, "count": len(items)})
    return items


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


@app.get("/assets/gong.mp3")
def serve_gong_mp3() -> FileResponse:
    if not GONG_MP3_PATH.is_file():
        raise HTTPException(status_code=404, detail="gong asset missing")
    return FileResponse(
        GONG_MP3_PATH,
        media_type="audio/mpeg",
        filename="gong.mp3",
    )


@app.get("/")
def home() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="Cache-Control" content="no-store, max-age=0" />
    <meta http-equiv="Pragma" content="no-cache" />
    <title>Dama — AN1 RAG (local)</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js" crossorigin="anonymous"></script>
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
      .rightmeta { display:flex; gap:10px; align-items:center; color: var(--muted); font-size: 12px; flex-wrap: wrap; justify-content: flex-end; }
      /* Link pairs (reference panel); violet so it is not confused with sutta chain labels in data */
      .reading .chainFlow {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        gap: 0;
        margin-top: 6px;
      }
      .reading .chainNode {
        border: 1px solid rgba(124, 92, 255, 0.32);
        border-left: 3px solid rgba(129, 140, 248, 0.75);
        border-radius: 8px;
        padding: 7px 10px;
        background: rgba(12, 18, 32, 0.92);
        font-size: 12px;
        line-height: 1.38;
        color: #dbe6ff;
      }
      .reading .chainFlowSep {
        align-self: center;
        padding: 3px 0;
        font-size: 12px;
        font-weight: 600;
        line-height: 1;
        color: #a5b4fc;
        text-shadow: none;
      }
      .layout {
        height: calc(100% - 54px);
        display: flex;
        flex-direction: row;
        align-items: stretch;
        gap: 10px;
        padding: 12px;
        min-height: 0;
      }
      .convCol {
        flex: 2 1 0;
        min-width: 140px;
        max-width: 28%;
        min-height: 0;
        transition: flex-basis 0.22s ease, min-width 0.22s ease, max-width 0.22s ease;
      }
      .convCol.collapsed {
        flex: 0 0 42px;
        min-width: 42px;
        max-width: 42px;
      }
      .convCol.collapsed .convBody { display: none !important; }
      .convCol.collapsed .convTitle { display: none; }
      .convCol.collapsed .hdr { justify-content: center; padding-left: 4px; padding-right: 4px; }
      .chatCol {
        flex: 5 1 0;
        min-width: 200px;
        min-height: 0;
      }
      .readGrip {
        flex: 0 0 12px;
        width: 12px;
        margin: 0 -4px;
        cursor: col-resize;
        align-self: stretch;
        display: flex;
        align-items: stretch;
        justify-content: center;
        z-index: 3;
        user-select: none;
      }
      .readGrip::before {
        content: '';
        width: 4px;
        border-radius: 3px;
        background: rgba(255,255,255,.12);
        margin: 16px 0;
        align-self: stretch;
      }
      .readGrip:hover::before { background: rgba(124,92,255,.45); }
      .readingCol {
        flex: 3 1 0;
        min-width: 220px;
        max-width: 55vw;
        min-height: 0;
      }
      .layout.resizing { cursor: col-resize; }
      .layout.resizing * { cursor: col-resize !important; user-select: none !important; }
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
      .reading .pastConv {
        margin-top: 8px;
        max-height: 260px;
        overflow-y: auto;
        border: 1px solid var(--border2);
        border-radius: 12px;
        background: rgba(0,0,0,.14);
        padding: 6px 8px;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .reading details.pastTurn {
        border: 1px solid var(--border2);
        border-radius: 10px;
        background: rgba(0,0,0,.18);
        overflow: hidden;
      }
      .reading details.pastTurn > summary {
        list-style: none;
        cursor: pointer;
        padding: 8px 10px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 10px;
        font-size: 12px;
        line-height: 1.35;
        color: #dbe6ff;
        user-select: none;
      }
      .reading details.pastTurn > summary::-webkit-details-marker { display: none; }
      .reading details.pastTurn > summary::before {
        content: '▸';
        flex: 0 0 auto;
        color: var(--muted);
        margin-right: 6px;
        font-size: 10px;
        margin-top: 2px;
      }
      .reading details.pastTurn[open] > summary::before { content: '▾'; }
      .reading .pastSumMain { flex: 1; min-width: 0; word-break: break-word; }
      .reading .pastSumMeta { flex: 0 0 auto; color: var(--muted); font-size: 11px; white-space: nowrap; }
      .reading details.pastTurn .pastBody {
        padding: 0 10px 10px 28px;
        border-top: 1px solid var(--border2);
        background: rgba(0,0,0,.12);
      }
      .reading .pastQ { color: #dbe6ff; white-space: pre-wrap; word-break: break-word; margin-top: 8px; font-size: 12px; line-height: 1.4; }
      .reading .pastA { color: #dbe6ff; white-space: pre-wrap; word-break: break-word; margin-top: 8px; font-size: 12px; line-height: 1.4; opacity: .96; }
      .muted { color: var(--muted); font-size: 12px; }
      .chatArea { display:flex; flex-direction: column; gap: 8px; min-height: 0; }
      .chatLog { flex: 1; min-height: 0; overflow: auto; padding-bottom: 2px; }
      textarea { resize: none; min-height: 44px; max-height: 180px; line-height: 1.35; overflow-y: auto; }
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
      .msg.asst .bubble { background: rgba(0,0,0,.14); white-space: normal; }
      .msg.asst .bubble small.cite { font-size: 0.76em; color: var(--muted); font-weight: 500; }
      a.cite.cite-open,
      button.cite.cite-open {
        display: inline-block;
        font: inherit;
        font-size: 0.92em;
        color: #c4b5ff;
        background: transparent;
        border: none;
        padding: 0 1px;
        margin: 0;
        cursor: pointer;
        text-decoration: underline;
        text-underline-offset: 2px;
        vertical-align: baseline;
        border-radius: 4px;
      }
      .md a.cite.cite-open { color: #c4b5ff; }
      a.cite.cite-open:hover,
      button.cite.cite-open:hover { color: #ffe08a; background: rgba(255,224,138,.08); }
      .bubble.bubbleErr { border-color: rgba(255,92,122,.45); background: rgba(255,92,122,.08); }
      .md { font-size: 13px; line-height: 1.5; }
      .md > *:first-child { margin-top: 0; }
      .md > *:last-child { margin-bottom: 0; }
      .md p { margin: 0.55em 0; }
      .md h1, .md h2, .md h3, .md h4 { margin: 0.75em 0 0.35em; font-weight: 600; color: #e9eefc; }
      .md h1 { font-size: 1.15em; } .md h2 { font-size: 1.08em; } .md h3 { font-size: 1.02em; }
      .md ol { margin: 0.45em 0; padding-left: 1.35em; }
      .md ol > li { margin: 0.2em 0; }
      .md ul {
        list-style: none;
        margin: 0.45em 0;
        padding-left: 0;
        display: flex;
        flex-direction: column;
        gap: 0;
      }
      .md ul > li {
        margin: 0;
        list-style: none;
        border: 1px solid rgba(212,175,55,.28);
        border-radius: 8px;
        padding: 6px 10px;
        background: linear-gradient(165deg, rgba(212,175,55,.06), rgba(90,120,200,.04));
        color: #e8e4dc;
      }
      .md ul > li:not(:first-child)::before {
        content: '↓';
        display: block;
        text-align: center;
        padding: 5px 0 7px;
        margin: -2px 0 2px;
        font-size: 13px;
        font-weight: 700;
        line-height: 1;
        color: #d4af37;
        text-shadow: 0 0 10px rgba(212,175,55,.28);
      }
      .md blockquote { margin: 0.5em 0; padding-left: 0.85em; border-left: 3px solid rgba(124,92,255,.45); color: #c9d4f0; }
      .md a { color: #b8a8ff; }
      .md table { border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 12px; }
      .md th, .md td { border: 1px solid var(--border2); padding: 6px 8px; text-align: left; }
      .md th { background: rgba(0,0,0,.2); }
      .preWrap { position: relative; margin: 0.55em 0; }
      .preWrap pre {
        margin: 0;
        padding: 10px 10px 10px 10px;
        border-radius: 10px;
        border: 1px solid var(--border2);
        background: rgba(0,0,0,.35);
        overflow-x: auto;
        font-size: 12px;
        line-height: 1.35;
      }
      .codeCopy {
        position: absolute;
        top: 6px;
        right: 6px;
        z-index: 1;
        padding: 4px 8px;
        font-size: 11px;
        border-radius: 8px;
        border: 1px solid var(--border2);
        background: rgba(0,0,0,.45);
        color: var(--muted);
        cursor: pointer;
      }
      .codeCopy:hover { color: var(--text); border-color: rgba(255,255,255,.2); }
      .md-fallback { white-space: pre-wrap; }
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
      .composer { display:flex; gap: 8px; align-items: flex-end; }
      .composer textarea { flex: 1; }
      .composer .right { display:flex; gap: 8px; align-items:center; }
      .composer .right label { display:flex; align-items:center; gap: 6px; }
      .pane.chatCol .body { display:flex; flex-direction: column; min-height: 0; }
      .convCol .body.convBody { display: flex; flex-direction: column; min-height: 0; gap: 8px; padding-top: 6px; }
      .convList {
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        overflow-x: hidden;
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding-right: 4px;
      }
      .convItem {
        text-align: left;
        padding: 6px 8px;
        border-radius: 10px;
        border: 1px solid var(--border2);
        background: rgba(0,0,0,.14);
        color: var(--text);
        font-size: 11px;
        line-height: 1.3;
        cursor: pointer;
      }
      .convItem:hover { border-color: rgba(255,255,255,.18); }
      .convItem.active { border-color: rgba(124,92,255,.55); background: rgba(124,92,255,.14); }
      .convItem .t { font-weight: 600; color: #dbe6ff; }
      .convItem .d { color: var(--muted); font-size: 10px; margin-top: 2px; }
      .profileBackdrop {
        position: fixed;
        inset: 0;
        z-index: 50;
        background: rgba(0,0,0,.55);
        backdrop-filter: blur(4px);
        display: none;
        align-items: center;
        justify-content: center;
      }
      .profileBackdrop.open { display: flex; }
      .profileBox {
        width: min(360px, calc(100vw - 32px));
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(15,22,38,.96);
        padding: 16px;
        box-shadow: 0 12px 48px rgba(0,0,0,.45);
      }
      .profileBox h3 { margin: 0 0 8px; font-size: 14px; color: var(--text); font-weight: 650; }
      .profileBox label { display: block; margin-bottom: 4px; }
      @media (max-width: 900px) {
        .layout { flex-direction: column; }
        .convCol { flex: 0 0 auto; width: 100%; max-width: none; max-height: 200px; }
        .convCol.collapsed { max-height: 48px; }
        .readGrip { display: none; }
        .readingCol { width: 100% !important; max-width: none; flex: 1 1 auto; min-height: 220px; }
        .chatCol { flex: 1 1 auto; min-height: 240px; }
      }
    </style>
  </head>
  <body>
    <div class="topbar">
      <div class="brand">
        <div class="name" id="brandName">Dama</div>
        <div class="sub">
          <span>Index: <span id="idxStatus">unknown</span></span>
        </div>
      </div>
      <div class="rightmeta">
        <span id="profileGreet" class="small" style="color:#dbe6ff;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
        <button type="button" id="profileBtn" class="tightBtn">Profile</button>
        <button type="button" id="rebuild" class="warn tightBtn">Rebuild</button>
      </div>
    </div>

    <div class="layout" id="mainLayout">
      <div class="pane convCol" id="convCol">
        <div class="hdr">
          <span class="convTitle">Past chats</span>
          <button type="button" id="convToggleBtn" class="tightBtn" title="Collapse past chats">«</button>
        </div>
        <div class="body convBody">
          <div class="row" style="flex-shrink:0;align-items:center;">
            <button type="button" id="newConvBtn" class="tightBtn">New chat</button>
            <span class="muted small" id="convPanelHint" style="margin-left:6px;">Pick a thread</span>
          </div>
          <div id="convList" class="convList"></div>
        </div>
      </div>

      <div class="pane chatCol">
        <div class="hdr">
          <div class="hdrLabel">Chat</div>
          <div class="muted small" style="text-align:right;max-width:min(220px,42vw);line-height:1.35;" title="Enter to send · Shift+Enter for newline">Enter=send · Shift+Enter=newline</div>
        </div>
        <div class="body">
          <div class="chatArea">
            <div id="chatLog" class="chatLog"></div>
            <div class="composer">
              <textarea id="q" placeholder="Ask across AN1 and AN2…" rows="1"></textarea>
              <div class="right">
                <button type="button" id="stopAsk" class="tightBtn" style="display:none;">Stop</button>
                <button type="button" id="regenBtn" class="tightBtn" disabled
                  title="Re-sends your last question to the model without adding a new user message, so you can get an alternative answer in this thread.">↻</button>
                <button type="button" id="ask" class="primary tightBtn" onclick="window.__damaAsk&amp;&amp;window.__damaAsk();">Send</button>
              </div>
            </div>
            <div id="chatStatus" class="muted small"></div>
            <div id="quotes"></div>
          </div>
        </div>
      </div>

      <div class="readGrip" id="readGrip" role="separator" aria-orientation="vertical" title="Drag to resize Reference panel"></div>

      <div class="pane reading readingCol" id="readingCol">
        <div class="hdr">
          <div>Reference</div>
          <div class="muted" id="readingMeta"></div>
        </div>
        <div class="body">
          <div id="readingEmpty" class="muted">Click an (AN …) or (cAN …) citation in chat to open sutta and commentary here. When a sutta has link metadata, use Sutta / Links above.</div>
          <div id="readingBody" style="display:none;">
            <div id="refModeRow" class="row refModeRow" style="display:none;margin-bottom:10px;gap:6px;align-items:center;">
              <span class="muted small">Reference</span>
              <button type="button" id="refModeSutta" class="tightBtn primary">Sutta</button>
              <button type="button" id="refModeLinks" class="tightBtn">Links</button>
            </div>
            <div id="panelSutta">
              <h2>SUTTA</h2>
              <div id="suttaText" class="txt"></div>
              <div class="split"></div>
              <h2>TEACHER COMMENTARY</h2>
              <div id="commText" class="txt"></div>
            </div>
            <div id="panelLinks" style="display:none;">
              <h2>LINKS</h2>
              <div id="chainSummary" class="muted small" style="margin-bottom:8px;"></div>
              <div id="chainVisual" class="txt"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div id="profileBackdrop" class="profileBackdrop" role="dialog" aria-modal="true" aria-labelledby="profileTitle">
      <div class="profileBox">
        <h3 id="profileTitle">Profile</h3>
        <p class="muted small" style="margin:0 0 12px;">Local only — stored in this browser, not on the server.</p>
        <label class="muted small" for="profileNameInput">Display name</label>
        <input type="text" id="profileNameInput" maxlength="64" placeholder="Your name" autocomplete="nickname" />
        <div class="row" style="margin-top:14px; justify-content:flex-end; gap:8px;">
          <button type="button" id="profileSignOut" class="tightBtn">Sign out</button>
          <button type="button" id="profileSave" class="primary tightBtn">Save</button>
        </div>
      </div>
    </div>

    <script>
      const idxStatusEl = document.getElementById('idxStatus');
      const rebuildBtn = document.getElementById('rebuild');
      const askBtn = document.getElementById('ask');
      const qEl = document.getElementById('q');
      const chatStatusEl = document.getElementById('chatStatus');
      const quotesEl = document.getElementById('quotes');
      const chatLogEl = document.getElementById('chatLog');
      const stopAskBtn = document.getElementById('stopAsk');
      const regenBtn = document.getElementById('regenBtn');

      const readingEmptyEl = document.getElementById('readingEmpty');
      const readingBodyEl = document.getElementById('readingBody');
      const readingMetaEl = document.getElementById('readingMeta');
      const suttaTextEl = document.getElementById('suttaText');
      const commTextEl = document.getElementById('commText');
      const convListEl = document.getElementById('convList');
      const newConvBtn = document.getElementById('newConvBtn');
      const convPanelHint = document.getElementById('convPanelHint');
      const convCol = document.getElementById('convCol');
      const convToggleBtn = document.getElementById('convToggleBtn');
      const mainLayout = document.getElementById('mainLayout');
      const readGrip = document.getElementById('readGrip');
      const readingCol = document.getElementById('readingCol');
      const refModeRow = document.getElementById('refModeRow');
      const refModeSuttaBtn = document.getElementById('refModeSutta');
      const refModeLinksBtn = document.getElementById('refModeLinks');
      const panelSutta = document.getElementById('panelSutta');
      const panelLinks = document.getElementById('panelLinks');
      const chainSummaryEl = document.getElementById('chainSummary');
      const chainVisualEl = document.getElementById('chainVisual');
      const REF_MODE_KEY = 'dama_ref_panel_mode';
      const profileBackdrop = document.getElementById('profileBackdrop');
      const profileBtn = document.getElementById('profileBtn');
      const profileGreet = document.getElementById('profileGreet');
      const profileNameInput = document.getElementById('profileNameInput');
      const profileSave = document.getElementById('profileSave');
      const profileSignOut = document.getElementById('profileSignOut');
      const PROFILE_KEY = 'an1_profile_name';

      function syncUiHints() {
        if (qEl) qEl.placeholder = 'Ask across AN1 and AN2…';
        if (readingEmptyEl) {
          readingEmptyEl.textContent =
            'Click an (AN …) or (cAN …) citation in chat to open sutta and commentary here. When a sutta has link metadata, use Sutta / Links above.';
        }
        if (refModeRow) refModeRow.style.display = 'none';
        if (panelSutta && panelLinks && refModeSuttaBtn && refModeLinksBtn) {
          panelLinks.style.display = 'none';
          panelSutta.style.display = '';
          refModeSuttaBtn.classList.add('primary');
          refModeLinksBtn.classList.remove('primary');
        }
      }

      function getRefMode() {
        try {
          const m = (localStorage.getItem(REF_MODE_KEY) || 'sutta').toLowerCase();
          return m === 'links' ? 'links' : 'sutta';
        } catch (e) {
          return 'sutta';
        }
      }

      function setRefMode(m) {
        const v = m === 'links' ? 'links' : 'sutta';
        try {
          localStorage.setItem(REF_MODE_KEY, v);
        } catch (e) {}
        if (!refModeSuttaBtn || !refModeLinksBtn) return;
        refModeSuttaBtn.classList.toggle('primary', v === 'sutta');
        refModeLinksBtn.classList.toggle('primary', v === 'links');
        if (!refModeRow || refModeRow.style.display === 'none') return;
        if (panelSutta) panelSutta.style.display = v === 'sutta' ? '' : 'none';
        if (panelLinks) panelLinks.style.display = v === 'links' ? '' : 'none';
      }

      function renderChainVisual(chain) {
        if (!chainVisualEl || !chainSummaryEl) return;
        if (!chain || typeof chain !== 'object') {
          chainSummaryEl.textContent = '';
          chainVisualEl.textContent = '(no chain for this discourse)';
          return;
        }
        const cat = (chain.category || '').toString().trim();
        const items = Array.isArray(chain.items) ? chain.items : [];
        const n = chain.count != null ? Number(chain.count) : items.length;
        const ord = !!chain.is_ordered;
        chainSummaryEl.textContent =
          (cat ? 'Category: ' + cat + ' · ' : '') + (ord ? 'Ordered' : 'Unordered') + (n ? ' · ' + n + ' items' : '');
        if (!items.length) {
          chainVisualEl.textContent = '';
          return;
        }
        const sep = ord ? '→' : '↓';
        const parts = [];
        for (let i = 0; i < items.length; i++) {
          parts.push('<div class="chainNode">' + esc(String(items[i])) + '</div>');
          if (i < items.length - 1) {
            parts.push('<div class="chainFlowSep" aria-hidden="true">' + sep + '</div>');
          }
        }
        chainVisualEl.innerHTML = '<div class="chainFlow">' + parts.join('') + '</div>';
      }

      let _gongAudio = null;
      function playPing() {
        try {
          if (!_gongAudio) {
            _gongAudio = new Audio('/assets/gong.mp3');
            _gongAudio.preload = 'auto';
            _gongAudio.volume = 0.52;
          }
          _gongAudio.pause();
          _gongAudio.currentTime = 0;
          void _gongAudio.play().catch(() => {});
        } catch (e) {}
      }

      function esc(s) { return (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
      function escAttr(s) {
        return String(s ?? '')
          .replace(/&/g, '&amp;')
          .replace(/"/g, '&quot;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
      }
      function normalizeSuttaCiteRef(raw) {
        const t0 = String(raw || '').trim().replace(/\s+/g, ' ');
        let m = t0.match(/^cAN\s*(\d+(?:\.\d+)*)\s*$/i);
        if (m) return 'cAN ' + m[1];
        m = t0.match(/^AN\s*(\d+(?:\.\d+)*)\s*$/i);
        if (m) return 'AN ' + m[1];
        return t0;
      }
      function citeOpenButton(refInner, kind) {
        const r = normalizeSuttaCiteRef(String(refInner || '').trim());
        const safe = esc(r);
        const a = escAttr(r);
        const k = kind === 'commentary' ? 'commentary' : 'sutta';
        /* <a> is valid inside <p>; <button> is often stripped by DOMPurify inside paragraphs */
        return '<a href="#" class="cite cite-open" data-kind="' + k + '" data-ref="' + a + '">(' + safe + ')</a>';
      }
      function citeifyBareOutsideButtons(s) {
        const str = String(s || '');
        const low = str.toLowerCase();
        let i = 0;
        let out = '';
        while (i < str.length) {
          const iBtn = low.indexOf('<button', i);
          const iA = low.indexOf('<a ', i);
          let open = -1;
          let closeTag = '';
          if (iBtn >= 0 && (iA < 0 || iBtn <= iA)) {
            open = iBtn;
            closeTag = '</button>';
          } else if (iA >= 0) {
            open = iA;
            closeTag = '</a>';
          }
          if (open === -1) {
            out += _replaceBareCitations(str.slice(i));
            break;
          }
          const gt = str.indexOf('>', open);
          if (gt < 0) {
            out += _replaceBareCitations(str.slice(i));
            out += str.slice(open);
            break;
          }
          const openTag = str.slice(open, gt + 1);
          const isCite = openTag.indexOf('cite-open') >= 0;
          if (!isCite) {
            out += _replaceBareCitations(str.slice(i, open));
            if (open === iBtn) {
              out += str.slice(open, open + 7);
              i = open + 7;
            } else {
              out += str.slice(open, open + 3);
              i = open + 3;
            }
            continue;
          }
          out += _replaceBareCitations(str.slice(i, open));
          const close = low.indexOf(closeTag, gt);
          if (close === -1) {
            out += str.slice(open);
            break;
          }
          out += str.slice(open, close + closeTag.length);
          i = close + closeTag.length;
        }
        return out;
      }
      function _replaceBareCitations(t) {
        let x = String(t || '');
        /* cAN before AN; optional space so AN1.5.8 matches like AN 1.5.8 */
        x = x.replace(/\\b(cAN)\\s*(\\d+(?:\\.\\d+)*)\\b/gi, (_, __, nums) => citeOpenButton('cAN ' + nums, 'commentary'));
        x = x.replace(/\\b(AN)\\s*(\\d+(?:\\.\\d+)*)\\b/g, (_, __, nums) => citeOpenButton('AN ' + nums, 'sutta'));
        return x;
      }
      function formatAsstAnswer(raw) {
        let s = esc(raw ?? '');
        /* [(] [)] = literal parens */
        s = s.replace(/[(](cAN[^)]{0,160})[)]/gi, (_, g1) => citeOpenButton(g1, 'commentary'));
        s = s.replace(/[(](AN[^)]{0,160})[)]/g, (_, g1) => citeOpenButton(g1, 'sutta'));
        s = citeifyBareOutsideButtons(s);
        return s;
      }
      function citeifyOutsideFences(raw) {
        const parts = String(raw ?? '').split('```');
        const out = [];
        for (let i = 0; i < parts.length; i++) {
          let chunk = parts[i];
          if (i % 2 === 0) {
            chunk = chunk.replace(/[(](cAN[^)]{0,160})[)]/gi, (_, g1) => citeOpenButton(g1, 'commentary'));
            chunk = chunk.replace(/[(](AN[^)]{0,160})[)]/g, (_, g1) => citeOpenButton(g1, 'sutta'));
            chunk = citeifyBareOutsideButtons(chunk);
          }
          out.push(chunk);
        }
        return out.join('```');
      }
      function safeMdHtml(html) {
        if (typeof DOMPurify === 'undefined') return esc(String(html ?? ''));
        return DOMPurify.sanitize(String(html ?? ''), {
          ALLOWED_TAGS: ['p','br','ul','ol','li','strong','em','del','h1','h2','h3','h4','blockquote','a','pre','code','table','thead','tbody','tr','th','td','hr','small','span','button'],
          ALLOWED_ATTR: ['href','title','class','colspan','rowspan','rel','target','type','data-kind','data-ref'],
          ALLOW_DATA_ATTR: false,
        });
      }
      function renderMarkdown(raw) {
        const src = String(raw ?? '');
        if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
          return '<div class="md md-fallback">' + formatAsstAnswer(src) + '</div>';
        }
        const cited = citeifyOutsideFences(src);
        let html;
        try {
          marked.setOptions({ gfm: true, breaks: true, headerIds: false, mangle: false });
          html = marked.parse(cited);
        } catch (e) {
          html = '<p>' + esc(src) + '</p>';
        }
        return '<div class="md">' + safeMdHtml(html) + '</div>';
      }
      function attachCodeCopyButtons(rootEl) {
        rootEl.querySelectorAll('.md pre').forEach((pre) => {
          if (pre.closest('.preWrap')) return;
          const wrap = document.createElement('div');
          wrap.className = 'preWrap';
          const parent = pre.parentNode;
          parent.insertBefore(wrap, pre);
          wrap.insertBefore(pre, wrap.firstChild);
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'codeCopy';
          btn.textContent = 'Copy';
          btn.addEventListener('click', () => {
            const code = pre.querySelector('code');
            const t = (code ? code.innerText : pre.innerText) || '';
            if (navigator.clipboard && navigator.clipboard.writeText) {
              navigator.clipboard.writeText(t).catch(() => {});
            }
          });
          wrap.appendChild(btn);
        });
      }
      function setChatStatus(s) { chatStatusEl.textContent = s || ''; }
      function setIdxStatus(s) { idxStatusEl.textContent = s || ''; }

      let selectedId = null;
      let activeConversationId = null;
      let chat = [];
      let lastUserQuestion = '';
      let askAbort = null;
      let askBusy = false;

      function clearChatUi() {
        chat = [];
        lastUserQuestion = '';
        if (chatLogEl) chatLogEl.innerHTML = '';
        if (regenBtn) regenBtn.disabled = true;
      }

      function resizeComposer() {
        if (!qEl) return;
        qEl.style.height = 'auto';
        qEl.style.height = Math.min(qEl.scrollHeight, 180) + 'px';
      }

      function setComposerBusy(busy) {
        askBusy = busy;
        if (askBtn) askBtn.disabled = busy;
        if (stopAskBtn) stopAskBtn.style.display = busy ? 'inline-block' : 'none';
        if (regenBtn) regenBtn.disabled = !lastUserQuestion || busy;
        if (rebuildBtn) rebuildBtn.disabled = busy;
      }

      function removeLastUserMessage() {
        if (!chat.length) return;
        const last = chat[chat.length - 1];
        if (last.role !== 'user') return;
        chat.pop();
        const nodes = chatLogEl.querySelectorAll('.msg.user');
        const lastNode = nodes[nodes.length - 1];
        if (lastNode) lastNode.remove();
      }

      function appendMsg(role, text, meta) {
        if (!chatLogEl) return;
        const id = 'm_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
        chat.push({ id, role, text, meta: meta || null, rating: 0 });
        const wrap = document.createElement('div');
        wrap.className = 'msg ' + (role === 'user' ? 'user' : 'asst');
        const b = document.createElement('div');
        b.className = 'bubble' + ((meta && meta.error) ? ' bubbleErr' : '');
        if (role === 'user') {
          b.textContent = text || '';
        } else {
          b.innerHTML = renderMarkdown(text || '');
          attachCodeCopyButtons(b);
        }
        wrap.appendChild(b);
        if (role !== 'user') {
          const footer = document.createElement('div');
          footer.className = 'msgFooter';
          const lat = (meta && meta.latency_ms != null) ? Math.round(Number(meta.latency_ms)) : null;
          if (lat != null && !Number.isNaN(lat)) {
            const latEl = document.createElement('span');
            latEl.className = 'muted small';
            latEl.textContent = lat + ' ms';
            latEl.style.marginRight = '6px';
            footer.appendChild(latEl);
          }
          const copyWhole = document.createElement('button');
          copyWhole.type = 'button';
          copyWhole.className = 'thumb';
          copyWhole.textContent = 'Copy';
          copyWhole.title = 'Copy full answer';
          copyWhole.addEventListener('click', () => {
            const t = text || '';
            if (navigator.clipboard && navigator.clipboard.writeText) {
              navigator.clipboard.writeText(t).catch(() => {});
            }
          });
          footer.appendChild(copyWhole);

          if (meta && meta.error && meta.retryQuestion) {
            const retry = document.createElement('button');
            retry.type = 'button';
            retry.className = 'thumb';
            retry.textContent = 'Retry';
            retry.addEventListener('click', () => {
              wrap.remove();
              chat = chat.filter((m) => m.id !== id);
              void askWithQuestion(meta.retryQuestion, { fromInput: false });
            });
            footer.appendChild(retry);
          }

          const spacer = document.createElement('div');
          spacer.style.flex = '1';
          footer.appendChild(spacer);

          if (!(meta && meta.error)) {
            const up = document.createElement('button');
            up.type = 'button';
            up.className = 'thumb';
            up.textContent = '👍';
            const down = document.createElement('button');
            down.type = 'button';
            down.className = 'thumb';
            down.textContent = '👎';

            function setRating(r) {
              chat = chat.map((m) => (m.id === id ? { ...m, rating: r } : m));
              up.classList.toggle('active', r === 1);
              down.classList.toggle('active', r === -1);
              fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  message_id: id,
                  rating: r,
                  question: (meta && meta.question) || '',
                  answer: text || '',
                  sources: (meta && meta.sources) || [],
                }),
              }).catch(() => {});
            }
            up.addEventListener('click', () => setRating(1));
            down.addEventListener('click', () => setRating(-1));
            footer.appendChild(up);
            footer.appendChild(down);
          }

          b.appendChild(footer);
        }
        chatLogEl.appendChild(wrap);
        chatLogEl.scrollTop = chatLogEl.scrollHeight;
      }

      function hydrateChatFromHistory(items) {
        chatLogEl.innerHTML = '';
        chat = [];
        lastUserQuestion = '';
        const ordered = (items || []).slice();
        for (let i = 0; i < ordered.length; i++) {
          const row = ordered[i];
          const q = (row.q != null) ? String(row.q) : '';
          const a = (row.a != null) ? String(row.a) : '';
          if (!q.trim() && !a.trim()) continue;
          if (q.trim()) appendMsg('user', q, null);
          appendMsg('asst', a.trim() ? a : '(no answer)', {
            question: q,
            sources: [],
            latency_ms: row.latency_ms != null ? Math.round(Number(row.latency_ms)) : null,
          });
        }
        if (ordered.length) {
          const last = ordered[ordered.length - 1];
          lastUserQuestion = (last.q != null) ? String(last.q).trim() : '';
        }
        regenBtn.disabled = !lastUserQuestion || askBusy;
        chatLogEl.scrollTop = chatLogEl.scrollHeight;
      }

      async function fetchJson(url, opts) {
        const o = Object.assign({ cache: 'no-store' }, opts || {});
        const res = await fetch(url, o);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const d = data.detail;
          const msg = typeof d === 'string' ? d : JSON.stringify(d || data);
          throw new Error(msg || res.statusText);
        }
        return data;
      }

      async function boot() {
        try {
          await refreshIndexStatus();
          await refreshConversationList();
        } catch (e) {
          setChatStatus('Startup: ' + (e && e.message ? e.message : e));
        }
      }

      async function refreshIndexStatus() {
        try {
          const data = await fetchJson('/api/index_status');
          const mode = (data.mode || '').toString();
          const brandEl = document.getElementById('brandName');
          if (brandEl) {
            brandEl.textContent = mode === 'vertex' ? 'Dama — Vertex' : 'Dama — local';
          }
          const n1 = Number(data.count) || 0;
          const n2 = Number(data.an2_count) || 0;
          const line =
            mode === 'vertex'
              ? (data.exists ? 'AN1 Vertex · ' + n1 + ' chunks' : 'AN1 Vertex missing') +
                (n2 ? ' · AN2 local ' + n2 : '')
              : 'AN1: ' + n1 + ' · AN2: ' + n2 + (mode ? ' · ' + mode : '');
          setIdxStatus(line);
          const tip = [
            data.build,
            data.mode,
            data.embedding_model,
            data.manifest_gcs,
            data.manifest_path,
            data.manifest_shard_count != null ? 'shards: ' + data.manifest_shard_count : '',
            data.bundle_gcs,
            data.bundle_path,
            data.error,
            data.an1_path,
            data.an2_path,
            data.persist_dir,
            data.persist_an2_dir,
            data.vertex_note,
          ]
            .filter(Boolean)
            .join('\\n');
          idxStatusEl.title = tip || '';
        } catch {
          setIdxStatus('unknown');
          idxStatusEl.title = '';
        }
      }

      async function refreshConversationList() {
        try {
          const data = await fetchJson('/api/conversations');
          const rows = data.conversations || [];
          convListEl.innerHTML = '';
          if (!rows.length) {
            convListEl.innerHTML = '<div class="muted small">No saved chats yet.</div>';
            return;
          }
          for (const row of rows) {
            const id = (row.id || '').toString();
            const title = esc((row.title || 'Chat').toString());
            const ts = row.updated_ts != null ? Number(row.updated_ts) : 0;
            const tstr = ts ? new Date(ts).toLocaleString() : '';
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'convItem' + (activeConversationId && id.toLowerCase() === activeConversationId.toLowerCase() ? ' active' : '');
            btn.innerHTML = '<div class="t">' + title + '</div><div class="d">' + esc(tstr) + '</div>';
            btn.addEventListener('click', () => void selectConversation(id));
            convListEl.appendChild(btn);
          }
        } catch {
          convListEl.innerHTML = '<div class="muted small">Could not load chat list.</div>';
        }
      }

      async function loadConversationMessages(conversationId, opts) {
        opts = opts || {};
        const hydrate = !!opts.hydrateChat;
        try {
          const bust = '_=' + Date.now();
          const data = await fetchJson('/api/conversations/' + encodeURIComponent(conversationId) + '/messages?' + bust);
          const items = data.items || [];
          if (hydrate) hydrateChatFromHistory(items);
        } catch {
          if (hydrate) hydrateChatFromHistory([]);
        }
      }

      async function selectConversation(conversationId) {
        activeConversationId = conversationId;
        convPanelHint.textContent = 'Active: ' + (conversationId || '').slice(0, 8) + '…';
        await loadConversationMessages(conversationId, { hydrateChat: true });
        await refreshConversationList();
      }

      function startNewConversation() {
        clearChatUi();
        activeConversationId = null;
        convPanelHint.textContent = 'New thread — saves after a reply';
        void refreshConversationList();
      }

      function applyReadingFromDetail(data, suttaidForMeta) {
        readingEmptyEl.style.display = 'none';
        readingBodyEl.style.display = 'block';
        const sid = (suttaidForMeta || (data.suttaid || '')).toString().trim();
        const cid = (data.commentary_id || '').toString().trim();
        selectedId = sid;
        readingMetaEl.textContent = cid ? ('suttaid: ' + sid + ' · commentary_id: ' + cid) : ('suttaid: ' + sid);
        suttaTextEl.textContent = data.sutta || '';
        const commRaw = (data.commentry != null && String(data.commentry).trim()) ? data.commentry : '(no commentary)';
        commTextEl.textContent = commRaw;
        const ch = data.chain != null ? data.chain : null;
        const hasChain = ch && typeof ch === 'object' && (Array.isArray(ch.items) ? ch.items.length > 0 : Object.keys(ch).length > 0);
        if (refModeRow) {
          if (hasChain) {
            refModeRow.style.display = 'flex';
            renderChainVisual(ch);
            setRefMode(getRefMode());
          } else {
            refModeRow.style.display = 'none';
            renderChainVisual(null);
          }
        }
      }

      async function openReadingFromCitation(kind, ref) {
        const r = (ref || '').trim();
        if (!r) return;
        suttaTextEl.textContent = '';
        commTextEl.textContent = '';
        try {
          let data;
          if (kind === 'commentary') {
            data = await fetchJson('/api/item_by_commentary_id/' + encodeURIComponent(r));
          } else {
            data = await fetchJson('/api/item/' + encodeURIComponent(r));
          }
          const sidMeta = (data.suttaid || '').toString().trim() || normalizeSuttaCiteRef(r);
          applyReadingFromDetail(data, sidMeta);
        } catch (e) {
          readingEmptyEl.style.display = 'none';
          readingBodyEl.style.display = 'block';
          suttaTextEl.textContent = 'Could not load: ' + (e?.message || e);
          commTextEl.textContent = '';
          renderChainVisual(null);
        }
      }

      async function rebuild() {
        rebuildBtn.disabled = true;
        setChatStatus('Rebuilding index…');
        try {
          const data = await fetchJson('/api/build?book=all', { method: 'POST' });
          setChatStatus('Index rebuilt · collection_count=' + data.collection_count);
        } catch (e) {
          setChatStatus('Rebuild error: ' + (e?.message || e));
        }
        await refreshIndexStatus();
        rebuildBtn.disabled = false;
      }

      async function ask() {
        if (!qEl) return;
        const question = (qEl.value || '').trim();
        if (!question) return;
        await askWithQuestion(question, { fromInput: true });
      }

      async function askWithQuestion(question, opts) {
        opts = opts || {};
        const fromInput = !!opts.fromInput;
        if (!question || askBusy) return;
        if (fromInput && !qEl) return;
        askAbort = new AbortController();
        setComposerBusy(true);
        quotesEl.innerHTML = '';
        setChatStatus('Searching…');
        if (fromInput) {
          appendMsg('user', question);
          qEl.value = '';
          resizeComposer();
        }
        lastUserQuestion = question;
        const t0 = performance.now();
        try {
          const data = await fetchJson('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, book: 'all' }),
            signal: askAbort.signal,
          });
          const latencyMs = Math.round(performance.now() - t0);
          setChatStatus('Done · ' + latencyMs + ' ms');
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
          const rawAnswer = data.answer != null ? String(data.answer) : '';
          const answerTrim = rawAnswer.trim();
          const answerText = answerTrim || '(no answer)';
          appendMsg('asst', answerText, { question, sources: sidKinds, latency_ms: latencyMs });
          playPing();
          const worthSaving = answerTrim.length > 0 && answerTrim !== '(no answer)';
          if (worthSaving) {
            try {
              let cid = activeConversationId;
              if (!cid) {
                const cdata = await fetchJson('/api/conversations', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ title: '' }),
                });
                cid = cdata.id;
                activeConversationId = cid;
                convPanelHint.textContent = 'New thread';
              }
              await fetchJson('/api/conversations/' + encodeURIComponent(cid) + '/messages', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, answer: answerTrim, latency_ms: latencyMs }),
              });
              await refreshConversationList();
            } catch (e) {
              /* persistence optional */
            }
          }
          quotesEl.innerHTML = '';
        } catch (e) {
          const latencyMs = Math.round(performance.now() - t0);
          const aborted = e && (e.name === 'AbortError' || String(e.message || '').toLowerCase().includes('abort'));
          if (aborted) {
            setChatStatus('Cancelled');
            if (fromInput) {
              removeLastUserMessage();
              qEl.value = question;
              resizeComposer();
            }
          } else {
            setChatStatus('Error after ' + latencyMs + ' ms: ' + (e && e.message ? e.message : e));
            appendMsg('asst', '**Request failed**\\n\\n' + (e && e.message ? e.message : String(e)), {
              error: true,
              question,
              retryQuestion: question,
              latency_ms: latencyMs,
            });
          }
        } finally {
          askAbort = null;
          setComposerBusy(false);
        }
      }

      async function regenerateLast() {
        if (!lastUserQuestion || askBusy) return;
        await askWithQuestion(lastUserQuestion, { fromInput: false });
      }

      if (rebuildBtn) rebuildBtn.addEventListener('click', rebuild);
      if (askBtn) askBtn.addEventListener('click', ask);
      window.__damaAsk = ask;
      if (stopAskBtn) {
        stopAskBtn.addEventListener('click', () => {
          if (askAbort) askAbort.abort();
        });
      }
      if (regenBtn) regenBtn.addEventListener('click', () => void regenerateLast());
      if (qEl) {
        qEl.addEventListener('input', () => resizeComposer());
        qEl.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            ask();
          }
        });
      }

      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
          void refreshConversationList();
        }
      });

      newConvBtn.addEventListener('click', () => startNewConversation());

      chatLogEl.addEventListener('click', (ev) => {
        const btn = ev.target.closest('.cite-open');
        if (!btn || !chatLogEl.contains(btn)) return;
        const kind = (btn.getAttribute('data-kind') || 'sutta').toString();
        const ref = (btn.getAttribute('data-ref') || '').toString();
        if (!ref) return;
        ev.preventDefault();
        void openReadingFromCitation(kind, ref);
      });

      function getProfileName() {
        try {
          return (localStorage.getItem(PROFILE_KEY) || '').trim();
        } catch (e) {
          return '';
        }
      }

      function syncProfileGreet() {
        const n = getProfileName();
        profileGreet.textContent = n ? ('Hello, ' + n) : '';
      }

      function openProfileModal() {
        profileNameInput.value = getProfileName();
        profileBackdrop.classList.add('open');
        profileNameInput.focus();
      }

      function closeProfileModal() {
        profileBackdrop.classList.remove('open');
      }

      function initProfileChrome() {
        syncProfileGreet();
        profileBtn.addEventListener('click', () => openProfileModal());
        profileBackdrop.addEventListener('click', (ev) => {
          if (ev.target === profileBackdrop) closeProfileModal();
        });
        profileSave.addEventListener('click', () => {
          const v = (profileNameInput.value || '').trim().slice(0, 64);
          try {
            if (v) localStorage.setItem(PROFILE_KEY, v);
            else localStorage.removeItem(PROFILE_KEY);
          } catch (e) {}
          syncProfileGreet();
          closeProfileModal();
        });
        profileSignOut.addEventListener('click', () => {
          try {
            localStorage.removeItem(PROFILE_KEY);
          } catch (e) {}
          profileNameInput.value = '';
          syncProfileGreet();
          closeProfileModal();
        });
        document.addEventListener('keydown', (ev) => {
          if (ev.key === 'Escape' && profileBackdrop.classList.contains('open')) {
            ev.preventDefault();
            closeProfileModal();
          }
        });
      }

      function readPanelClamp(w) {
        const max = Math.min(Math.floor(window.innerWidth * 0.72), 980);
        const min = 240;
        let n = Math.round(Number(w) || 420);
        if (!Number.isFinite(n)) n = 420;
        return Math.min(Math.max(n, min), max);
      }

      function applyReadPanelWidth(px) {
        const w = readPanelClamp(px);
        readingCol.style.width = w + 'px';
        readingCol.style.flex = '0 0 ' + w + 'px';
        try {
          localStorage.setItem('an1_read_px', String(w));
        } catch (e) {}
      }

      function clearReadPanelCustomWidth() {
        readingCol.style.width = '';
        readingCol.style.flex = '';
        try {
          localStorage.removeItem('an1_read_px');
        } catch (e) {}
      }

      function applyConvCollapsed(collapsed) {
        convCol.classList.toggle('collapsed', !!collapsed);
        convToggleBtn.textContent = collapsed ? '»' : '«';
        convToggleBtn.title = collapsed ? 'Expand past chats' : 'Collapse past chats';
        try {
          localStorage.setItem('an1_conv_collapsed', collapsed ? '1' : '0');
        } catch (e) {}
      }

      function initLayoutChrome() {
        let rw = NaN;
        try {
          const s = localStorage.getItem('an1_read_px');
          if (s) rw = parseInt(s, 10);
        } catch (e) {}
        if (Number.isFinite(rw) && rw > 0) {
          applyReadPanelWidth(rw);
        } else {
          clearReadPanelCustomWidth();
        }
        try {
          applyConvCollapsed(localStorage.getItem('an1_conv_collapsed') === '1');
        } catch (e) {}

        convToggleBtn.addEventListener('click', () => {
          applyConvCollapsed(!convCol.classList.contains('collapsed'));
        });

        readGrip.addEventListener('mousedown', (e) => {
          if (window.innerWidth <= 900) return;
          e.preventDefault();
          const startX = e.clientX;
          const startW = readingCol.getBoundingClientRect().width;
          mainLayout.classList.add('resizing');
          function onMove(ev) {
            const dx = ev.clientX - startX;
            applyReadPanelWidth(startW - dx);
          }
          function onUp() {
            mainLayout.classList.remove('resizing');
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
          }
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });

        window.addEventListener('resize', () => {
          if (window.innerWidth <= 900) return;
          try {
            if (!localStorage.getItem('an1_read_px')) return;
          } catch (e) {
            return;
          }
          applyReadPanelWidth(readingCol.getBoundingClientRect().width);
        });
      }

      if (refModeSuttaBtn) refModeSuttaBtn.addEventListener('click', () => setRefMode('sutta'));
      if (refModeLinksBtn) refModeLinksBtn.addEventListener('click', () => setRefMode('links'));
      syncUiHints();
      setRefMode(getRefMode());

      initProfileChrome();
      initLayoutChrome();
      void boot();
    </script>
  </body>
</html>
""",
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
    )


@app.get("/api/items")
def api_items(q: str = Query(default=""), book: str = Query("all")) -> JSONResponse:
    bq = (book or "all").strip().lower()
    bk = "all" if bq not in ("an1", "an2") else bq
    items = _merged_item_details() if bk == "all" else _load_items(_norm_book(bk))
    qn = (q or "").strip().lower()
    filtered: List[ItemDetail] = []
    if qn:
        for i in items:
            chain_blob = ""
            if i.chain:
                try:
                    chain_blob = json.dumps(i.chain, ensure_ascii=False)
                except Exception:
                    chain_blob = str(i.chain)
            blob = f"{i.suttaid}\n{i.sutta}\n{i.commentry}\n{i.commentary_id}\n{chain_blob}".lower()
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
    _dbg("H4", "an1_app.py:api_items", "Serve items", {"book": bk, "count": len(out), "q": (q or "")[:80]})
    return JSONResponse({"items": out})


@app.get("/api/item/{suttaid}", response_model=ItemDetail)
def api_item(suttaid: str, book: str = Query("all")) -> ItemDetail:
    bq = (book or "all").strip().lower()
    if bq in ("an1", "an2"):
        for it in _load_items(bq):
            if it.suttaid == suttaid:
                return it
    else:
        hit = _find_item_by_suttaid(suttaid)
        if hit:
            return hit
    raise HTTPException(status_code=404, detail=f"Unknown suttaid: {suttaid}")


@app.get("/api/item_by_commentary_id/{commentary_id}", response_model=ItemDetail)
def api_item_by_commentary_id(commentary_id: str, book: str = Query("all")) -> ItemDetail:
    want = (commentary_id or "").strip()
    if not want:
        raise HTTPException(status_code=400, detail="empty commentary_id")
    bq = (book or "all").strip().lower()
    if bq in ("an1", "an2"):
        for it in _load_items(bq):
            if (it.commentary_id or "").strip() == want:
                return it
    else:
        hit = _find_item_by_commentary_id(want)
        if hit:
            return hit
    raise HTTPException(status_code=404, detail=f"Unknown commentary_id: {want}")


def _read_jsonl_messages(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.is_file():
        return items
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            continue
    return items


@app.get("/api/conversations")
def api_conversations_list() -> JSONResponse:
    with _lock:
        _prune_empty_conversations()
        rows = _list_conversations_merged()
    return JSONResponse({"conversations": rows})


@app.post("/api/conversations")
def api_conversations_create(body: ConversationCreate) -> JSONResponse:
    cid = str(uuid.uuid4())
    title = (body.title or "").strip() or "New chat"
    now = int(time.time() * 1000)
    with _lock:
        idx = [r for r in _read_conversation_index() if isinstance(r, dict) and str(r.get("id", "")).lower() != cid]
        idx.append({"id": cid, "title": title, "updated_ts": now})
        _write_conversation_index(idx)
    return JSONResponse({"id": cid, "title": title, "updated_ts": now})


@app.get("/api/conversations/{conversation_id}/messages")
def api_conversation_messages_get(conversation_id: str) -> JSONResponse:
    path = _conversation_jsonl_path(conversation_id)
    cid_norm = str(uuid.UUID(conversation_id))
    items = _read_jsonl_messages(path)
    return JSONResponse(
        {"conversation_id": cid_norm, "items": items},
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
    )


@app.post("/api/conversations/{conversation_id}/messages")
def api_conversation_messages_append(conversation_id: str, body: ChatHistoryAppend) -> JSONResponse:
    path = _conversation_jsonl_path(conversation_id)
    cid_norm = str(uuid.UUID(conversation_id))
    qstrip = (body.question or "").strip()
    astrip = (body.answer or "").strip()
    if not qstrip:
        raise HTTPException(status_code=400, detail="empty question")
    if not astrip or astrip == "(no answer)":
        raise HTTPException(status_code=400, detail="refusing to store without a model answer")
    rec = {
        "ts": int(time.time() * 1000),
        "q": qstrip,
        "a": astrip,
        "latency_ms": int(body.latency_ms),
    }
    qstrip = qstrip.replace("\n", " ")
    qstrip = re.sub(r"\s+", " ", qstrip)
    snippet = (qstrip[:56] + "…") if len(qstrip) > 56 else qstrip
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        rows = [dict(r) for r in _read_conversation_index() if isinstance(r, dict)]
        found = False
        for r in rows:
            if str(r.get("id", "")).lower() == cid_norm.lower():
                r["updated_ts"] = rec["ts"]
                t = str(r.get("title") or "").strip()
                if (not t or t == "New chat") and snippet:
                    r["title"] = snippet
                found = True
                break
        if not found:
            rows.append({"id": cid_norm, "title": snippet or "Chat", "updated_ts": rec["ts"]})
        _write_conversation_index(rows)
    return JSONResponse({"ok": True, "conversation_id": cid_norm})


def _collection_count_safe(book: str) -> int:
    try:
        return int(_get_collection(_norm_book(book)).count())
    except Exception:
        return 0


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
                    "an2_path": str(AN2_PATH.resolve()),
                    "an2_count": _collection_count_safe("an2"),
                    "runtime": os.environ.get("DAMA_RUNTIME", "").strip() or None,
                    "build": AN1_APP_BUILD,
                    "bundle_gcs": os.environ.get("AN1_VERTEX_BUNDLE_GCS_URI", "").strip(),
                    "bundle_path": os.environ.get("AN1_VERTEX_BUNDLE_PATH", "").strip(),
                    "manifest_gcs": os.environ.get("AN1_VERTEX_MANIFEST_GCS_URI", "").strip(),
                    "manifest_path": os.environ.get("AN1_VERTEX_MANIFEST_PATH", "").strip(),
                    "embedding_model": meta.get("embedding_model") or "",
                    "bundle_format": meta.get("format") or "",
                    "manifest_shard_count": meta.get("manifest_shard_count"),
                    "manifest_version": meta.get("manifest_version"),
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
                    "an2_path": str(AN2_PATH.resolve()),
                    "an2_count": _collection_count_safe("an2"),
                    "build": AN1_APP_BUILD,
                    "error": str(e),
                }
            )

    c1 = _collection_count_safe("an1")
    c2 = _collection_count_safe("an2")
    return JSONResponse(
        {
            "exists": bool(c1 > 0 or c2 > 0),
            "count": c1,
            "an2_count": c2,
            "mode": "local_chroma",
            "persist_dir": str(PERSIST_DIR.resolve()),
            "persist_an2_dir": str(PERSIST_AN2_DIR.resolve()),
            "an1_path": str(AN1_PATH.resolve()),
            "an2_path": str(AN2_PATH.resolve()),
            "build": AN1_APP_BUILD,
        }
    )


@app.post("/api/build", response_model=BuildResponse)
def api_build(
    book: str = Query(
        "all",
        description="Build: all (AN1+AN2), an1 only, or an2 only (local Chroma). Vertex path rebuilds local Chroma + vertex_corpus shards (manifest); optional AN1_VERTEX_CORPUS_UPLOAD_BASE_GCS.",
    ),
) -> BuildResponse:
    raw = (book or "all").strip().lower()
    b = raw if raw in ("an1", "an2", "all") else "all"
    with _lock:
        _dbg("H2", "an1_app.py:api_build", "Build requested", {"book": b})
        try:
            if vx.an1_vertex_enabled():
                from an1_build_vertex_bundle import write_vertex_corpus

                an1_build.build_an1_index()
                an1_build.build_an2_index()
                corpus_dir = PERSIST_DIR / "vertex_corpus"
                upload_base = os.environ.get("AN1_VERTEX_CORPUS_UPLOAD_BASE_GCS", "").strip()
                write_vertex_corpus(corpus_dir, upload_base_gcs=upload_base)
                vx.invalidate_bundle_cache()
                bundle = vx.ensure_bundle_loaded(PERSIST_DIR)
                _invalidate_items_cache()
                return BuildResponse(ok=True, collection_count=len(vx.bundle_chunk_rows(bundle)))

            if b == "an2":
                an1_build.build_an2_index()
                _invalidate_items_cache("an2")
                return BuildResponse(ok=True, collection_count=_collection_count_safe("an2"))

            if b == "an1":
                an1_build.build_an1_index()
                _invalidate_items_cache("an1")
                return BuildResponse(ok=True, collection_count=_collection_count_safe("an1"))

            an1_build.build_an1_index()
            if AN2_PATH.exists():
                an1_build.build_an2_index()
            _invalidate_items_cache()
            c1 = _collection_count_safe("an1")
            c2 = _collection_count_safe("an2")
            return BuildResponse(ok=True, collection_count=c1 + c2)
        except Exception as e:
            _dbg("H2", "an1_app.py:api_build", "Build failed", {"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
def api_query(req: QueryRequest) -> QueryResponse:
    with _lock:
        req.use_llm = True
        req.k = 8
        book = (req.book or "all").strip().lower()
        if book not in ("an1", "an2", "all"):
            book = "all"
        _dbg(
            "H4",
            "an1_app.py:api_query",
            "Query",
            {"book": book, "k": req.k, "use_llm": req.use_llm, "question_len": len(req.question)},
        )

        if _is_chat_only_message(req.question):
            _dbg(
                "H4",
                "an1_app.py:api_query",
                "Chat-only message; skip retrieval/LLM",
                {"question_len": len(req.question)},
            )
            return QueryResponse(chunks=[], answer=_CHAT_ONLY_REPLY, used_llm=False)

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
                    answer = _call_llm(req.question, llm_chunks, book=book)
                except Exception as e:
                    _dbg("H3", "an1_app.py:api_query", "Vertex LLM failed", {"error": str(e)})
                    raise HTTPException(status_code=502, detail=f"Vertex LLM call failed: {e}") from e

            return QueryResponse(chunks=chunks, answer=answer, used_llm=used_llm)

        if not PERSIST_DIR.exists() and not PERSIST_AN2_DIR.exists():
            _dbg(
                "H2",
                "an1_app.py:api_query",
                "Index missing",
                {"persist_dir": str(PERSIST_DIR), "persist_an2": str(PERSIST_AN2_DIR)},
            )
            raise HTTPException(
                status_code=400,
                detail="No local index found. Run Rebuild (POST /api/build) first.",
            )

        embed_model = _get_embed_model()
        chunks = _retrieve_merged_local(embed_model, req.question, req.k)

        used_llm = False
        answer = ""
        if req.use_llm:
            used_llm = True
            llm_chunks = chunks[:10]
            try:
                _dbg(
                    "H3",
                    "an1_app.py:api_query",
                    "Calling Ollama (merged AN1+AN2)",
                    {"chunks": len(llm_chunks), "model": OLLAMA_MODEL},
                )
                answer = _call_llm(req.question, llm_chunks, book="all")
            except Exception as e:
                _dbg("H3", "an1_app.py:api_query", "Ollama failed", {"error": str(e)})
                raise HTTPException(status_code=502, detail=f"Ollama LLM call failed: {e}") from e

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

