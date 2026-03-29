import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
import requests
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder

import build_index as build_index_module


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "rag_index"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "conversations.db"
COLLECTION_NAME = "dama_transcripts"
OLLAMA_BASE_URL = "http://localhost:11434"
# Align default with ft/build_ollama_modelfile.py and DAMA_HF_MODEL (Qwen2.5 0.5B).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b-instruct")

_STATIC_INDEX = BASE_DIR / "static" / "index.html"
_index_html_cache: Optional[str] = None


def _get_index_html() -> str:
    global _index_html_cache
    if _index_html_cache is None:
        if not _STATIC_INDEX.is_file():
            return (
                "<!doctype html><html><body><h1>Missing UI file</h1>"
                "<p>Expected <code>static/index.html</code> beside the app. Restore it from the repo.</p></body></html>"
            )
        _index_html_cache = _STATIC_INDEX.read_text(encoding="utf-8")
    return _index_html_cache


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = Field(default=True)
    session_id: Optional[str] = Field(default=None)
    an_book: Optional[int] = Field(
        default=None,
        ge=1,
        description="If set, restrict retrieval to chunks whose filename metadata matches this AN book number.",
    )


class SessionCreateResponse(BaseModel):
    session_id: str


class ChatMessageItem(BaseModel):
    role: str
    content: str


class SessionMessagesResponse(BaseModel):
    messages: List[ChatMessageItem]


class Chunk(BaseModel):
    source: str = ""
    distance: Optional[float] = None
    text: str
    an_book: int = 0
    lecture_ord: int = 0
    an_sutta_ref: str = ""
    sutta_chunk_part: int = 0


class QueryResponse(BaseModel):
    chunks: List[Chunk]
    answer: str = ""
    used_llm: bool = False
    elapsed_ms: float = Field(description="Total server-side time for this request in milliseconds")
    rewrite_ms: float = Field(
        default=0,
        description="Ollama query-rewrite time for chat follow-ups (ms); 0 if unused",
    )
    retrieval_ms: float = Field(
        default=0,
        description="Embedding search + rerank in _retrieve (ms)",
    )
    context_ms: float = Field(
        default=0,
        description="Extra Chroma fetch of chunks from the primary source file before LLM (ms); 0 if LLM off",
    )
    llm_ms: float = Field(
        default=0,
        description="Ollama answer generation / map-reduce (ms); 0 if LLM off",
    )


class BuildResponse(BaseModel):
    ok: bool
    collection_count: int


class OllamaBenchResponse(BaseModel):
    ok: bool
    ollama_ms: float = Field(description="Wall time for one minimal Ollama chat completion (ms)")
    model: str = ""


def _get_collection() -> Any:
    client = chromadb.PersistentClient(
        path=str(PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)

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


def _chunk_from_doc_meta(doc: Any, meta: Any, dist: Optional[float]) -> Chunk:
    src = ""
    ab = 0
    lo = 0
    ref = ""
    scp = 0
    if isinstance(meta, dict):
        src = str(meta.get("source") or "")
        try:
            ab = int(meta.get("an_book") or 0)
        except (TypeError, ValueError):
            ab = 0
        try:
            lo = int(meta.get("lecture_ord") or 0)
        except (TypeError, ValueError):
            lo = 0
        ref = str(meta.get("an_sutta_ref") or "")
        try:
            scp = int(meta.get("sutta_chunk_part") or 0)
        except (TypeError, ValueError):
            scp = 0
    return Chunk(
        source=src,
        distance=float(dist) if dist is not None else None,
        text=str(doc),
        an_book=ab,
        lecture_ord=lo,
        an_sutta_ref=ref,
        sutta_chunk_part=scp,
    )


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
        # Only boost when majority of meaningful terms match.
        if ratio >= 0.6:
            score += int(ratio * 20)
    return score


def _retrieve(
    embed_model: SentenceTransformer,
    collection: Any,
    query: str,
    k: int,
    an_book: Optional[int] = None,
) -> List[Chunk]:
    q_emb = embed_model.encode([query])[0].tolist()
    # Pull extra candidates, then rerank. Embedding-only search often misses
    # exact keyword/phrase matches for short queries.
    n_candidates = min(max(k * 10, 50), 200)
    q_kw: Dict[str, Any] = {
        "query_embeddings": [q_emb],
        "n_results": n_candidates,
        "include": ["documents", "metadatas", "distances"],
    }
    if an_book is not None:
        q_kw["where"] = {"an_book": an_book}
    results: Dict[str, Any] = collection.query(**q_kw)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out: List[Chunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(_chunk_from_doc_meta(doc, meta, float(dist) if dist is not None else None))

    # Keyword fallback: also pull chunks that literally contain query terms.
    # This is critical for short keyword searches like "thai forest".
    phrase = (query or "").strip().lower()
    terms = []
    if phrase and len(phrase) <= 60 and len(phrase.split()) >= 2:
        terms.append(phrase)
    terms.extend(_tokenize_query(query))
    terms = list(dict.fromkeys([t for t in terms if t]))  # stable unique

    seen_keys = set()
    for c in out:
        seen_keys.add((c.source, c.text[:200]))  # best-effort dedupe

    for t in terms[:6]:
        try:
            g_kw: Dict[str, Any] = {
                "where_document": {"$contains": t},
                "include": ["documents", "metadatas"],
                "limit": min(200, max(50, k * 20)),
            }
            if an_book is not None:
                g_kw["where"] = {"an_book": an_book}
            got = collection.get(**g_kw)
        except Exception:
            continue

        k_docs = (got or {}).get("documents", []) or []
        k_metas = (got or {}).get("metadatas", []) or []
        for doc, meta in zip(k_docs, k_metas):
            text = str(doc)
            ch = _chunk_from_doc_meta(doc, meta, None)
            key = (ch.source, text[:200])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(ch)

    # Stage 1: lexical + embedding score to get top candidates
    out.sort(
        key=lambda c: (
            -_lexical_score(query, c.text),
            c.distance if c.distance is not None else 1e9,
        )
    )
    candidates = out[:max(k * 4, 20)]

    # Stage 2: cross-encoder reranking for precision
    if len(candidates) > k:
        try:
            reranker = _get_reranker()
            pairs = [[query, c.text] for c in candidates]
            scores = reranker.predict(pairs)
            scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
            candidates = [c for c, _ in scored]
        except Exception:
            pass

    return candidates[:k]


def _sanitize_history_for_llm(
    history: List[Dict[str, str]], max_messages: int = 12, max_chars: int = 3200
) -> List[Dict[str, str]]:
    """Trim prior turns for the LLM so follow-ups stay in context without blowing num_ctx."""
    out: List[Dict[str, str]] = []
    for m in history[-max_messages:]:
        r = str(m.get("role", "")).strip()
        c = (m.get("content") or "").strip()
        if r not in ("user", "assistant") or not c:
            continue
        if len(c) > max_chars:
            c = c[: max_chars - 3] + "..."
        out.append({"role": r, "content": c})
    return out


def _history_conversation_block(history: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for m in _sanitize_history_for_llm(history):
        label = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{label}: {m['content']}")
    return "\n\n".join(lines)


def _ollama_chat(messages: list, temperature: float = 0, num_ctx: int = 4096, timeout: int = 600) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
              "options": {"temperature": temperature, "num_ctx": num_ctx}},
        timeout=timeout,
    )
    if resp.status_code != 200:
        error_detail = resp.text[:500]
        raise RuntimeError(f"Ollama returned {resp.status_code}: {error_detail}")
    return resp.json().get("message", {}).get("content", "")


def _map_extract(query: str, chunk: Chunk, index: int, convo_block: str = "") -> str:
    """Map step: extract relevant facts and quotes from a single chunk."""
    prefix = ""
    if convo_block.strip():
        prefix = (
            "Recent conversation (use only to interpret what the question refers to):\n"
            f"{convo_block}\n\n"
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise fact extractor. Given a question and a transcript excerpt, "
                "extract ONLY the facts relevant to the question. For each fact, quote the exact "
                "words from the excerpt. If the excerpt contains nothing relevant, respond with "
                "exactly: NOT_RELEVANT"
            ),
        },
        {
            "role": "user",
            "content": (
                f"{prefix}"
                f"Question: {query}\n\n"
                f"[Excerpt {index} | source: {chunk.source}]\n{chunk.text}\n\n"
                f"Extract relevant facts as bullet points with quotes:"
            ),
        },
    ]
    return _ollama_chat(messages, num_ctx=4096, timeout=300)


def _reduce_synthesize(query: str, extracted_notes: str, convo_block: str = "") -> str:
    """Reduce step: synthesize a complete answer from extracted notes."""
    prefix = ""
    if convo_block.strip():
        prefix = (
            "Recent conversation (use only to interpret the question; facts must come from the notes):\n"
            f"{convo_block}\n\n"
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions ONLY from the provided extracted notes.\n\n"
                "STRICT RULES — violating any of these is a failure:\n"
                "- You have NO prior knowledge. The notes below are the ONLY facts that exist.\n"
                "- NEVER state anything not in the notes — not even if you \"know\" it is true.\n"
                "- For every claim, include the supporting quote from the notes.\n"
                "- If the notes do not contain the answer, say: "
                "\"The provided excerpts do not contain information to answer this question.\"\n"
                "- Do NOT guess, infer, or fill gaps with outside knowledge."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{prefix}"
                f"Question: {query}\n\n"
                f"EXTRACTED NOTES FROM TRANSCRIPTS:\n{extracted_notes}\n\n"
                f"Combine ALL the notes above into a complete, well-organized answer. "
                f"Include quotes to support each point."
            ),
        },
    ]
    return _ollama_chat(messages, num_ctx=8192, timeout=300)


def _call_llm(
    query: str, chunks: List[Chunk], chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    hist = _sanitize_history_for_llm(chat_history) if chat_history else []
    convo_block = _history_conversation_block(chat_history) if chat_history else ""
    num_ctx = 8192 if hist else 4096

    sys_rules = (
        "You answer questions ONLY from the provided transcript excerpts.\n\n"
        "STRICT RULES:\n"
        "1. You have NO prior knowledge. The excerpts below are your ONLY source of truth.\n"
        "2. SKIP any excerpt that is not directly relevant to the question.\n"
        "3. For every claim, quote the supporting words from the excerpt.\n"
        "4. Be concise. Do not pad the answer with tangential information.\n"
        "5. If no excerpt answers the question, say: "
        "\"The provided excerpts do not contain information to answer this question.\"\n"
        "6. Do NOT guess, infer, or fill gaps with outside knowledge."
    )
    if hist:
        sys_rules += (
            "\n\n7. Prior user/assistant turns help interpret follow-ups (pronouns, \"that\", etc.); "
            "still ground every factual claim in the transcript excerpts in the final user message."
        )

    if len(chunks) <= 10:
        numbered = "\n\n".join(
            [f"[Excerpt {i+1} | source: {c.source}]\n{c.text}" for i, c in enumerate(chunks)]
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": sys_rules}]
        for m in hist:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"TRANSCRIPT EXCERPTS:\n{numbered}\n\n"
                    f"Answer the question using ONLY relevant excerpts. "
                    f"Ignore excerpts that do not address the question. "
                    f"Quote key passages and combine into a focused answer."
                ),
            }
        )
        return _ollama_chat(messages, num_ctx=num_ctx, timeout=300)

    # Map-Reduce for many chunks: extract per-chunk, then synthesize.
    notes_parts = []
    for i, chunk in enumerate(chunks, 1):
        extracted = _map_extract(query, chunk, i, convo_block)
        if "NOT_RELEVANT" not in extracted.upper():
            notes_parts.append(f"[From excerpt {i} | source: {chunk.source}]\n{extracted}")

    if not notes_parts:
        return "The provided excerpts do not contain information to answer this question."

    all_notes = "\n\n".join(notes_parts)
    return _reduce_synthesize(query, all_notes, convo_block)


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH), timeout=60.0) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                title TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            """
        )
        conn.commit()


@contextmanager
def _db_conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=60.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return row is not None


def _load_recent_messages(conn: sqlite3.Connection, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    rows = list(reversed(rows))
    return [{"role": str(r["role"]), "content": str(r["content"])} for r in rows]


def _append_exchange(conn: sqlite3.Connection, session_id: str, user_q: str, assistant_a: str) -> None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
        (session_id, user_q, now),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
        (session_id, assistant_a, now),
    )


def _rewrite_query_for_search(history: List[Dict[str, str]], new_question: str) -> str:
    lines: List[str] = []
    for m in history[-10:]:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    transcript = "\n".join(lines)
    if len(transcript) > 1500:
        transcript = transcript[-1500:]
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite follow-up messages into one standalone search query for a Buddhist "
                "transcript library. Resolve pronouns and implicit references using the conversation. "
                "Output ONLY the search query text — no quotes, labels, or explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{transcript}\n\nNew user message: {new_question}\n\n"
                f"Standalone search query:"
            ),
        },
    ]
    try:
        raw = _ollama_chat(messages, temperature=0, num_ctx=2048, timeout=120)
    except Exception:
        return new_question
    q = (raw or "").strip().strip('"').strip("'").split("\n")[0].strip()
    return q if len(q) >= 3 else new_question


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _init_db()
    yield


app = FastAPI(title="Dama RAG (local)", lifespan=lifespan)

_lock = threading.RLock()
_embed_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(
        content=_get_index_html(),
        media_type="text/html; charset=utf-8",
    )


@app.post("/api/build", response_model=BuildResponse)
def build_index() -> BuildResponse:
    with _lock:
        try:
            # Rebuild the persistent collection on disk (full corpus; use CLI for --only-prefix).
            build_index_module.run_build()
            col = _get_collection()
            return BuildResponse(ok=True, collection_count=int(col.count()))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def _decompose_query(question: str) -> List[str]:
    """Split a complex question into 2-3 simpler sub-queries for better retrieval."""
    word_count = len(question.split())
    if word_count <= 8:
        return [question]

    try:
        result = _ollama_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You decompose complex questions into simpler sub-queries for a search engine. "
                        "Output 2-3 short search queries, one per line. No numbering, no bullets, "
                        "just the queries. If the question is already simple, output it unchanged."
                    ),
                },
                {"role": "user", "content": question},
            ],
            num_ctx=2048, timeout=120,
        )
        sub_queries = [q.strip().strip("-•*0123456789.") for q in result.strip().splitlines() if q.strip()]
        sub_queries = [q for q in sub_queries if len(q) >= 5]
        if not sub_queries:
            return [question]
        return sub_queries[:3]
    except Exception:
        return [question]


@app.post("/api/sessions", response_model=SessionCreateResponse)
def create_session_api() -> SessionCreateResponse:
    with _lock:
        sid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with _db_conn() as conn:
            conn.execute("INSERT INTO sessions (id, created_at, title) VALUES (?, ?, NULL)", (sid, now))
        return SessionCreateResponse(session_id=sid)


@app.get("/api/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
def get_session_messages(session_id: str) -> SessionMessagesResponse:
    """Return saved chat turns so the web UI can restore the thread after refresh."""
    ui_message_limit = 500
    with _lock:
        with _db_conn() as conn:
            if not _session_exists(conn, session_id):
                raise HTTPException(status_code=404, detail="Unknown session_id")
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, ui_message_limit),
            ).fetchall()
    return SessionMessagesResponse(
        messages=[ChatMessageItem(role=str(r["role"]), content=str(r["content"])) for r in rows]
    )


@app.get("/api/ollama-bench", response_model=OllamaBenchResponse)
def ollama_bench() -> OllamaBenchResponse:
    """One minimal chat completion to measure local Ollama round-trip latency."""
    _t0 = time.perf_counter()
    try:
        _ollama_chat(
            [{"role": "user", "content": "Reply with exactly the single word: OK"}],
            temperature=0,
            num_ctx=256,
            timeout=120,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama failed: {e}")
    ms = (time.perf_counter() - _t0) * 1000
    return OllamaBenchResponse(ok=True, ollama_ms=round(ms, 2), model=OLLAMA_MODEL)


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    with _lock:
        _q_wall0 = time.perf_counter()
        if not PERSIST_DIR.exists():
            raise HTTPException(status_code=400, detail="Index not found. Click Rebuild Index first.")

        try:
            col = _get_collection()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to open collection: {e}")

        history: List[Dict[str, str]] = []
        retrieval_q = req.question
        _rewrite_ms = 0.0
        if req.session_id:
            with _db_conn() as conn:
                if not _session_exists(conn, req.session_id):
                    raise HTTPException(status_code=400, detail="Unknown session_id. Create a session first.")
                history = _load_recent_messages(conn, req.session_id, limit=10)
            if history:
                _trw = time.perf_counter()
                retrieval_q = _rewrite_query_for_search(history, req.question)
                _rewrite_ms = (time.perf_counter() - _trw) * 1000

        embed_model = _get_embed_model()
        _t_r0 = time.perf_counter()
        chunks = _retrieve(embed_model, col, retrieval_q, req.k, an_book=req.an_book)
        _retrieve_total_ms = (time.perf_counter() - _t_r0) * 1000

        used_llm = False
        answer = ""
        _expand_ms = 0.0
        if req.use_llm:
            used_llm = True
            # Expand context: pull all chunks from the primary source file
            # so the LLM sees the full lecture argument, not just isolated hits.
            llm_chunks = list(chunks)
            seen = set((c.source, c.text[:200]) for c in llm_chunks)
            primary_src = chunks[0].source if chunks else ""
            if primary_src:
                try:
                    _t_ex = time.perf_counter()
                    if req.an_book is not None:
                        exp_where: Dict[str, Any] = {
                            "$and": [{"source": primary_src}, {"an_book": req.an_book}]
                        }
                    else:
                        exp_where = {"source": primary_src}
                    got = col.get(
                        where=exp_where,
                        include=["documents", "metadatas"],
                        limit=50,
                    )
                    for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
                        ch = _chunk_from_doc_meta(doc, meta, None)
                        key = (ch.source, ch.text[:200])
                        if key not in seen:
                            seen.add(key)
                            llm_chunks.append(ch)
                    _expand_ms = (time.perf_counter() - _t_ex) * 1000
                except Exception:
                    pass
            # Cap LLM context to avoid overwhelming the model.
            llm_chunks = llm_chunks[:8]
            try:
                _t_llm = time.perf_counter()
                answer = _call_llm(req.question, llm_chunks, chat_history=history or None)
                _llm_ms = (time.perf_counter() - _t_llm) * 1000
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Ollama LLM call failed: {e}")
        else:
            _llm_ms = 0.0

        if req.session_id:
            with _db_conn() as conn:
                _append_exchange(conn, req.session_id, req.question, answer)

        _elapsed_ms = round((time.perf_counter() - _q_wall0) * 1000, 2)
        return QueryResponse(
            chunks=chunks,
            answer=answer,
            used_llm=used_llm,
            elapsed_ms=_elapsed_ms,
            rewrite_ms=round(_rewrite_ms, 2),
            retrieval_ms=round(_retrieve_total_ms, 2),
            context_ms=round(_expand_ms, 2),
            llm_ms=round(_llm_ms, 2),
        )

