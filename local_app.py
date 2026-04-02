import threading
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
COLLECTION_NAME = "dama_transcripts"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:instruct"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = Field(default=True)


class Chunk(BaseModel):
    source: str = ""
    distance: Optional[float] = None
    text: str


class QueryResponse(BaseModel):
    chunks: List[Chunk]
    answer: str = ""
    used_llm: bool = False


class BuildResponse(BaseModel):
    ok: bool
    collection_count: int


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


def _retrieve(embed_model: SentenceTransformer, collection: Any, query: str, k: int) -> List[Chunk]:
    q_emb = embed_model.encode([query])[0].tolist()
    # Pull extra candidates, then rerank. Embedding-only search often misses
    # exact keyword/phrase matches for short queries.
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
        src = ""
        if isinstance(meta, dict):
            src = str(meta.get("source") or "")
        out.append(Chunk(source=src, distance=float(dist) if dist is not None else None, text=str(doc)))

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
            # Chroma supports document substring filtering with where_document.
            # We use `get()` so we don't need another embedding query.
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
            src = ""
            if isinstance(meta, dict):
                src = str(meta.get("source") or "")
            text = str(doc)
            key = (src, text[:200])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(Chunk(source=src, distance=None, text=text))

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


def _map_extract(query: str, chunk: Chunk, index: int) -> str:
    """Map step: extract relevant facts and quotes from a single chunk."""
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
                f"Question: {query}\n\n"
                f"[Excerpt {index} | source: {chunk.source}]\n{chunk.text}\n\n"
                f"Extract relevant facts as bullet points with quotes:"
            ),
        },
    ]
    return _ollama_chat(messages, num_ctx=4096, timeout=300)


def _reduce_synthesize(query: str, extracted_notes: str) -> str:
    """Reduce step: synthesize a complete answer from extracted notes."""
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
                f"Question: {query}\n\n"
                f"EXTRACTED NOTES FROM TRANSCRIPTS:\n{extracted_notes}\n\n"
                f"Combine ALL the notes above into a complete, well-organized answer. "
                f"Include quotes to support each point."
            ),
        },
    ]
    return _ollama_chat(messages, num_ctx=8192, timeout=300)


def _call_llm(query: str, chunks: List[Chunk]) -> str:
    if len(chunks) <= 10:
        numbered = "\n\n".join(
            [f"[Excerpt {i+1} | source: {c.source}]\n{c.text}" for i, c in enumerate(chunks)]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions ONLY from the provided transcript excerpts.\n\n"
                    "STRICT RULES:\n"
                    "1. You have NO prior knowledge. The excerpts below are your ONLY source of truth.\n"
                    "2. SKIP any excerpt that is not directly relevant to the question.\n"
                    "3. For every claim, quote the supporting words from the excerpt.\n"
                    "4. Be concise. Do not pad the answer with tangential information.\n"
                    "5. If no excerpt answers the question, say: "
                    "\"The provided excerpts do not contain information to answer this question.\"\n"
                    "6. Do NOT guess, infer, or fill gaps with outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"TRANSCRIPT EXCERPTS:\n{numbered}\n\n"
                    f"Answer the question using ONLY relevant excerpts. "
                    f"Ignore excerpts that do not address the question. "
                    f"Quote key passages and combine into a focused answer."
                ),
            },
        ]
        return _ollama_chat(messages)

    # Map-Reduce for 4+ chunks: extract per-chunk, then synthesize.
    notes_parts = []
    for i, chunk in enumerate(chunks, 1):
        extracted = _map_extract(query, chunk, i)
        if "NOT_RELEVANT" not in extracted.upper():
            notes_parts.append(f"[From excerpt {i} | source: {chunk.source}]\n{extracted}")

    if not notes_parts:
        return "The provided excerpts do not contain information to answer this question."

    all_notes = "\n\n".join(notes_parts)
    return _reduce_synthesize(query, all_notes)


app = FastAPI(title="Dama RAG (local)")

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
def home() -> str:
    # Minimal single-file UI so you can just run and use it.
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Dama RAG (local)</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; max-width: 980px; }
      h1 { margin: 0 0 12px; }
      .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin: 12px 0; }
      input[type="text"] { flex: 1; min-width: 280px; padding: 10px 12px; font-size: 14px; }
      input[type="number"] { width: 90px; padding: 10px 8px; }
      button { padding: 10px 14px; font-size: 14px; cursor: pointer; }
      .muted { color: #666; font-size: 13px; }
      .box { border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin-top: 12px; }
      pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
      .chunk { margin-top: 10px; padding-top: 10px; border-top: 1px dashed #ddd; }
      .src { font-size: 12px; color: #444; margin-bottom: 6px; }
    </style>
  </head>
  <body>
    <h1>Dama RAG (local)</h1>
    <div class="muted">Ask questions against your transcript index. Use “Rebuild Index” if you add/change transcript files.</div>

    <div class="row">
      <input id="q" type="text" placeholder="Ask a question…" />
      <label class="muted">k</label>
      <input id="k" type="number" min="1" max="20" value="5" />
      <label class="muted"><input id="use_llm" type="checkbox" checked /> use Ollama LLM</label>
      <button id="ask">Ask</button>
      <button id="rebuild">Rebuild Index</button>
    </div>

    <div id="status" class="muted"></div>
    <div id="answer" class="box" style="display:none"></div>
    <div id="chunks" class="box" style="display:none"></div>

    <script>
      const statusEl = document.getElementById('status');
      const answerEl = document.getElementById('answer');
      const chunksEl = document.getElementById('chunks');

      function setStatus(s) { statusEl.textContent = s; }
      function esc(s) { return (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

      async function ask() {
        const question = document.getElementById('q').value.trim();
        const k = parseInt(document.getElementById('k').value || '5', 10);
        const use_llm = document.getElementById('use_llm').checked;
        if (!question) return;

        answerEl.style.display = 'none';
        chunksEl.style.display = 'none';
        setStatus('Asking…');

        const res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, k, use_llm })
        });

        if (!res.ok) {
          const t = await res.text();
          setStatus('Error: ' + t);
          return;
        }

        const data = await res.json();
        setStatus(data.used_llm ? 'Done (LLM answer generated).' : 'Done (retrieval only).');

        answerEl.innerHTML = '<h3>Answer</h3><pre>' + esc(data.answer || '(no answer)') + '</pre>';
        answerEl.style.display = 'block';

        let html = '<h3>Top matching chunks</h3>';
        for (const c of (data.chunks || [])) {
          html += '<div class="chunk">';
          html += '<div class="src"><b>source</b>: ' + esc(c.source) + (c.distance != null ? (' &nbsp; <b>distance</b>: ' + esc(c.distance)) : '') + '</div>';
          html += '<pre>' + esc(c.text) + '</pre>';
          html += '</div>';
        }
        chunksEl.innerHTML = html;
        chunksEl.style.display = 'block';
      }

      async function rebuild() {
        answerEl.style.display = 'none';
        chunksEl.style.display = 'none';
        setStatus('Rebuilding index… (this can take a bit)');
        const res = await fetch('/api/build', { method: 'POST' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          setStatus('Rebuild error: ' + (data.detail || JSON.stringify(data)));
          return;
        }
        setStatus('Index rebuilt. collection_count=' + data.collection_count);
      }

      document.getElementById('ask').addEventListener('click', ask);
      document.getElementById('rebuild').addEventListener('click', rebuild);
      document.getElementById('q').addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });
    </script>
  </body>
</html>
"""


@app.post("/api/build", response_model=BuildResponse)
def build_index() -> BuildResponse:
    with _lock:
        try:
            # Rebuild the persistent collection on disk.
            build_index_module.main()
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


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    with _lock:
        if not PERSIST_DIR.exists():
            raise HTTPException(status_code=400, detail="Index not found. Click Rebuild Index first.")

        try:
            col = _get_collection()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to open collection: {e}")

        embed_model = _get_embed_model()
        chunks = _retrieve(embed_model, col, req.question, req.k)

        used_llm = False
        answer = ""
        if req.use_llm:
            used_llm = True
            # Expand context: pull all chunks from the primary source file
            # so the LLM sees the full lecture argument, not just isolated hits.
            llm_chunks = list(chunks)
            seen = set((c.source, c.text[:200]) for c in llm_chunks)
            primary_src = chunks[0].source if chunks else ""
            if primary_src:
                try:
                    got = col.get(
                        where={"source": primary_src},
                        include=["documents", "metadatas"],
                        limit=50,
                    )
                    for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
                        src = str(meta.get("source", "")) if isinstance(meta, dict) else ""
                        text = str(doc)
                        key = (src, text[:200])
                        if key not in seen:
                            seen.add(key)
                            llm_chunks.append(Chunk(source=src, text=text))
                except Exception:
                    pass
            # Cap LLM context to avoid overwhelming the model.
            llm_chunks = llm_chunks[:8]
            try:
                answer = _call_llm(req.question, llm_chunks)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Ollama LLM call failed: {e}")

        return QueryResponse(chunks=chunks, answer=answer, used_llm=used_llm)

