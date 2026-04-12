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
    conversation_id: str = ""
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


def _chunk_from_doc(
    doc: Any, meta: Any, dist: Optional[float] = None
) -> Chunk:
    src = ""
    cid = ""
    if isinstance(meta, dict):
        src = str(meta.get("source") or "")
        cid = str(meta.get("conversation_id") or "")
    return Chunk(
        source=src,
        conversation_id=cid,
        distance=float(dist) if dist is not None else None,
        text=str(doc),
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
        out.append(_chunk_from_doc(doc, meta, float(dist) if dist is not None else None))

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
        seen_keys.add((c.source, c.conversation_id, c.text[:200]))  # best-effort dedupe

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
            ch = _chunk_from_doc(doc, meta, None)
            key = (ch.source, ch.conversation_id, ch.text[:200])
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
                f"[Excerpt {index} | source: {chunk.source} | conv: {chunk.conversation_id or '-'}]\n{chunk.text}\n\n"
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
            [
                f"[Excerpt {i+1} | source: {c.source} | conv: {c.conversation_id or '-'}]\n{c.text}"
                for i, c in enumerate(chunks)
            ]
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
    <div class="muted">Ask questions against your index (repo-root *.txt transcripts plus *.chat.jsonl under training-data/, cursor-export/, grok-export/). Use “Rebuild Index” after adding files.</div>

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
          html += '<div class="src"><b>source</b>: ' + esc(c.source) + (c.conversation_id ? (' &nbsp; <b>conv</b>: ' + esc(c.conversation_id)) : '') + (c.distance != null ? (' &nbsp; <b>distance</b>: ' + esc(c.distance)) : '') + '</div>';
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


@app.get("/v2", response_class=HTMLResponse)
def home_v2() -> str:
    # A nicer UI that still stays single-file and dependency-free.
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Dama RAG (local) — v2</title>
    <style>
      :root {
        --bg: #0b1020;
        --panel: #111a33;
        --panel2: #0f1730;
        --text: #e9eefc;
        --muted: #a7b3d6;
        --border: rgba(255,255,255,.10);
        --accent: #7c5cff;
        --good: #41d17a;
        --bad: #ff5c7a;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
        color: var(--text);
        background: radial-gradient(1200px 600px at 20% 0%, #18244a 0%, var(--bg) 60%) fixed;
      }
      .wrap { max-width: 1100px; margin: 0 auto; padding: 18px; }
      header {
        display: flex; gap: 14px; align-items: center; justify-content: space-between;
        padding: 14px 16px; border: 1px solid var(--border); border-radius: 14px;
        background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
        backdrop-filter: blur(6px);
      }
      .title { display: flex; flex-direction: column; gap: 2px; }
      .title h1 { font-size: 16px; margin: 0; letter-spacing: .2px; }
      .title .sub { font-size: 12px; color: var(--muted); }
      .pill { display:inline-flex; align-items:center; gap:8px; padding:8px 10px; border-radius: 999px;
              border: 1px solid var(--border); background: rgba(255,255,255,.04); color: var(--muted); font-size: 12px; }
      .grid { display: grid; grid-template-columns: 1fr; gap: 14px; margin-top: 14px; }
      @media (min-width: 980px) { .grid { grid-template-columns: 1.1fr .9fr; } }
      .card {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(17,26,51,.75);
        backdrop-filter: blur(6px);
        overflow: hidden;
      }
      .card h2 { font-size: 13px; margin: 0; padding: 12px 14px; border-bottom: 1px solid var(--border); color: var(--muted); }
      .card .body { padding: 12px 14px; }
      .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
      input[type="text"]{
        flex: 1; min-width: 260px; padding: 12px 12px; border-radius: 12px;
        border: 1px solid var(--border); background: rgba(0,0,0,.20); color: var(--text);
        outline: none;
      }
      input[type="number"]{
        width: 86px; padding: 12px 10px; border-radius: 12px;
        border: 1px solid var(--border); background: rgba(0,0,0,.20); color: var(--text);
        outline: none;
      }
      button{
        padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border);
        background: rgba(255,255,255,.06); color: var(--text); cursor: pointer;
      }
      button.primary { background: linear-gradient(180deg, rgba(124,92,255,.9), rgba(124,92,255,.6)); border-color: rgba(124,92,255,.7); }
      button.ghost { background: transparent; }
      button:disabled { opacity: .55; cursor: not-allowed; }
      label { color: var(--muted); font-size: 12px; }
      .status { margin-top: 10px; font-size: 12px; color: var(--muted); }
      .answer {
        margin-top: 12px; padding: 12px; border-radius: 12px;
        border: 1px solid var(--border); background: rgba(0,0,0,.18);
        white-space: pre-wrap; word-break: break-word; font-size: 14px;
      }
      .chunks { display: flex; flex-direction: column; gap: 10px; }
      .chunk {
        padding: 12px; border-radius: 12px; border: 1px solid var(--border);
        background: rgba(15,23,48,.55);
      }
      .meta { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom: 8px; color: var(--muted); font-size: 12px; }
      .meta b { color: var(--text); font-weight: 600; }
      .chunk pre { margin: 0; white-space: pre-wrap; word-break: break-word; color: #dbe6ff; font-size: 13px; line-height: 1.35; }
      .copy {
        margin-left: auto; display:inline-flex; align-items:center; gap:6px;
        padding: 6px 10px; border-radius: 10px; border: 1px solid var(--border);
        background: rgba(255,255,255,.05); color: var(--muted); cursor: pointer;
      }
      .hint { color: var(--muted); font-size: 12px; margin-top: 8px; }
      .kpi { display:flex; gap:10px; flex-wrap:wrap; }
      .kpi .pill strong { color: var(--text); font-weight: 700; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <div class="title">
          <h1>Dama RAG (local) — v2</h1>
          <div class="sub">RAG over transcripts + your exported chats. Quotes are shown so you can verify “it came from my logs”.</div>
        </div>
        <div class="kpi">
          <div class="pill">UI: <strong>v2</strong> <span style="opacity:.7">/v2</span></div>
          <div class="pill">API: <strong>/api/query</strong></div>
        </div>
      </header>

      <div class="grid">
        <div class="card">
          <h2>Ask</h2>
          <div class="body">
            <div class="row">
              <input id="q" type="text" placeholder="Ask about your chats… (e.g. “what did we decide about ports?”)" />
              <label>k</label>
              <input id="k" type="number" min="1" max="20" value="6" />
              <label><input id="use_llm" type="checkbox" checked /> use Ollama LLM</label>
              <button id="ask" class="primary">Ask</button>
              <button id="rebuild" class="ghost">Rebuild index</button>
            </div>
            <div id="status" class="status"></div>
            <div id="answer" class="answer" style="display:none"></div>
            <div class="hint">Tip: turn off “use Ollama LLM” to see pure retrieval without any summarization.</div>
          </div>
        </div>

        <div class="card">
          <h2>Top matching quotes</h2>
          <div class="body">
            <div id="chunks" class="chunks"></div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const statusEl = document.getElementById('status');
      const answerEl = document.getElementById('answer');
      const chunksEl = document.getElementById('chunks');
      const askBtn = document.getElementById('ask');
      const rebuildBtn = document.getElementById('rebuild');

      function setStatus(s) { statusEl.textContent = s || ''; }
      function esc(s) { return (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

      async function ask() {
        const question = document.getElementById('q').value.trim();
        const k = parseInt(document.getElementById('k').value || '6', 10);
        const use_llm = document.getElementById('use_llm').checked;
        if (!question) return;

        askBtn.disabled = true;
        rebuildBtn.disabled = true;
        answerEl.style.display = 'none';
        chunksEl.innerHTML = '';
        setStatus('Searching…');

        try {
          const res = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, k, use_llm })
          });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(data.detail || JSON.stringify(data) || res.statusText);

          setStatus(data.used_llm ? 'Done (LLM answer generated).' : 'Done (retrieval only).');
          answerEl.textContent = (data.answer || '(no answer)');
          answerEl.style.display = 'block';

          const chunks = (data.chunks || []);
          if (!chunks.length) {
            chunksEl.innerHTML = '<div class="hint">No matches. Try fewer words or increase k.</div>';
          } else {
            for (const c of chunks) {
              const div = document.createElement('div');
              div.className = 'chunk';
              const conv = c.conversation_id ? esc(c.conversation_id) : '-';
              const dist = (c.distance != null) ? String(c.distance) : '-';
              const src = esc(c.source || '');
              const text = (c.text || '').toString();

              div.innerHTML =
                '<div class=\"meta\">' +
                  '<span><b>source</b>: ' + src + '</span>' +
                  '<span><b>conv</b>: ' + conv + '</span>' +
                  '<span><b>distance</b>: ' + esc(dist) + '</span>' +
                  '<span class=\"copy\" title=\"Copy quote\">Copy</span>' +
                '</div>' +
                '<pre>' + esc(text) + '</pre>';

              div.querySelector('.copy').onclick = async () => {
                try { await navigator.clipboard.writeText(text); setStatus('Copied quote to clipboard.'); }
                catch { setStatus('Copy failed (clipboard permissions).'); }
              };
              chunksEl.appendChild(div);
            }
          }
        } catch (e) {
          setStatus('Error: ' + (e?.message || e));
        }

        askBtn.disabled = false;
        rebuildBtn.disabled = false;
      }

      async function rebuild() {
        askBtn.disabled = true;
        rebuildBtn.disabled = true;
        setStatus('Rebuilding index… (this can take a bit)');
        answerEl.style.display = 'none';
        chunksEl.innerHTML = '';
        try {
          const res = await fetch('/api/build', { method: 'POST' });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(data.detail || JSON.stringify(data) || res.statusText);
          setStatus('Index rebuilt. collection_count=' + data.collection_count);
        } catch (e) {
          setStatus('Rebuild error: ' + (e?.message || e));
        }
        askBtn.disabled = false;
        rebuildBtn.disabled = false;
      }

      askBtn.addEventListener('click', ask);
      rebuildBtn.addEventListener('click', rebuild);
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
            # Expand context: prefer all chunks from the same chat conversation_id
            # (merged JSONL shares one filename across many chats).
            llm_chunks = list(chunks)
            seen = set((c.source, c.conversation_id, c.text[:200]) for c in llm_chunks)
            primary = chunks[0] if chunks else None
            where_filter: Optional[Dict[str, Any]] = None
            if primary and primary.conversation_id.strip():
                where_filter = {"conversation_id": primary.conversation_id}
            elif primary and primary.source:
                where_filter = {"source": primary.source}
            if where_filter:
                try:
                    got = col.get(
                        where=where_filter,
                        include=["documents", "metadatas"],
                        limit=50,
                    )
                    for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
                        ch = _chunk_from_doc(doc, meta, None)
                        key = (ch.source, ch.conversation_id, ch.text[:200])
                        if key not in seen:
                            seen.add(key)
                            llm_chunks.append(ch)
                except Exception:
                    pass
            # Cap LLM context to avoid overwhelming the model.
            llm_chunks = llm_chunks[:8]
            try:
                answer = _call_llm(req.question, llm_chunks)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Ollama LLM call failed: {e}")

        return QueryResponse(chunks=chunks, answer=answer, used_llm=used_llm)

