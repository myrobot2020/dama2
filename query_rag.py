import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import re
import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "rag_index"
COLLECTION_NAME = "dama_transcripts"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b-instruct")


def get_collection():
    client = chromadb.PersistentClient(
        path=str(PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


def retrieve(context_embed_model: SentenceTransformer, collection, query: str, k: int = 5) -> List[str]:
    q_emb = context_embed_model.encode([query])[0].tolist()
    n_candidates = min(max(k * 10, 50), 200)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    _stopwords = frozenset(
        "a an and are as at be but by do for from get got has have he her him his how "
        "if in into is it its just me my no nor not now of on or our out own she so "
        "some than that the their them then there these they this those through to too "
        "very was we were what when where which who will with would you your".split()
    )

    def _tokenize_query(q: str) -> List[str]:
        tokens = [
            t for t in re.findall(r"[a-zA-Z0-9']+", (q or "").lower())
            if len(t) >= 3 and t not in _stopwords
        ]
        return tokens[:12]

    def _lexical_score(q: str, doc_text: str) -> int:
        q = (q or "").strip().lower()
        if not q or not doc_text:
            return 0
        text = str(doc_text).lower()
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

    # Keyword fallback via `where_document $contains`
    phrase = (query or "").strip().lower()
    terms: List[str] = []
    if phrase and len(phrase) <= 60 and len(phrase.split()) >= 2:
        terms.append(phrase)
    terms.extend(_tokenize_query(query))
    terms = list(dict.fromkeys([t for t in terms if t]))

    packed = []
    for doc, meta, dist in zip(docs, metas, dists):
        src = ""
        if isinstance(meta, dict):
            src = str(meta.get("source") or "")
        packed.append((src, dist, doc))

    seen_keys = set((src, str(doc)[:200]) for (src, _, doc) in packed)
    for t in terms[:6]:
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
            src = ""
            if isinstance(meta, dict):
                src = str(meta.get("source") or "")
            key = (src, str(doc)[:200])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            packed.append((src, None, doc))

    packed.sort(key=lambda x: (-_lexical_score(query, x[2]), x[1] if x[1] is not None else 1e9))
    packed = packed[:k]
    return [f"[source={src} distance={dist}]\n{doc}" for (src, dist, doc) in packed]


def _sanitize_history_for_llm(
    history: List[Dict[str, str]], max_messages: int = 12, max_chars: int = 3200
) -> List[Dict[str, str]]:
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


def _ollama_chat(messages: list, temperature: float = 0, num_ctx: int = 8192, timeout: int = 300) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
              "options": {"temperature": temperature, "num_ctx": num_ctx}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


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


def call_llm(
    query: str, chunks: List[str], chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    hist = _sanitize_history_for_llm(chat_history) if chat_history else []
    convo_block = _history_conversation_block(chat_history) if chat_history else ""
    num_ctx = 8192 if hist else 4096

    sys_rules = (
        "You answer questions ONLY from the provided transcript excerpts.\n\n"
        "STRICT RULES — violating any of these is a failure:\n"
        "- You have NO prior knowledge. The excerpts below are the ONLY facts that exist.\n"
        "- NEVER state anything not written in the excerpts — not even if you \"know\" it is true.\n"
        "- For every claim you make, quote the specific words from the excerpt that support it.\n"
        "- If the excerpts do not contain the answer, you MUST say: "
        "\"The provided excerpts do not contain information to answer this question.\"\n"
        "- Do NOT guess, infer, or fill gaps with outside knowledge."
    )
    if hist:
        sys_rules += (
            "\n- Prior turns help interpret follow-ups; every factual claim must still come from the excerpts."
        )

    if len(chunks) <= 3:
        numbered = "\n\n".join([f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(chunks)])
        messages: List[dict] = [{"role": "system", "content": sys_rules}]
        for m in hist:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"TRANSCRIPT EXCERPTS:\n{numbered}\n\n"
                    f"Instructions:\n"
                    f"1. Go through EACH excerpt and identify any information relevant to the question.\n"
                    f"2. For each relevant excerpt, quote the key passage.\n"
                    f"3. After reviewing ALL excerpts, combine the quoted evidence into a complete answer.\n"
                    f"4. If no excerpt answers the question, say so — do NOT make anything up."
                ),
            },
        )
        return _ollama_chat(messages, num_ctx=num_ctx, timeout=300)

    notes_parts = []
    prefix = ""
    if convo_block.strip():
        prefix = (
            "Recent conversation (use only to interpret what the question refers to):\n"
            f"{convo_block}\n\n"
        )
    for i, chunk in enumerate(chunks, 1):
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
                    f"[Excerpt {i}]\n{chunk}\n\n"
                    f"Extract relevant facts as bullet points with quotes:"
                ),
            },
        ]
        extracted = _ollama_chat(messages, num_ctx=4096, timeout=120)
        if "NOT_RELEVANT" not in extracted.upper():
            notes_parts.append(f"[From excerpt {i}]\n{extracted}")

    if not notes_parts:
        return "The provided excerpts do not contain information to answer this question."

    all_notes = "\n\n".join(notes_parts)
    reduce_prefix = ""
    if convo_block.strip():
        reduce_prefix = (
            "Recent conversation (interpret the question only; facts from notes):\n"
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
                f"{reduce_prefix}"
                f"Question: {query}\n\n"
                f"EXTRACTED NOTES FROM TRANSCRIPTS:\n{all_notes}\n\n"
                f"Combine ALL the notes above into a complete, well-organized answer. "
                f"Include quotes to support each point."
            ),
        },
    ]
    return _ollama_chat(messages, num_ctx=num_ctx, timeout=300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the Dhamma transcript RAG index.")
    parser.add_argument("--question", "-q", type=str, default="", help="Ask a single question and exit.")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--no-llm", action="store_true", help="Only retrieve chunks; do not call the LLM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_llm = not args.no_llm

    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Index not found at {PERSIST_DIR}. Run build_index.py first to create the vector store."
        )

    collection = get_collection()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def run_once(q: str, retrieval_query: str, hist: Optional[List[Dict[str, str]]]) -> Optional[str]:
        try:
            chunks = retrieve(embed_model, collection, retrieval_query, k=args.k)
        except Exception as e:
            print(f"\nError while retrieving from the index: {e}")
            return None

        print("\nTop matching chunks:\n")
        for i, chunk in enumerate(chunks, start=1):
            print(f"--- Chunk {i} ---")
            print(chunk[:2000])
            print()

        if use_llm:
            try:
                answer = call_llm(q, chunks, chat_history=hist if hist else None)
            except Exception as e:
                print(f"Error while calling Ollama: {e}")
                return None
            print("=== Answer ===")
            print(answer)
            return answer
        return ""

    if args.question.strip():
        run_once(args.question.strip(), args.question.strip(), None)
        return

    hist: List[Dict[str, str]] = []
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if not query or query.lower() in {"q", "quit", "exit"}:
            break

        retrieval_q = _rewrite_query_for_search(hist, query) if hist and use_llm else query
        if hist and use_llm and retrieval_q != query:
            print(f"(search query: {retrieval_q})")

        ans = run_once(query, retrieval_q, hist if hist else None)
        if ans is not None and use_llm:
            hist.append({"role": "user", "content": query})
            hist.append({"role": "assistant", "content": ans})
            hist = hist[-24:]


if __name__ == "__main__":
    main()

