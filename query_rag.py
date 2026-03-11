import argparse
from pathlib import Path
from typing import List

import re
import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "rag_index"
COLLECTION_NAME = "dama_transcripts"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b"


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


def call_llm(query: str, chunks: List[str]) -> str:
    numbered = "\n\n".join(
        [f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(chunks)]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions ONLY from the provided transcript excerpts.\n\n"
                "STRICT RULES — violating any of these is a failure:\n"
                "- You have NO prior knowledge. The excerpts below are the ONLY facts that exist.\n"
                "- NEVER state anything not written in the excerpts — not even if you \"know\" it is true.\n"
                "- For every claim you make, quote the specific words from the excerpt that support it.\n"
                "- If the excerpts do not contain the answer, you MUST say: "
                "\"The provided excerpts do not contain information to answer this question.\"\n"
                "- Do NOT guess, infer, or fill gaps with outside knowledge."
            ),
        },
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
    ]
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0, "num_ctx": 8192}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


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

    def run_once(q: str) -> None:
        try:
            chunks = retrieve(embed_model, collection, q, k=args.k)
        except Exception as e:
            print(f"\nError while retrieving from the index: {e}")
            return

        print("\nTop matching chunks:\n")
        for i, chunk in enumerate(chunks, start=1):
            print(f"--- Chunk {i} ---")
            print(chunk[:2000])
            print()

        if use_llm:
            try:
                answer = call_llm(q, chunks)
            except Exception as e:
                print(f"Error while calling Ollama: {e}")
                return
            print("=== Answer ===")
            print(answer)

    if args.question.strip():
        run_once(args.question.strip())
        return

    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if not query or query.lower() in {"q", "quit", "exit"}:
            break

        run_once(query)


if __name__ == "__main__":
    main()

