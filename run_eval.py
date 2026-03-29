"""
Run self-made eval questions from eval_results.txt against the local API.

Usage (server on 127.0.0.1:8000):
  python run_eval.py
  set USE_LLM=1 && python run_eval.py   # include Ollama synthesis (slower)
  set EVAL_AN_BOOK=1 && python run_eval.py   # pass an_book filter to /api/query
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import requests

BASE = os.environ.get("EVAL_BASE", "http://127.0.0.1:8000").rstrip("/")
USE_LLM = os.environ.get("USE_LLM", "").lower() in ("1", "true", "yes")
K = int(os.environ.get("EVAL_K", "5"))
EVAL_AN_BOOK_RAW = os.environ.get("EVAL_AN_BOOK", "").strip()


def load_questions(eval_path: Path) -> list[str]:
    # utf-8-sig strips BOM so ^===Q1=== matches the first line
    text = eval_path.read_text(encoding="utf-8-sig", errors="replace")
    return [m.strip() for m in re.findall(r"^===Q\d+===\s*(.+)$", text, re.MULTILINE) if m.strip()]


# Loose retrieval hints (any match in combined top-k chunk text = pass hint)
HINTS: list[list[str]] = [
    ["samatha", "vipassana", "jhana", "insight", "concentrat"],
    ["500", "pollut", "wrong", "teaching"],
    ["four", "arah", "arhat"],
    ["stream", "sutta", "sutra"],
    ["sangha", "dhamma", "dharma"],
    ["worthy", "aryan", "arah", "offering"],
]


def combined_chunks_text(data: dict) -> str:
    parts = []
    for c in data.get("chunks") or []:
        parts.append(str(c.get("text", "")))
        parts.append(str(c.get("source", "")))
    return " ".join(parts).lower()


def main() -> int:
    root = Path(__file__).resolve().parent
    eval_file = root / "eval_results.txt"
    if not eval_file.exists():
        print("Missing eval_results.txt", file=sys.stderr)
        return 1

    questions = load_questions(eval_file)
    if not questions:
        print("No ===Qn=== questions found in eval_results.txt", file=sys.stderr)
        return 1

    eval_an_book: int | None = None
    if EVAL_AN_BOOK_RAW:
        try:
            eval_an_book = int(EVAL_AN_BOOK_RAW)
        except ValueError:
            eval_an_book = None
    print(f"Base URL: {BASE}  use_llm={USE_LLM}  k={K}  an_book={eval_an_book}")
    fails = 0

    for i, q in enumerate(questions):
        payload: dict = {"question": q, "k": K, "use_llm": USE_LLM}
        if eval_an_book is not None and eval_an_book >= 1:
            payload["an_book"] = eval_an_book
        r = requests.post(
            f"{BASE}/api/query",
            json=payload,
            timeout=600 if USE_LLM else 120,
        )
        tag = f"Q{i + 1}"
        if r.status_code != 200:
            print(f"FAIL {tag} HTTP {r.status_code}: {r.text[:300]}")
            fails += 1
            continue
        data = r.json()
        chunks = data.get("chunks") or []
        n = len(chunks)
        if n < 1:
            print(f"FAIL {tag}: no chunks")
            fails += 1
            continue
        blob = combined_chunks_text(data)
        hint_ok = True
        if i < len(HINTS):
            need = HINTS[i]
            hint_ok = any(h in blob for h in need)
        llm_note = ""
        if USE_LLM:
            ans = (data.get("answer") or "").strip()
            llm_note = f" answer_len={len(ans)}"
        hint_note = " hints_ok" if hint_ok else " HINT_MISS"
        if not hint_ok:
            fails += 1
        elapsed = data.get("elapsed_ms")
        time_note = f" time={float(elapsed):.0f}ms" if isinstance(elapsed, (int, float)) else ""
        llm_t = data.get("llm_ms")
        if USE_LLM and isinstance(llm_t, (int, float)) and llm_t > 0:
            time_note += f" llm={llm_t:.0f}ms"
        print(f"OK {tag} chunks={n}{hint_note}{time_note}{llm_note}")

    # Conversation memory smoke test (rewrite path): session + vague follow-up
    try:
        s = requests.post(f"{BASE}/api/sessions", timeout=30)
        s.raise_for_status()
        sid = s.json()["session_id"]
        r1 = requests.post(
            f"{BASE}/api/query",
            json={
                "question": "What does the Anguttara Nikaya say about samatha and vipassana?",
                "k": K,
                "use_llm": False,
                "session_id": sid,
            },
            timeout=120,
        )
        r1.raise_for_status()
        r2 = requests.post(
            f"{BASE}/api/query",
            json={
                "question": "Give one concrete quote the teacher used about that.",
                "k": K,
                "use_llm": False,
                "session_id": sid,
            },
            timeout=120,
        )
        r2.raise_for_status()
        d1, d2 = r1.json(), r2.json()
        c2 = combined_chunks_text(d2)
        t1 = d1.get("elapsed_ms")
        t2 = d2.get("elapsed_ms")
        mem_time = ""
        if isinstance(t1, (int, float)) and isinstance(t2, (int, float)):
            mem_time = f" (q1={t1:.0f}ms q2={t2:.0f}ms)"
        if "samatha" in c2 or "vipassana" in c2 or "jhana" in c2 or "excerpt" in c2:
            print("OK MEM vague_followup retrieval still on-topic" + mem_time)
        else:
            print("WARN MEM vague_followup: top chunks may be off-topic (check manually)")
    except Exception as e:
        print(f"WARN MEM session test failed: {e}")

    if fails:
        print(f"\nDone with {fails} failing check(s).")
        return 1
    print("\nAll automated checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
