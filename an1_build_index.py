import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Chroma / sentence_transformers are imported inside main() only so an1_app can load this module
# in the slim Vertex Docker image (no local index deps at runtime).

BASE_DIR = Path(__file__).resolve().parent
AN1_PATH = BASE_DIR / "processed scipts2" / "an1.json"
PERSIST_DIR = BASE_DIR / "rag_index_an1"
COLLECTION_NAME = "an1_sutta"

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


def read_in_chunks_from_string(text: str, chunk_size: int = 1500, overlap: int = 400) -> Iterable[str]:
    if not (text or "").strip():
        return
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:]
            if chunk.strip():
                yield chunk
            break

        boundary = text.rfind(". ", start + chunk_size // 2, end + 200)
        if boundary != -1:
            end = boundary + 2
        else:
            boundary = text.rfind(" ", start + chunk_size // 2, end + 100)
            if boundary != -1:
                end = boundary + 1

        yield text[start:end]
        start = end - overlap


def _commentary_body(rec: Dict[str, Any]) -> str:
    return str(rec.get("commentary") or rec.get("commentry") or "").strip()


def _commentary_id(rec: Dict[str, Any]) -> str:
    cid = str(rec.get("commentary_id") or "").strip()
    if cid:
        return cid
    sid = str(rec.get("suttaid") or "").strip()
    if sid.startswith("AN "):
        return "c" + sid
    return ("c" + sid) if sid else ""


def _parse_json_lenient(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        pass
    rge = __import__("re")
    s = raw.lstrip("\ufeff")
    s = rge.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    s = rge.sub(
        r'("sutta"\s*:\s*"(?:[^"\\\\]|\\\\.)*")(\s*\n\s*"commentry"\s*:)',
        r"\1,\2",
        s,
    )
    s = rge.sub(
        r'("sutta"\s*:\s*"(?:[^"\\\\]|\\\\.)*")(\s*\n\s*"commentary_id"\s*:)',
        r"\1,\2",
        s,
    )
    s = rge.sub(
        r'("commentary_id"\s*:\s*"(?:[^"\\\\]|\\\\.)*")(\s*\n\s*"commentary"\s*:)',
        r"\1,\2",
        s,
    )
    s = rge.sub(
        r'("sutta"\s*:\s*"(?:[^"\\\\]|\\\\.)*")(\s*\n\s*"commentary"\s*:)',
        r"\1,\2",
        s,
    )
    s = rge.sub(r",(\s*[}\]])", r"\1", s)
    return json.loads(s)


def _extract_records_fallback(raw: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    cur: Dict[str, str] = {}
    mode: Optional[str] = None

    def flush():
        nonlocal cur, mode
        comm = (cur.get("commentary") or cur.get("commentry") or "").strip()
        if cur.get("suttaid") and (cur.get("sutta") or comm):
            row: Dict[str, str] = {
                "suttaid": cur.get("suttaid", "").strip(),
                "sutta": (cur.get("sutta") or "").strip(),
                "commentry": comm,
            }
            cid = (cur.get("commentary_id") or "").strip()
            if cid:
                row["commentary_id"] = cid
            records.append(row)
        cur = {}
        mode = None

    def unquote(s: str) -> str:
        s = (s or "").strip().rstrip(",")
        if s.startswith('"') and s.endswith('"') and len(s) >= 2:
            return s[1:-1]
        return s

    rgx = __import__("re")
    for line in (raw or "").splitlines():
        ln = line.strip()
        if not ln:
            continue

        if '"suttaid"' in ln:
            flush()
            m = rgx.search(r'"suttaid"\s*:\s*"([^"]+)"', ln)
            if m:
                cur["suttaid"] = m.group(1)
            mode = None
            continue

        if '"commentary_id"' in ln:
            m_id = rgx.search(r'"commentary_id"\s*:\s*"([^"]*)"', ln)
            if m_id:
                cur["commentary_id"] = m_id.group(1)
            continue

        m = rgx.search(r'"sutta"\s*:\s*"(.*)"\s*,?\s*$', ln)
        if m:
            cur["sutta"] = m.group(1)
            mode = "sutta"
            continue

        if '"commentry"' in ln:
            m1 = rgx.search(r'"commentry"\s*:\s*"(.*)"\s*,?\s*$', ln)
            if m1:
                cur["commentry"] = m1.group(1)
            else:
                m2 = rgx.search(r'"commentry"\s*:\s*(.*)\s*$', ln)
                if m2:
                    cur["commentry"] = unquote(m2.group(1))
            mode = "commentry"
            continue

        if '"commentary"' in ln and '"commentary_id"' not in ln:
            m1 = rgx.search(r'"commentary"\s*:\s*"(.*)"\s*,?\s*$', ln)
            if m1:
                cur["commentary"] = m1.group(1)
            else:
                m2 = rgx.search(r'"commentary"\s*:\s*(.*)\s*$', ln)
                if m2:
                    cur["commentary"] = unquote(m2.group(1))
            mode = "commentary"
            continue

        if mode == "sutta":
            cur["sutta"] = (cur.get("sutta") or "") + "\n" + unquote(ln)
        elif mode == "commentry":
            cur["commentry"] = (cur.get("commentry") or "") + "\n" + unquote(ln)
        elif mode == "commentary":
            cur["commentary"] = (cur.get("commentary") or "") + "\n" + unquote(ln)

    flush()
    return records


def _record_to_text(record: Dict[str, Any]) -> Tuple[str, str, str]:
    suttaid = str(record.get("suttaid") or "").strip()
    sutta = str(record.get("sutta") or "").strip()
    commentary = _commentary_body(record)
    cid = _commentary_id(record)
    combined = f"SUTTAID: {suttaid}\nCOMMENTARY_ID: {cid}\n\nSUTTA:\n{sutta}"
    if commentary:
        combined += f"\n\nTEACHER COMMENTARY:\n{commentary}"
    return suttaid, sutta, combined


def _record_to_docs(record: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """
    Return a list of (kind, suttaid, commentary_id, text) docs so we can attribute citations.
    kind: 'sutta' | 'commentary' | 'combined'
    """
    suttaid = str(record.get("suttaid") or "").strip()
    cid = _commentary_id(record)
    sutta = str(record.get("sutta") or "").strip()
    commentary = _commentary_body(record)
    out: List[Tuple[str, str, str, str]] = []
    if suttaid and sutta:
        out.append(
            ("sutta", suttaid, cid, f"SUTTAID: {suttaid}\nCOMMENTARY_ID: {cid}\n\nSUTTA:\n{sutta}")
        )
    if suttaid and commentary:
        out.append(
            (
                "commentary",
                suttaid,
                cid,
                f"SUTTAID: {suttaid}\nCOMMENTARY_ID: {cid}\n\nTEACHER COMMENTARY:\n{commentary}",
            )
        )
    if not out and suttaid:
        # fallback: keep something indexable
        combined = f"SUTTAID: {suttaid}\nCOMMENTARY_ID: {cid}\n\nSUTTA:\n{sutta}"
        if commentary:
            combined += f"\n\nTEACHER COMMENTARY:\n{commentary}"
        out.append(("combined", suttaid, cid, combined))
    return out


def main() -> None:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    _dbg("H1", "an1_build_index.py:main", "Start build", {"an1_path": str(AN1_PATH), "persist_dir": str(PERSIST_DIR), "collection": COLLECTION_NAME})

    if not AN1_PATH.exists():
        _dbg("H1", "an1_build_index.py:main", "an1.json missing", {"an1_path": str(AN1_PATH)})
        raise FileNotFoundError(f"Missing an1.json at: {AN1_PATH}")

    os.makedirs(PERSIST_DIR, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    _dbg("H5", "an1_build_index.py:main", "Loading embedding model", {"model": "all-MiniLM-L6-v2"})
    model = SentenceTransformer("all-MiniLM-L6-v2")

    raw = AN1_PATH.read_text(encoding="utf-8", errors="ignore")
    try:
        data = _parse_json_lenient(raw)
    except Exception as e:
        _dbg("H1", "an1_build_index.py:main", "JSON parse failed", {"error": str(e), "bytes": len(raw)})
        data = _extract_records_fallback(raw)
        _dbg("H1", "an1_build_index.py:main", "Fallback extracted records", {"count": len(data)})

    if not isinstance(data, list):
        _dbg("H1", "an1_build_index.py:main", "Unexpected JSON shape", {"type": str(type(data))})
        raise ValueError("Expected an1.json to be a JSON list of records.")

    _dbg("H1", "an1_build_index.py:main", "Loaded records", {"count": len(data)})

    batch_size = 64
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    texts: List[str] = []

    global_chunk_index = 0

    def flush_batch() -> None:
        nonlocal ids, metadatas, texts
        if not texts:
            return
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        ids, metadatas, texts = [], [], []

    for rec_i, rec in enumerate(tqdm(data, desc="AN1 records")):
        if not isinstance(rec, dict):
            continue
        docs = _record_to_docs(rec)
        for doc_kind, suttaid, comm_id, doc_text in docs:
            if not doc_text.strip():
                continue
            chunk_index = 0
            for chunk in read_in_chunks_from_string(doc_text):
                ids.append(f"an1-{doc_kind}-{rec_i}-c{chunk_index}-g{global_chunk_index}")
                metadatas.append(
                    {
                        "source": str(AN1_PATH.relative_to(BASE_DIR)).replace("\\", "/"),
                        "suttaid": suttaid,
                        "commentary_id": comm_id,
                        "kind": doc_kind,
                        "chunk_index": chunk_index,
                        "global_chunk_index": global_chunk_index,
                    }
                )
                texts.append(chunk)
                chunk_index += 1
                global_chunk_index += 1
                if len(texts) >= batch_size:
                    flush_batch()

    flush_batch()
    _dbg("H2", "an1_build_index.py:main", "Build finished", {"collection_count": int(collection.count())})


if __name__ == "__main__":
    main()

