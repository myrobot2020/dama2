import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "rag_index"
COLLECTION_NAME = "dama_transcripts"

# Chat exports in merged JSONL (one JSON object per line, `messages` list).
CHAT_JSONL_GLOBS = (
    "training-data/*.chat.jsonl",
    "cursor-export/*.chat.jsonl",
    "grok-export/*.chat.jsonl",
)


def iter_transcript_files(base_dir: Path) -> List[Path]:
    """
    Prefer indexing the per-talk `.txt` files in the folder.

    `all_transcripts.txt` can be extremely large and slow to embed; we skip it by default.
    """
    files = sorted(
        [
            p
            for p in base_dir.glob("*.txt")
            if p.is_file() and p.name.lower() != "all_transcripts.txt"
        ]
    )
    return files


def iter_chat_jsonl_files(base_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for pattern in CHAT_JSONL_GLOBS:
        paths.extend(base_dir.glob(pattern))
    # Stable unique order
    return sorted({p.resolve() for p in paths if p.is_file()})


def conversation_to_text(obj: Dict[str, Any]) -> Optional[str]:
    """Flatten one chat-jsonl record into plain text for chunking."""
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    src = str(obj.get("source") or "")
    cid = str(obj.get("conversation_id") or "")
    header = f"[source={src} conversation_id={cid}]"
    parts: List[str] = [header]
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "unknown")
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        parts.append(f"{role.upper()}: {content.strip()}")
    if len(parts) <= 1:
        return None
    return "\n\n".join(parts)


def read_in_chunks(path: Path, chunk_size: int = 1500, overlap: int = 400):
    """
    Read the file and yield overlapping text chunks of roughly `chunk_size` characters.
    Splits on sentence boundaries ('. ') when possible, falling back to space boundaries,
    so chunks work well even for transcripts with no line breaks.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return

    start = 0
    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:]
            if chunk.strip():
                yield chunk
            break

        # Try to break at a sentence boundary ('. ') near the end of the chunk.
        boundary = text.rfind(". ", start + chunk_size // 2, end + 200)
        if boundary != -1:
            end = boundary + 2
        else:
            # Fall back to a space boundary.
            boundary = text.rfind(" ", start + chunk_size // 2, end + 100)
            if boundary != -1:
                end = boundary + 1

        yield text[start:end]
        start = end - overlap


def iter_jsonl_conversation_texts(path: Path) -> Iterable[Tuple[str, str]]:
    """
    Yield (conversation_id, flattened_text) for each valid line in a chat JSONL file.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  skip {path.name}:{line_no} (invalid JSON)", flush=True)
                continue
            if not isinstance(obj, dict):
                continue
            text = conversation_to_text(obj)
            if not text:
                continue
            cid = str(obj.get("conversation_id") or f"{path.stem}-line-{line_no}")
            yield cid, text


def main() -> None:
    transcript_files = iter_transcript_files(BASE_DIR)
    chat_files = iter_chat_jsonl_files(BASE_DIR)
    if not transcript_files and not chat_files:
        raise FileNotFoundError(
            f"No sources found under {BASE_DIR}.\n"
            "- Add per-talk *.txt transcripts in the repo root, and/or\n"
            "- Add *.chat.jsonl under training-data/, cursor-export/, or grok-export/."
        )

    os.makedirs(PERSIST_DIR, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Reset the collection if it exists so you can rebuild cleanly.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(
        f"Building index: {len(transcript_files)} .txt file(s), "
        f"{len(chat_files)} chat .jsonl file(s) …",
        flush=True,
    )

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

    for file_path in transcript_files:
        file_chunk_index = 0
        for doc in tqdm(
            read_in_chunks(file_path),
            desc=f"TXT {file_path.name}",
            leave=False,
        ):
            ids.append(f"txt-{file_path.stem}-c{file_chunk_index}-g{global_chunk_index}")
            metadatas.append(
                {
                    "source": file_path.name,
                    "conversation_id": "",
                    "kind": "transcript_txt",
                    "chunk_index": file_chunk_index,
                    "global_chunk_index": global_chunk_index,
                }
            )
            texts.append(doc)
            file_chunk_index += 1
            global_chunk_index += 1

            if len(texts) >= batch_size:
                flush_batch()

    for file_path in chat_files:
        for conv_id, conv_text in tqdm(
            list(iter_jsonl_conversation_texts(file_path)),
            desc=f"CHAT {file_path.name}",
            leave=False,
        ):
            file_chunk_index = 0
            for doc in read_in_chunks_from_string(conv_text):
                safe_cid = conv_id.replace("/", "-")[:120]
                ids.append(f"chat-{file_path.stem}-c{file_chunk_index}-g{global_chunk_index}")
                metadatas.append(
                    {
                        "source": file_path.name,
                        "conversation_id": conv_id,
                        "kind": "chat_jsonl",
                        "chunk_index": file_chunk_index,
                        "global_chunk_index": global_chunk_index,
                    }
                )
                texts.append(doc)
                file_chunk_index += 1
                global_chunk_index += 1

                if len(texts) >= batch_size:
                    flush_batch()

    flush_batch()
    print(f"Done. Index persisted in {PERSIST_DIR}", flush=True)


def read_in_chunks_from_string(
    text: str, chunk_size: int = 1500, overlap: int = 400
) -> Iterable[str]:
    """Same chunking as read_in_chunks but from an in-memory string."""
    if not text.strip():
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


if __name__ == "__main__":
    main()
