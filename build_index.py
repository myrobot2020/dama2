import os
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "rag_index"
COLLECTION_NAME = "dama_transcripts"


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


def main() -> None:
    transcript_files = iter_transcript_files(BASE_DIR)
    if not transcript_files:
        raise FileNotFoundError(
            f"No transcript .txt files found in {BASE_DIR}. "
            "Expected per-talk .txt files (and optionally all_transcripts.txt)."
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

    print(f"Building index from {len(transcript_files)} transcript files in {BASE_DIR} ...")

    batch_size = 64
    ids: List[str] = []
    metadatas = []
    texts = []

    global_chunk_index = 0
    for file_path in transcript_files:
        file_chunk_index = 0
        for doc in tqdm(
            read_in_chunks(file_path),
            desc=f"Indexing {file_path.name}",
            leave=False,
        ):
            ids.append(f"{file_path.stem}-chunk-{file_chunk_index}")
            metadatas.append(
                {
                    "source": file_path.name,
                    "chunk_index": file_chunk_index,
                    "global_chunk_index": global_chunk_index,
                }
            )
            texts.append(doc)
            file_chunk_index += 1
            global_chunk_index += 1

            if len(texts) >= batch_size:
                embeddings = model.encode(texts, show_progress_bar=False).tolist()
                collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
                ids, metadatas, texts = [], [], []

    if texts:
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    print(f"Done. Index persisted in {PERSIST_DIR}")


if __name__ == "__main__":
    main()

