import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lecture_metadata import parse_lecture_metadata, split_sutta_segments


BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "rag_index"
COLLECTION_NAME = "dama_transcripts"

DEFAULT_CHUNK = 1500
DEFAULT_OVERLAP = 400
DEFAULT_MAX_SEGMENT = 4000


def iter_transcript_files(base_dir: Path, only_prefix: str = "") -> List[Path]:
    """
    Prefer indexing the per-talk `.txt` files in the folder.
    Skips `all_transcripts.txt`. Optional `only_prefix` (e.g. `001_`) for fast iteration.
    """
    files = sorted(
        [
            p
            for p in base_dir.glob("*.txt")
            if p.is_file() and p.name.lower() != "all_transcripts.txt"
        ]
    )
    if only_prefix:
        files = [p for p in files if p.name.startswith(only_prefix)]
    return files


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[str]:
    """Yield overlapping chunks from an in-memory string (same rules as file chunking)."""
    text = text.strip()
    if not text:
        return
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        if end >= n:
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


def read_in_chunks(path: Path, chunk_size: int = DEFAULT_CHUNK, overlap: int = DEFAULT_OVERLAP) -> Iterator[str]:
    """Read file and yield chunks."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    yield from chunk_text(text, chunk_size, overlap)


def _subchunk_segment(
    segment: str, max_chars: int, chunk_size: int, overlap: int
) -> List[str]:
    """If segment is long, chunk it; else one element."""
    segment = segment.strip()
    if not segment:
        return []
    if len(segment) <= max_chars:
        return [segment]
    return list(chunk_text(segment, chunk_size, overlap))


def iter_file_chunks(
    file_path: Path,
    *,
    sutta_split: bool = True,
    chunk_size: int = DEFAULT_CHUNK,
    overlap: int = DEFAULT_OVERLAP,
    max_segment_chars: int = DEFAULT_MAX_SEGMENT,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Yield (document_text, chroma_metadata) per chunk.
    """
    base = parse_lecture_metadata(file_path)
    base = {
        **base,
        "source": file_path.name,
        "content_layer": "teacher_lecture",
    }

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    use_sutta = sutta_split and "### REFERENCE:" in text
    if not use_sutta:
        file_chunk_index = 0
        for doc in read_in_chunks(file_path, chunk_size, overlap):
            meta = {
                **base,
                "chunk_index": file_chunk_index,
                "an_sutta_ref": "",
                "sutta_chunk_part": 0,
            }
            yield doc, meta
            file_chunk_index += 1
        return

    segments = split_sutta_segments(text)
    file_chunk_index = 0
    for an_ref, seg_body in segments:
        pieces = _subchunk_segment(seg_body, max_segment_chars, chunk_size, overlap)
        for part_i, doc in enumerate(pieces):
            if not doc.strip():
                continue
            meta = {
                **base,
                "chunk_index": file_chunk_index,
                "an_sutta_ref": an_ref,
                "sutta_chunk_part": part_i,
            }
            yield doc, meta
            file_chunk_index += 1


def run_build(
    base_dir: Optional[Path] = None,
    only_prefix: str = "",
    sutta_split: bool = True,
) -> None:
    base_dir = base_dir or BASE_DIR
    transcript_files = iter_transcript_files(base_dir, only_prefix=only_prefix)
    if not transcript_files:
        raise FileNotFoundError(
            f"No transcript .txt files found in {base_dir} "
            f"(only_prefix={only_prefix!r}). "
            "Expected per-talk .txt files (and optionally all_transcripts.txt)."
        )

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

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(
        f"Building index from {len(transcript_files)} transcript files in {base_dir} "
        f"(only_prefix={only_prefix!r}, sutta_split={sutta_split}) ..."
    )

    batch_size = 64
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    texts: List[str] = []

    global_chunk_index = 0
    for file_path in transcript_files:
        file_chunk_index = 0
        for doc, meta in tqdm(
            iter_file_chunks(file_path, sutta_split=sutta_split),
            desc=f"Indexing {file_path.name}",
            leave=False,
        ):
            meta = dict(meta)
            meta["global_chunk_index"] = global_chunk_index
            stem = file_path.stem
            ref_tag = str(meta.get("an_sutta_ref") or "lecture").replace(" ", "_")[:80]
            part = int(meta.get("sutta_chunk_part") or 0)
            ids.append(f"{stem}-{ref_tag}-p{part}-c{file_chunk_index}")
            metadatas.append(meta)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Chroma transcript index with lecture + sutta metadata.")
    ap.add_argument(
        "--only-prefix",
        default="",
        help="Only index files whose basename starts with this (e.g. 001_)",
    )
    ap.add_argument(
        "--no-sutta-split",
        action="store_true",
        help="Disable ### REFERENCE-based segmentation; use plain sliding windows.",
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory containing *.txt transcripts",
    )
    args = ap.parse_args()
    run_build(
        base_dir=args.base_dir.resolve(),
        only_prefix=args.only_prefix,
        sutta_split=not args.no_sutta_split,
    )


if __name__ == "__main__":
    main()
