"""
Vertex corpus builders (same chunking as Chroma indexers), embeddings via Vertex AI.

Monolithic (legacy):
  python an1_build_vertex_bundle.py [--out PATH] [--upload gs://bucket/obj.json]

Manifest + shards (incremental books; loader: AN1_VERTEX_MANIFEST_* / auto rag_index/vertex_corpus/manifest.json):
  python an1_build_vertex_bundle.py --shards DIR [--upload-base gs://bucket/prefix/]

Requires: GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_REGION, processed scipts2/an1.json; an2.json optional for AN2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from an1_build_index import (
    AN1_PATH,
    AN2_PATH,
    BASE_DIR,
    PERSIST_DIR,
    _extract_records_fallback,
    _parse_json_lenient,
    _record_to_docs,
    _record_to_docs_an2,
    read_in_chunks_from_string,
)

# Import after path — uses Vertex on first embed
import an1_vertex_core as vx


def _gather_chunk_specs_an1(data: List[Dict[str, Any]]) -> List[Tuple[str, str, str, str, str, str]]:
    """Return list of (book, kind, suttaid, commentary_id, source_rel, text)."""
    out: List[Tuple[str, str, str, str, str, str]] = []
    source_rel = str(AN1_PATH.relative_to(BASE_DIR)).replace("\\", "/")
    for rec in data:
        if not isinstance(rec, dict):
            continue
        for doc_kind, suttaid, comm_id, doc_text in _record_to_docs(rec):
            if not (doc_text or "").strip():
                continue
            for chunk in read_in_chunks_from_string(doc_text):
                out.append(("an1", doc_kind, suttaid, comm_id, source_rel, chunk))
    return out


def _gather_chunk_specs_an2(data: List[Dict[str, Any]]) -> List[Tuple[str, str, str, str, str, str]]:
    out: List[Tuple[str, str, str, str, str, str]] = []
    source_rel = str(AN2_PATH.relative_to(BASE_DIR)).replace("\\", "/")
    for rec in data:
        if not isinstance(rec, dict):
            continue
        for doc_kind, suttaid, comm_id, doc_text in _record_to_docs_an2(rec):
            if not (doc_text or "").strip():
                continue
            for chunk in read_in_chunks_from_string(doc_text):
                out.append(("an2", doc_kind, suttaid, comm_id, source_rel, chunk))
    return out


def specs_for_book(book: str) -> List[Tuple[str, str, str, str, str, str]]:
    b = (book or "").strip().lower()
    if b == "an1":
        if not AN1_PATH.exists():
            raise FileNotFoundError(f"Missing an1.json at: {AN1_PATH}")
        raw1 = AN1_PATH.read_text(encoding="utf-8", errors="ignore")
        try:
            parsed1 = _parse_json_lenient(raw1)
        except Exception:
            parsed1 = _extract_records_fallback(raw1)
        if not isinstance(parsed1, list):
            raise ValueError("Expected an1.json to be a JSON list of records.")
        return _gather_chunk_specs_an1(parsed1)
    if b == "an2":
        if not AN2_PATH.exists():
            raise FileNotFoundError(f"Missing an2.json at: {AN2_PATH}")
        raw2 = AN2_PATH.read_text(encoding="utf-8", errors="ignore")
        try:
            parsed2 = _parse_json_lenient(raw2)
        except Exception:
            parsed2 = _extract_records_fallback(raw2)
        if not isinstance(parsed2, list):
            raise ValueError("Expected an2.json to be a JSON list of records.")
        return _gather_chunk_specs_an2(parsed2)
    raise ValueError(f"Unknown book {book!r} (supported: an1, an2)")


def build_shard_dict(book: str) -> Dict[str, Any]:
    """One book: chunk records with Vertex embeddings (shard file body)."""
    b = (book or "").strip().lower()
    specs = specs_for_book(b)
    texts = [s[5] for s in specs]
    vectors = vx.embed_texts_vertex(texts)
    chunks: List[Dict[str, Any]] = []
    for spec, emb in zip(specs, vectors):
        book_id, kind, suttaid, comm_id, source_rel, text = spec
        chunks.append(
            {
                "book": book_id,
                "kind": kind,
                "suttaid": suttaid,
                "commentary_id": comm_id,
                "source": source_rel,
                "text": text,
                "embedding": emb,
            }
        )
    return {
        "format": vx.SHARD_FORMAT_V1,
        "book": b,
        "embedding_model": vx.embedding_model_name(),
        "chunks": chunks,
    }


def write_vertex_corpus(out_dir: Path, *, upload_base_gcs: str = "") -> Path:
    """
    Write manifest.json + shard_an1.json [+ shard_an2.json] under out_dir.
    If upload_base_gcs is set (e.g. gs://bucket/prefix/), upload each file to that prefix.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    em = vx.embedding_model_name()
    shard_entries: List[Dict[str, Any]] = []
    for book in ("an1", "an2"):
        if book == "an2" and not AN2_PATH.exists():
            continue
        shard_name = f"shard_{book}.json"
        shard_path = out_dir / shard_name
        sd = build_shard_dict(book)
        shard_path.write_text(json.dumps(sd, ensure_ascii=False), encoding="utf-8")
        shard_entries.append(
            {"book": book, "uri": shard_name, "chunk_count": len(sd.get("chunks") or [])}
        )
        if upload_base_gcs.strip():
            dest = upload_base_gcs.strip().rstrip("/") + "/" + shard_name
            vx.upload_bundle_file(shard_path, dest)
    manifest: Dict[str, Any] = {
        "format": vx.MANIFEST_FORMAT_V1,
        "version": 1,
        "embedding_model": em,
        "shards": shard_entries,
    }
    mp = out_dir / "manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    if upload_base_gcs.strip():
        dest_m = upload_base_gcs.strip().rstrip("/") + "/manifest.json"
        vx.upload_bundle_file(mp, dest_m)
    return mp


def build_bundle_dict() -> Dict[str, Any]:
    if not AN1_PATH.exists():
        raise FileNotFoundError(f"Missing an1.json at: {AN1_PATH}")

    specs: List[Tuple[str, str, str, str, str, str]] = specs_for_book("an1")
    if AN2_PATH.exists():
        specs.extend(specs_for_book("an2"))

    texts = [s[5] for s in specs]
    vectors = vx.embed_texts_vertex(texts)

    chunks: List[Dict[str, Any]] = []
    for spec, emb in zip(specs, vectors):
        book, kind, suttaid, comm_id, source_rel, text = spec
        chunks.append(
            {
                "book": book,
                "kind": kind,
                "suttaid": suttaid,
                "commentary_id": comm_id,
                "source": source_rel,
                "text": text,
                "embedding": emb,
            }
        )

    return {
        "format": "an1_vertex_bundle_v2",
        "embedding_model": vx.embedding_model_name(),
        "chunks": chunks,
    }


def write_bundle(out_path: Path, upload_gs_uri: str = "") -> Path:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    if upload_gs_uri.strip():
        vx.upload_bundle_file(out_path, upload_gs_uri.strip())
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build Vertex corpus: legacy monolithic JSON or manifest+shards (incremental-friendly)."
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help=f"Monolithic bundle output path (default: {PERSIST_DIR / 'an1_vertex_bundle.json'})",
    )
    p.add_argument(
        "--upload",
        type=str,
        default="",
        help="If set with monolithic build, upload single bundle to this gs:// URI.",
    )
    p.add_argument(
        "--shards",
        type=str,
        default="",
        help="If set, write manifest.json + shard_an*.json under this directory (embed per book).",
    )
    p.add_argument(
        "--upload-base",
        type=str,
        default="",
        help="With --shards: after writing locally, upload each file to this gs://bucket/prefix/ (trailing slash optional).",
    )
    args = p.parse_args(argv)

    if (args.shards or "").strip():
        out_dir = Path(args.shards.strip())
        mp = write_vertex_corpus(out_dir, upload_base_gcs=(args.upload_base or "").strip())
        print(f"Wrote manifest: {mp}")
        if (args.upload_base or "").strip():
            print(f"Uploaded corpus to {(args.upload_base or '').strip().rstrip('/')}/")
        return 0

    out_path = Path(args.out) if args.out.strip() else (PERSIST_DIR / "an1_vertex_bundle.json")
    path = write_bundle(out_path, upload_gs_uri=args.upload)
    print(f"Wrote bundle: {path} ({path.stat().st_size} bytes)")
    if args.upload.strip():
        print(f"Uploaded to {args.upload.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
