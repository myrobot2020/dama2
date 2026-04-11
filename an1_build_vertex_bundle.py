"""
Build an1_vertex_bundle.json: same chunks as an1_build_index (Chroma path), embeddings via Vertex AI.

Usage:
  set GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_REGION (or defaults)
  python an1_build_vertex_bundle.py [--out PATH] [--upload gs://bucket/obj.json]

Requires: google-cloud-aiplatform (vertexai), same an1.json as an1_build_index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import an1_build_index as an1_build
from an1_build_index import (
    AN1_PATH,
    BASE_DIR,
    PERSIST_DIR,
    _extract_records_fallback,
    _parse_json_lenient,
    _record_to_docs,
    read_in_chunks_from_string,
)

# Import after path — uses Vertex on first embed
import an1_vertex_core as vx


def _gather_chunk_specs(data: List[Dict[str, Any]]) -> List[Tuple[str, str, str, str, str]]:
    """Return list of (kind, suttaid, commentary_id, source_rel, text)."""
    out: List[Tuple[str, str, str, str, str]] = []
    source_rel = str(AN1_PATH.relative_to(BASE_DIR)).replace("\\", "/")
    for rec in data:
        if not isinstance(rec, dict):
            continue
        for doc_kind, suttaid, comm_id, doc_text in _record_to_docs(rec):
            if not (doc_text or "").strip():
                continue
            for chunk in read_in_chunks_from_string(doc_text):
                out.append((doc_kind, suttaid, comm_id, source_rel, chunk))
    return out


def build_bundle_dict() -> Dict[str, Any]:
    if not AN1_PATH.exists():
        raise FileNotFoundError(f"Missing an1.json at: {AN1_PATH}")

    raw = AN1_PATH.read_text(encoding="utf-8", errors="ignore")
    try:
        parsed = _parse_json_lenient(raw)
    except Exception:
        parsed = _extract_records_fallback(raw)
    if not isinstance(parsed, list):
        raise ValueError("Expected an1.json to be a JSON list of records.")

    specs = _gather_chunk_specs(parsed)
    texts = [s[4] for s in specs]
    vectors = vx.embed_texts_vertex(texts)

    chunks: List[Dict[str, Any]] = []
    for spec, emb in zip(specs, vectors):
        kind, suttaid, comm_id, source_rel, text = spec
        chunks.append(
            {
                "kind": kind,
                "suttaid": suttaid,
                "commentary_id": comm_id,
                "source": source_rel,
                "text": text,
                "embedding": emb,
            }
        )

    return {
        "format": "an1_vertex_bundle_v1",
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
    p = argparse.ArgumentParser(description="Build AN1 Vertex embedding bundle (JSON).")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help=f"Output JSON path (default: {PERSIST_DIR / 'an1_vertex_bundle.json'})",
    )
    p.add_argument("--upload", type=str, default="", help="If set, upload to gs://bucket/path.json after build.")
    args = p.parse_args(argv)

    out_path = Path(args.out) if args.out.strip() else (PERSIST_DIR / "an1_vertex_bundle.json")
    path = write_bundle(out_path, upload_gs_uri=args.upload)
    print(f"Wrote bundle: {path} ({path.stat().st_size} bytes)")
    if args.upload.strip():
        print(f"Uploaded to {args.upload.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
