"""
Vertex-only path for AN1 RAG: embeddings + Gemini chat, GCS or local JSON bundle (no Chroma/torch).

Enable with AN1_USE_VERTEX=1 or DAMA_USE_VERTEX=1 (same flags as topic_search_server).
Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_REGION (default us-central1).

Bundle: AN1_VERTEX_BUNDLE_GCS_URI=gs://bucket/path/an1_vertex_bundle.json
   or AN1_VERTEX_BUNDLE_PATH=/path/to/an1_vertex_bundle.json
   default local: <rag_index_an1>/an1_vertex_bundle.json
"""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

_bundle_lock = threading.RLock()
_bundle: Optional[Dict[str, Any]] = None
_embed_model_obj: Any = None


def an1_vertex_enabled() -> bool:
    v = os.environ.get("AN1_USE_VERTEX", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    v = os.environ.get("DAMA_USE_VERTEX", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def vertex_project_and_location() -> Tuple[str, str]:
    project = (
        os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.environ.get("GCP_PROJECT", "").strip()
    )
    location = (
        os.environ.get("GOOGLE_CLOUD_REGION", "").strip()
        or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        or "us-central1"
    )
    return project, location


def embedding_model_name() -> str:
    return os.environ.get("AN1_VERTEX_EMBEDDING_MODEL", "").strip() or "textembedding-gecko@003"


def chat_model_name() -> str:
    return os.environ.get("DAMA_VERTEX_MODEL", "").strip() or os.environ.get("AN1_VERTEX_MODEL", "").strip() or "gemini-2.5-flash"


def max_output_tokens() -> int:
    try:
        return int(os.environ.get("DAMA_MAX_OUTPUT_TOKENS", "").strip() or "2048")
    except ValueError:
        return 2048


def _parse_gcs_object_uri(uri: str) -> Tuple[str, str]:
    u = (uri or "").strip()
    if not u.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    p = urlparse(u)
    bucket = p.netloc
    blob_path = (p.path or "").lstrip("/")
    if not bucket or not blob_path:
        raise ValueError(f"Invalid GCS object URI: {uri!r}")
    return bucket, blob_path


def _download_gcs_to_bytes(gs_uri: str) -> bytes:
    from google.cloud import storage  # type: ignore

    bucket_name, blob_path = _parse_gcs_object_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()


def _upload_bytes_to_gcs(gs_uri: str, data: bytes, content_type: str = "application/json") -> None:
    from google.cloud import storage  # type: ignore

    bucket_name, blob_path = _parse_gcs_object_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)


def upload_bundle_file(local_path: Path, gs_uri: str) -> None:
    raw = local_path.read_bytes()
    _upload_bytes_to_gcs(gs_uri, raw)


def _get_embed_model() -> Any:
    global _embed_model_obj
    if _embed_model_obj is None:
        import vertexai  # type: ignore
        from vertexai.language_models import TextEmbeddingModel  # type: ignore

        project, location = vertex_project_and_location()
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT (or GCP_PROJECT) is required for Vertex embeddings.")
        vertexai.init(project=project, location=location)
        _embed_model_obj = TextEmbeddingModel.from_pretrained(embedding_model_name())
    return _embed_model_obj


def embed_texts_vertex(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = _get_embed_model()
    out_vectors: List[List[float]] = []
    batch = 32
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        embs = model.get_embeddings(chunk)
        for e in embs:
            vals = getattr(e, "values", None)
            if vals is None:
                raise RuntimeError("Vertex embedding response missing .values")
            out_vectors.append(list(vals))
    return out_vectors


def cosine_distance(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1e9
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1e9
    sim = dot / (na * nb)
    return float(1.0 - sim)


def gemini_generate(system_text: str, user_text: str, temperature: float = 0.2) -> str:
    import vertexai  # type: ignore
    from vertexai.generative_models import GenerativeModel  # type: ignore

    project, location = vertex_project_and_location()
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT (or GCP_PROJECT) is required for Vertex chat.")
    vertexai.init(project=project, location=location)
    gm = GenerativeModel(chat_model_name())
    prompt = f"[SYSTEM]\n{system_text}\n\n[USER]\n{user_text}"
    resp = gm.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens(),
        },
    )
    return (getattr(resp, "text", None) or "").strip()


def invalidate_bundle_cache() -> None:
    global _bundle
    with _bundle_lock:
        _bundle = None


def _default_bundle_path(persist_dir: Path) -> Path:
    return persist_dir / "an1_vertex_bundle.json"


def _load_bundle_from_path(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Bundle must be a JSON object.")
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Bundle missing non-empty 'chunks' array.")
    return data


def ensure_bundle_loaded(persist_dir: Path) -> Dict[str, Any]:
    """
    Load vertex bundle once (thread-safe). Resolution order:
    1) AN1_VERTEX_BUNDLE_PATH if set and file exists
    2) AN1_VERTEX_BUNDLE_GCS_URI download to persist_dir cache file
    3) default local persist_dir/an1_vertex_bundle.json
    """
    global _bundle
    with _bundle_lock:
        if _bundle is not None:
            return _bundle

        env_path = os.environ.get("AN1_VERTEX_BUNDLE_PATH", "").strip()
        gcs_uri = os.environ.get("AN1_VERTEX_BUNDLE_GCS_URI", "").strip()
        default_local = _default_bundle_path(persist_dir)

        if env_path:
            p = Path(env_path)
            if not p.is_file():
                raise FileNotFoundError(f"AN1_VERTEX_BUNDLE_PATH not found: {p}")
            _bundle = _load_bundle_from_path(p)
            return _bundle

        if gcs_uri:
            persist_dir.mkdir(parents=True, exist_ok=True)
            cache_file = persist_dir / "_vertex_bundle_cache.json"
            raw = _download_gcs_to_bytes(gcs_uri)
            cache_file.write_bytes(raw)
            _bundle = _load_bundle_from_path(cache_file)
            return _bundle

        if default_local.is_file():
            _bundle = _load_bundle_from_path(default_local)
            return _bundle

        raise FileNotFoundError(
            "Vertex mode: no embedding bundle. Set AN1_VERTEX_BUNDLE_GCS_URI, AN1_VERTEX_BUNDLE_PATH, "
            f"or place a bundle at {default_local}. Build with: python an1_build_vertex_bundle.py"
        )


def bundle_chunk_rows(bundle: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    b = bundle if bundle is not None else _bundle
    if not b:
        return []
    rows = b.get("chunks")
    if not isinstance(rows, list):
        return []
    return [x for x in rows if isinstance(x, dict)]


def bundle_meta_for_status(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "embedding_model": bundle.get("embedding_model") or "",
        "format": bundle.get("format") or "",
    }
