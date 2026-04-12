"""
Vertex-only path for AN1 RAG: embeddings + Gemini chat, GCS or local JSON bundle (no Chroma/torch).

Enable with DAMA_RUNTIME=cloud, or AN1_USE_VERTEX=1 / DAMA_USE_VERTEX=1 (same flags as topic_search_server).
Use DAMA_RUNTIME=local to force local Chroma even if legacy vertex env vars are set.
Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_REGION (default us-central1).

Corpus (pick one style):

- **Manifest + shards (incremental books):**  
  `AN1_VERTEX_MANIFEST_GCS_URI=gs://bucket/prefix/manifest.json` or `AN1_VERTEX_MANIFEST_PATH=/path/manifest.json`  
  Manifest lists shard files (relative names or full `gs://` URIs). Same embedding model across shards.

- **Legacy single file:**  
  `AN1_VERTEX_BUNDLE_GCS_URI=gs://bucket/path/an1_vertex_bundle.json` or `AN1_VERTEX_BUNDLE_PATH=...`  
  default local: `<persist_dir>/an1_vertex_bundle.json`

Resolution order: manifest path, manifest GCS, bundle path, bundle GCS, default local bundle.
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
    """DAMA_RUNTIME=local forces Chroma; DAMA_RUNTIME=cloud forces Vertex. Otherwise AN1_USE_VERTEX / DAMA_USE_VERTEX."""
    r = os.environ.get("DAMA_RUNTIME", "").strip().lower()
    if r == "local":
        return False
    if r == "cloud":
        return True
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
    # textembedding-gecko@003 retired; text-embedding-005 works with TextEmbeddingModel.from_pretrained (768-dim default).
    return os.environ.get("AN1_VERTEX_EMBEDDING_MODEL", "").strip() or "text-embedding-005"


def chat_model_name() -> str:
    return os.environ.get("DAMA_VERTEX_MODEL", "").strip() or os.environ.get("AN1_VERTEX_MODEL", "").strip() or "gemini-2.5-flash"


def max_output_tokens() -> int:
    try:
        return int(os.environ.get("DAMA_MAX_OUTPUT_TOKENS", "").strip() or "2048")
    except ValueError:
        return 2048


MANIFEST_FORMAT_V1 = "dama_vertex_manifest_v1"
SHARD_FORMAT_V1 = "dama_vertex_shard_v1"


def _gcs_parent_prefix(gs_uri: str) -> str:
    """gs://bucket/a/b/manifest.json -> gs://bucket/a/b/"""
    s = (gs_uri or "").strip()
    if not s.startswith("gs://"):
        return ""
    rest = s[5:]
    if "/" not in rest:
        return s.rstrip("/") + "/"
    _bucket, _, path = rest.partition("/")
    if "/" not in path:
        return f"gs://{_bucket}/"
    parent = path.rsplit("/", 1)[0]
    return f"gs://{_bucket}/{parent}/"


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


def _load_manifest_json(raw: str) -> Dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object.")
    fmt = (data.get("format") or "").strip()
    if fmt != MANIFEST_FORMAT_V1:
        raise ValueError(f"Unsupported manifest format: {fmt!r} (expected {MANIFEST_FORMAT_V1})")
    shards = data.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Manifest missing non-empty 'shards' array.")
    return data


def _fetch_shard_bytes(*, base_gcs: str, base_dir: Optional[Path], shard_uri: str) -> bytes:
    u = (shard_uri or "").strip()
    if not u:
        raise ValueError("Shard entry missing uri")
    if u.startswith("gs://"):
        return _download_gcs_to_bytes(u)
    if base_gcs:
        return _download_gcs_to_bytes(base_gcs.rstrip("/") + "/" + u.lstrip("/"))
    if base_dir is not None:
        p = base_dir / u
        if not p.is_file():
            raise FileNotFoundError(f"Shard file not found: {p}")
        return p.read_bytes()
    raise ValueError("No base path for relative shard URI")


def _parse_shard_dict(raw_bytes: bytes, book_hint: str) -> List[Dict[str, Any]]:
    data = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Shard must be a JSON object.")
    fmt = (data.get("format") or "").strip()
    if fmt != SHARD_FORMAT_V1:
        raise ValueError(f"Unsupported shard format: {fmt!r} (expected {SHARD_FORMAT_V1})")
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Shard {book_hint!r} has no chunks")
    return [x for x in chunks if isinstance(x, dict)]


def _merge_manifest_to_virtual_bundle(manifest: Dict[str, Any], merged_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    em = (manifest.get("embedding_model") or "").strip()
    ver = manifest.get("version")
    return {
        "format": f"{MANIFEST_FORMAT_V1}+merged",
        "embedding_model": em,
        "chunks": merged_chunks,
        "manifest_version": ver,
        "manifest_shard_count": len(manifest.get("shards") or []),
    }


def _load_merged_bundle_from_manifest_path(manifest_path: Path) -> Dict[str, Any]:
    raw_m = manifest_path.read_text(encoding="utf-8")
    manifest = _load_manifest_json(raw_m)
    em_master = (manifest.get("embedding_model") or "").strip()
    base_dir = manifest_path.parent
    all_chunks: List[Dict[str, Any]] = []
    for ent in manifest["shards"]:
        if not isinstance(ent, dict):
            continue
        book = str(ent.get("book") or "unknown").strip()
        suri = str(ent.get("uri") or "").strip()
        shard_bytes = _fetch_shard_bytes(base_gcs="", base_dir=base_dir, shard_uri=suri)
        rows = _parse_shard_dict(shard_bytes, book)
        shard_em = ""
        # embedding_model on shard optional; enforce match if both set
        try:
            sd = json.loads(shard_bytes.decode("utf-8"))
            if isinstance(sd, dict):
                shard_em = (sd.get("embedding_model") or "").strip()
        except Exception:
            pass
        if em_master and shard_em and em_master != shard_em:
            raise ValueError(
                f"Embedding model mismatch: manifest={em_master!r} shard {book!r}={shard_em!r}"
            )
        all_chunks.extend(rows)
    if not all_chunks:
        raise ValueError("Manifest produced zero chunks after loading shards")
    return _merge_manifest_to_virtual_bundle(manifest, all_chunks)


def _load_merged_bundle_from_manifest_gcs(manifest_gcs: str, persist_dir: Path) -> Dict[str, Any]:
    persist_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = persist_dir / "_vertex_manifest_cache.json"
    raw_m = _download_gcs_to_bytes(manifest_gcs)
    cache_manifest.write_bytes(raw_m)
    manifest = _load_manifest_json(raw_m.decode("utf-8"))
    em_master = (manifest.get("embedding_model") or "").strip()
    base_gcs = _gcs_parent_prefix(manifest_gcs)
    if not base_gcs:
        raise ValueError(f"Could not derive GCS prefix from {manifest_gcs!r}")
    all_chunks: List[Dict[str, Any]] = []
    for ent in manifest["shards"]:
        if not isinstance(ent, dict):
            continue
        book = str(ent.get("book") or "unknown").strip()
        suri = str(ent.get("uri") or "").strip()
        shard_bytes = _fetch_shard_bytes(base_gcs=base_gcs, base_dir=None, shard_uri=suri)
        rows = _parse_shard_dict(shard_bytes, book)
        try:
            sd = json.loads(shard_bytes.decode("utf-8"))
            shard_em = (sd.get("embedding_model") or "").strip() if isinstance(sd, dict) else ""
        except Exception:
            shard_em = ""
        if em_master and shard_em and em_master != shard_em:
            raise ValueError(
                f"Embedding model mismatch: manifest={em_master!r} shard {book!r}={shard_em!r}"
            )
        all_chunks.extend(rows)
    if not all_chunks:
        raise ValueError("Manifest produced zero chunks after loading shards from GCS")
    return _merge_manifest_to_virtual_bundle(manifest, all_chunks)


def ensure_bundle_loaded(persist_dir: Path) -> Dict[str, Any]:
    """
    Load vertex corpus once (thread-safe). Resolution order:
    1) AN1_VERTEX_MANIFEST_PATH (local manifest.json)
    2) AN1_VERTEX_MANIFEST_GCS_URI (manifest on GCS, then shard files); if this fails and
       AN1_VERTEX_BUNDLE_GCS_URI is set, falls back to legacy monolithic bundle download.
    3) AN1_VERTEX_BUNDLE_PATH (legacy monolithic bundle)
    4) AN1_VERTEX_BUNDLE_GCS_URI
    5) persist_dir/vertex_corpus/manifest.json (local rebuild convention)
    6) default local persist_dir/an1_vertex_bundle.json
    """
    global _bundle
    with _bundle_lock:
        if _bundle is not None:
            return _bundle

        manifest_path = os.environ.get("AN1_VERTEX_MANIFEST_PATH", "").strip()
        manifest_gcs = os.environ.get("AN1_VERTEX_MANIFEST_GCS_URI", "").strip()
        env_path = os.environ.get("AN1_VERTEX_BUNDLE_PATH", "").strip()
        gcs_uri = os.environ.get("AN1_VERTEX_BUNDLE_GCS_URI", "").strip()
        default_local = _default_bundle_path(persist_dir)

        if manifest_path:
            mp = Path(manifest_path)
            if not mp.is_file():
                raise FileNotFoundError(f"AN1_VERTEX_MANIFEST_PATH not found: {mp}")
            _bundle = _load_merged_bundle_from_manifest_path(mp)
            return _bundle

        if manifest_gcs:
            try:
                _bundle = _load_merged_bundle_from_manifest_gcs(manifest_gcs, persist_dir)
                return _bundle
            except Exception as manifest_exc:
                if gcs_uri:
                    persist_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = persist_dir / "_vertex_bundle_cache.json"
                    raw = _download_gcs_to_bytes(gcs_uri)
                    cache_file.write_bytes(raw)
                    _bundle = _load_bundle_from_path(cache_file)
                    return _bundle
                raise manifest_exc

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

        auto_manifest = persist_dir / "vertex_corpus" / "manifest.json"
        if auto_manifest.is_file():
            _bundle = _load_merged_bundle_from_manifest_path(auto_manifest)
            return _bundle

        if default_local.is_file():
            _bundle = _load_bundle_from_path(default_local)
            return _bundle

        raise FileNotFoundError(
            "Vertex mode: no corpus. Set AN1_VERTEX_MANIFEST_GCS_URI / AN1_VERTEX_MANIFEST_PATH, "
            "or AN1_VERTEX_BUNDLE_GCS_URI / AN1_VERTEX_BUNDLE_PATH, "
            f"or place a bundle at {default_local}. Build: python an1_build_vertex_bundle.py --shards DIR"
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
    out: Dict[str, Any] = {
        "embedding_model": bundle.get("embedding_model") or "",
        "format": bundle.get("format") or "",
    }
    msc = bundle.get("manifest_shard_count")
    if msc is not None:
        out["manifest_shard_count"] = int(msc)
    mv = bundle.get("manifest_version")
    if mv is not None:
        out["manifest_version"] = mv
    return out
