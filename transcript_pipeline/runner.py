from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from transcript_pipeline.context import PipelineContext
from transcript_pipeline.load_config import TransformSpec, load_pipeline_config
from transcript_pipeline.registry import get_transform

import transcript_pipeline.transforms  # noqa: F401 — register built-in transforms

# region agent log
_DEBUG_LOG = Path(__file__).resolve().parent.parent / "debug-37c3af.log"


def _agent_dbg(message: str, data: dict, hypothesis_id: str) -> None:
    payload = {
        "sessionId": "37c3af",
        "timestamp": int(time.time() * 1000),
        "location": "transcript_pipeline/runner.py",
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "runId": "pipeline",
    }
    with _DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# endregion


def _build_chain(specs: List[TransformSpec], resource_root: Path):
    chain = []
    for spec in specs:
        opts = dict(spec.get("options") or {})
        opts.setdefault("base_dir", str(resource_root))
        chain.append(get_transform(spec["name"], opts))
    return chain


def run_pipeline_on_text(
    text: str,
    *,
    resource_root: Path,
    config_path: Path,
    ctx: PipelineContext | None = None,
) -> str:
    specs = load_pipeline_config(config_path)
    chain = _build_chain(specs, resource_root)
    # region agent log
    _agent_dbg(
        "chain_built",
        {
            "step_names": [getattr(s, "name", type(s).__name__) for s in chain],
            "in_len": len(text),
            "ref_before": text.count("### REFERENCE"),
        },
        "H1",
    )
    # endregion
    ctx = ctx or PipelineContext(base_dir=resource_root)
    for step in chain:
        text = step.apply(text, ctx)
        # region agent log
        _agent_dbg(
            "after_transform",
            {
                "step": getattr(step, "name", type(step).__name__),
                "out_len": len(text),
                "ref_count": text.count("### REFERENCE"),
            },
            "H3",
        )
        # endregion
    return text


def run_pipeline_on_file(
    source: Path,
    *,
    resource_root: Path,
    config_path: Path,
    dest: Path,
) -> None:
    text = source.read_text(encoding="utf-8", errors="ignore")
    ctx = PipelineContext(base_dir=resource_root, source_path=source.resolve())
    out = run_pipeline_on_text(
        text, resource_root=resource_root, config_path=config_path, ctx=ctx
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out, encoding="utf-8", newline="\n")
