from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict


class TransformSpec(TypedDict, total=False):
    name: str
    options: Dict[str, Any]


def load_pipeline_config(path: Path) -> List[TransformSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Pipeline config root must be a JSON object")
    steps = data.get("transforms")
    if not isinstance(steps, list):
        raise TypeError('Pipeline config must contain a "transforms" array')
    out: List[TransformSpec] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise TypeError(f"transforms[{i}] must be an object")
        name = step.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"transforms[{i}] needs string 'name'")
        opts = step.get("options") or {}
        if not isinstance(opts, dict):
            raise TypeError(f"transforms[{i}].options must be an object")
        out.append({"name": name, "options": dict(opts)})
    return out
