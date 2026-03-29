from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transcript_pipeline.context import PipelineContext
from transcript_pipeline.registry import register
from transcript_pipeline.transforms.base import Transform


def _compile_replacers(pairs: List[Tuple[str, str]]) -> List[Tuple[re.Pattern[str], str]]:
    """Longest source keys first; word-boundary style (no alnum before/after)."""
    out: List[Tuple[re.Pattern[str], str]] = []
    for source, target in pairs:
        source = source.strip()
        if not source:
            continue
        if " " in source:
            parts = source.split()
            inner = r"\s+".join(re.escape(p) for p in parts)
            pat = r"(?<![\w/])" + inner + r"(?![\w/])"
        else:
            pat = r"(?<![\w/])" + re.escape(source) + r"(?![\w/])"
        out.append((re.compile(pat, re.IGNORECASE), target))
    return out


@register("glossary")
class GlossaryTransform(Transform):
    """Replace ASR/noisy spellings using a JSON object of source -> target (longest match first)."""

    name = "glossary"

    def __init__(self, options: Dict[str, Any] | None = None) -> None:
        super().__init__(options)
        base = Path(self.options.get("base_dir", ".")).resolve()
        map_path = Path(self.options.get("map_path", "pali_normalize_map.json"))
        if not map_path.is_absolute():
            map_path = (base / map_path).resolve()
        raw = json.loads(map_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"Glossary JSON must be an object, got {type(raw)}")
        pairs: List[Tuple[str, str]] = []
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            pairs.append((k, v))
        pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
        self._map_path = map_path
        self._replacers = _compile_replacers(pairs)

    def apply(self, text: str, ctx: PipelineContext) -> str:
        for rx, target in self._replacers:
            text = rx.sub(target, text)
        return text
