"""Insert canonical sutta reference markers for known AN id patterns in transcripts."""

from __future__ import annotations

import re
from typing import Any, Dict

from transcript_pipeline.context import PipelineContext
from transcript_pipeline.registry import register
from transcript_pipeline.transforms.base import Transform

_DEFAULT_MARKERS: Dict[str, str] = {
    r"4\.4\.36": "AN 4.36 (Doṇa Sutta)",
    r"4\.4\.37": "AN 4.37 (Aparihāna Sutta)",
    r"4\.5\.41": "AN 4.41 (Samādhi-bhāvanā Sutta)",
    r"4\.5\.49": "AN 4.49 (Vipallāsa Sutta)",
}


@register("sutta_tags")
class SuttaTagTransform(Transform):
    name = "sutta_tags"

    def __init__(self, options: Dict[str, Any] | None = None) -> None:
        super().__init__(options)
        custom = self.options.get("markers")
        if isinstance(custom, dict) and custom:
            self._markers = {str(k): str(v) for k, v in custom.items()}
        else:
            self._markers = dict(_DEFAULT_MARKERS)

    def apply(self, text: str, ctx: PipelineContext) -> str:
        for pattern, canonical in self._markers.items():
            repl = f"\n\n### REFERENCE: {canonical}\n\n"
            text = re.sub(pattern, repl, text)
        return text
