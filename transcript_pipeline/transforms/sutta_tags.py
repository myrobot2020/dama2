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

_PAT_ONES_ID3 = re.compile(r"(?<!\d)(1\.\d+\.\d+)(?!\d)")
# 2-part ids like `1.5` that are NOT the prefix of a 3-part id like `1.5.9`
_PAT_ONES_ID2 = re.compile(r"(?<!\d)(1\.\d+)(?!\.\d)(?!\d)")


def _is_book_of_ones_file(ctx) -> bool:
    # File names look like: "003_Anguttara Nikaya Book 1B 15 - 11014 by ...txt"
    src = getattr(ctx, "source_path", None)
    name = getattr(src, "name", "") or ""
    return any(x in name for x in ("Book 1A", "Book 1B", "Book 1C", "Book 1D"))


def _insert_reference_headers_for_matches(text: str, pat: re.Pattern[str]) -> str:
    """
    Insert:
      ### REFERENCE: AN <sutta_id>
    directly before each id match.

    Uses a small lookbehind window to avoid double-inserting when `### REFERENCE:` already
    exists nearby.
    """
    out_parts: list[str] = []
    last_i = 0

    for m in pat.finditer(text):
        start = m.start()
        # If the match already appears near an existing REFERENCE header, skip insertion.
        # (The pipeline usually runs on raw transcripts, so this is mostly for idempotency.)
        window = text[max(0, start - 90) : start]
        if "### REFERENCE:" in window:
            continue

        token = m.group(1)
        out_parts.append(text[last_i:start])
        out_parts.append(f"\n\n### REFERENCE: {f'AN {token}'}\n\n{token}")
        last_i = m.end()

    out_parts.append(text[last_i:])
    return "".join(out_parts)


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
        # Insert canonical headers for Book-of-Ones (AN 1.x.y) sutta boundary tokens.
        # This enables exact `an_sutta_ref` filtering at query time.
        if _is_book_of_ones_file(ctx):
            # 3-part ids first, then 2-part ids (that are not prefixes of 3-part ids).
            text = _insert_reference_headers_for_matches(text, _PAT_ONES_ID3)
            text = _insert_reference_headers_for_matches(text, _PAT_ONES_ID2)

        for pattern, canonical in self._markers.items():
            repl = f"\n\n### REFERENCE: {canonical}\n\n"
            text = re.sub(pattern, repl, text)
        return text
