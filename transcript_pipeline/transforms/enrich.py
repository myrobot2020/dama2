"""Session-specific phonetic fixes, jhāna phrasing, and bold markers for retrieval."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from transcript_pipeline.context import PipelineContext
from transcript_pipeline.registry import register
from transcript_pipeline.transforms.base import Transform

# Word-boundary on short tokens avoids e.g. "eel" matching inside "wheel".
_DEFAULT_PHONETIC: List[Tuple[str, str]] = [
    (r"\bdonor\b", "Doṇa"),
    (r"\bsudah\b", "sutta"),
    (r"\beel\b", "ill (dukkha)"),
    (r"\bsubah\b", "subha"),
    (r"not cell", "not-self (anattā)"),
    (r"fija charana", "vijjā-caraṇa"),
]

_DEFAULT_BOLD_KEYS = [
    "āsavas",
    "Nibbāna",
    "samādhi-bhāvanā",
    "ñāṇadassana",
    "satipaṭṭhāna",
    "vipallāsa",
]


@register("phonetic_cleanup")
class PhoneticCleanupTransform(Transform):
    name = "phonetic_cleanup"

    def __init__(self, options: Dict[str, Any] | None = None) -> None:
        super().__init__(options)
        raw = self.options.get("replacements")
        if isinstance(raw, list) and raw:
            self._pairs: List[Tuple[str, str]] = []
            for row in raw:
                if isinstance(row, dict) and "pattern" in row and "replacement" in row:
                    self._pairs.append((str(row["pattern"]), str(row["replacement"])))
        else:
            self._pairs = list(_DEFAULT_PHONETIC)

    def apply(self, text: str, ctx: PipelineContext) -> str:
        for pattern, fix in self._pairs:
            text = re.sub(pattern, fix, text, flags=re.IGNORECASE)
        return text


@register("format_jhanas")
class FormatJhanasTransform(Transform):
    name = "format_jhanas"

    def apply(self, text: str, ctx: PipelineContext) -> str:
        return re.sub(r"fort janna", "fourth jhāna", text, flags=re.IGNORECASE)


@register("bold_canonical_keys")
class BoldCanonicalKeysTransform(Transform):
    name = "bold_canonical_keys"

    def __init__(self, options: Dict[str, Any] | None = None) -> None:
        super().__init__(options)
        keys = self.options.get("keys")
        if isinstance(keys, list) and keys:
            self._keys = [str(k) for k in keys]
        else:
            self._keys = list(_DEFAULT_BOLD_KEYS)

    def apply(self, text: str, ctx: PipelineContext) -> str:
        for key in self._keys:
            rx = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
            text = rx.sub(f"**{key}**", text)
        return text
