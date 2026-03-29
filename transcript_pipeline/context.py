from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class PipelineContext:
    """Per-file metadata passed through each transform (extend fields as needed)."""

    base_dir: Path
    source_path: Path | None = None
    extra: Dict[str, Any] = field(default_factory=dict)
