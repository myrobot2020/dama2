from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from transcript_pipeline.context import PipelineContext


class Transform(ABC):
    """One ordered step in the pipeline. Subclasses implement apply().

    To add a new step:
      1. Subclass Transform in a new module under transforms/.
      2. Decorate with @register("unique_name").
      3. Import that module from transforms/__init__.py.
      4. Append {"name": "unique_name", "options": {...}} to pipeline JSON.
    """

    name: str = "base"

    def __init__(self, options: Dict[str, Any] | None = None) -> None:
        self.options = dict(options or {})

    @abstractmethod
    def apply(self, text: str, ctx: PipelineContext) -> str:
        ...
