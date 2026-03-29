from __future__ import annotations

from typing import Any, Dict, Type

# Populated by each transform module at import time.
TRANSFORM_CLASSES: Dict[str, Type[Any]] = {}


def register(name: str):
    """Decorator: register a Transform subclass under `name` (used in pipeline JSON)."""

    def deco(cls: Type[Any]) -> Type[Any]:
        if not name:
            raise ValueError("Transform name must be non-empty")
        if name in TRANSFORM_CLASSES and TRANSFORM_CLASSES[name] is not cls:
            raise ValueError(f"Duplicate transform name: {name!r}")
        setattr(cls, "name", name)
        TRANSFORM_CLASSES[name] = cls
        return cls

    return deco


def get_transform(name: str, options: Dict[str, Any] | None = None) -> Any:
    try:
        cls = TRANSFORM_CLASSES[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown transform {name!r}. Known: {sorted(TRANSFORM_CLASSES)}"
        ) from e
    return cls(options)
