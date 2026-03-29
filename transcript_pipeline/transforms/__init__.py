"""Import transform modules here so @register runs before the pipeline runs."""

from transcript_pipeline.transforms import enrich  # noqa: F401
from transcript_pipeline.transforms import glossary  # noqa: F401
from transcript_pipeline.transforms import sutta_tags  # noqa: F401

__all__: list[str] = []
