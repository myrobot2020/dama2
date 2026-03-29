"""Extensible transcript normalization pipeline (rule-based transforms, config-driven)."""

from transcript_pipeline.context import PipelineContext
from transcript_pipeline.runner import run_pipeline_on_file, run_pipeline_on_text

__all__ = [
    "PipelineContext",
    "run_pipeline_on_text",
    "run_pipeline_on_file",
]
