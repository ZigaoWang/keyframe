"""keyframe: stream-friendly video keyframe extraction + LLM understanding.

Public API
----------
Most callers only need the high-level entry points:

>>> from keyframe import PipelineConfig, run_pipeline
>>> result = run_pipeline("video.mp4", PipelineConfig())

For UIs and notebooks that want incremental updates, iterate the pipeline
instead:

>>> from keyframe import iter_pipeline, PipelineConfig
>>> for event in iter_pipeline("video.mp4", PipelineConfig()):
...     print(event.stage, event.message)
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import (
    CaptionerConfig,
    EmbedderConfig,
    PipelineConfig,
    SamplerConfig,
    SegmenterConfig,
    SelectorConfig,
)
from .embedders import build_embedder, list_embedders
from .pipeline import PipelineResult, ProgressEvent, iter_pipeline, run_pipeline
from .segmenter import Segment, StreamingSegmenter
from .selector import Keyframe, select_all_keyframes

__all__ = [
    "__version__",
    "PipelineConfig",
    "SamplerConfig",
    "SegmenterConfig",
    "SelectorConfig",
    "CaptionerConfig",
    "EmbedderConfig",
    "PipelineResult",
    "ProgressEvent",
    "Keyframe",
    "Segment",
    "StreamingSegmenter",
    "iter_pipeline",
    "run_pipeline",
    "build_embedder",
    "list_embedders",
    "select_all_keyframes",
]
