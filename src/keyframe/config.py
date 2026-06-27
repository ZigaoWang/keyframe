"""Configuration objects for the keyframe pipeline.

All tunables live here so the rest of the code stays parameter-free.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class SamplerConfig:
    """How densely we sample the incoming stream."""
    interval_sec: float = 1.0
    """Seconds between analyzed frames. Smaller = sharper boundary detection, more compute."""


@dataclass
class SegmenterConfig:
    """Online shot detection."""
    sim_threshold: float = 0.96
    """cos_sim(frame_embedding, shot_running_mean). Below this -> new segment.
    Tuned for downscaled inputs from the ffmpeg fast path; raise to 0.97 for
    very static videos, drop to 0.92 for cut-heavy montages."""

    use_ema: bool = True
    """Exponential moving average instead of arithmetic mean. Prevents drift on long shots."""

    ema_alpha: float = 0.25
    """EMA weight for the new frame. Higher = shot mean adapts faster."""

    min_shot_sec: float = 2.0
    """Reject shot cuts that would produce segments shorter than this. Anti-flicker."""


@dataclass
class SelectorConfig:
    """Pick the best frame(s) inside each segment."""
    max_frames_per_segment: int = 3
    """Upper bound on keyframes per segment. Long segments get more."""

    seconds_per_keyframe: float = 8.0
    """Roughly one keyframe per N seconds of segment length, capped by max_frames_per_segment."""

    min_temporal_spacing_sec: float = 1.0
    """Picked frames within a segment must be at least this far apart in time."""

    sharpness_weight: float = 0.45
    """Weight on Laplacian-variance sharpness (anti-blur) in composite score."""

    representativeness_weight: float = 0.55
    """Weight on cosine similarity to segment centroid (anti-outlier) in composite score."""

    diversity_sim_threshold: float = 0.97
    """After per-segment selection, drop any keyframe whose embedding has
    cosine similarity above this with an already-kept keyframe. Set close to
    1.0 to only drop true duplicates; lower to thin the result more aggressively.
    Default 0.97 keeps near-identical-but-distinct frames so the user gets
    coverage rather than a minimum set."""

    min_keyframes: int = 6
    """Floor on total keyframes returned. If the diversity filter drops below
    this, keep the highest-scoring duplicates back to hit the floor."""


@dataclass
class CaptionerConfig:
    """LLM caption stage."""
    enabled: bool = True
    model: str = "gpt-5.4"
    """Default model id. Override with --model on the CLI."""

    base_url_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    detail: str = "low"
    """``'low'`` or ``'high'``. Low is cheaper and faster, fine for downscaled frames."""

    max_keyframes: int = 24
    """Hard cap on frames sent in a single LLM call."""

    thumb_max_width: int = 512
    """Resize keyframes to at most this width before base64 encoding."""

    jpeg_quality: int = 80

    fallback_models: tuple[str, ...] = (
        "google/gemini-2.5-flash",
        "qwen/qwen2.5-vl-72b-instruct",
        "anthropic/claude-3.5-sonnet",
    )
    """Tried in order when the primary model returns a regional-block / auth
    error. OpenRouter rejects some ``openai/*`` ids in certain regions; the
    fallbacks here are regularly available worldwide."""


@dataclass
class EmbedderConfig:
    """Which embedding backend to use."""
    name: str = "yolov8n"
    """Registered name. See embedders.py: yolov8n / yolov8s / yolov8m / phash / hsv."""

    device: str = "auto"
    """'auto' picks mps -> cuda -> cpu. Or pass 'cpu' explicitly."""


@dataclass
class PipelineConfig:
    """Top-level config that combines all stages."""
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    captioner: CaptionerConfig = field(default_factory=CaptionerConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)

    output_root: Path = field(default_factory=lambda: Path("outputs"))
    """Base directory for per-run output folders."""

    realtime_pacing: bool = False
    """When source is a file, sleep so the loop ticks at the source fps (true streaming)."""

    save_every_analyzed_frame: bool = False
    """Debug: also keep every sampled frame on disk. Off by default to save space."""

    # ---- speed knobs --------------------------------------------------- #
    batch_size: int = 8
    """Frames per embed call when running on a file. Stream sources stay at 1."""

    cache_thumb_width: int = 640
    """Resize cached analyzed frames to this max width before writing to disk."""

    embed_max_width: int = 960
    """Resize frames to this max width before sending to the embedder."""

    viz_refresh_every: int = 8
    """Redraw timeline + similarity curve every N frames. Lower = smoother live UI."""

    async_disk_writes: bool = True
    """Push cache writes onto a background thread so embed/segment is not blocked by IO."""

    use_ffmpeg: bool = True
    """Use an ffmpeg subprocess for hardware-accelerated decode+resize on file sources.
    Falls back to cv2 if ffmpeg is not installed or the source is a live stream."""

    ffmpeg_hwaccel: str = "auto"
    """ffmpeg hwaccel mode. 'auto' picks videotoolbox/cuda/vaapi when available.
    Set to 'none' to force CPU decode."""

    num_decode_workers: int = 1
    """Parallel ffmpeg workers. Default 1 (sequential). Higher values can
    decode faster but require careful time-ordering; the parallel implementation
    is currently disabled because it shuffled segment IDs."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(requested: str) -> str:
    """auto-detect best torch device."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_dotenv(path: Path | str = ".env") -> None:
    """Tiny dotenv loader. No external dep.

    Reads ``KEY=VALUE`` lines, strips surrounding single or double quotes from
    the value, and only sets keys that are not already present in the process
    environment. Lines starting with ``#`` or missing ``=`` are ignored.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)
