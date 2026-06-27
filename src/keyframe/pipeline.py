"""End-to-end pipeline orchestrator.

Two entry points, same algorithm:

- ``iter_pipeline`` yields :class:`ProgressEvent` after every analyzed frame,
  every closed segment, and every later milestone. Web UIs and notebooks
  subscribe to this generator for live streaming updates.

- ``run_pipeline`` is a thin wrapper that exhausts ``iter_pipeline`` and
  returns the final :class:`PipelineResult`. The CLI uses this.

Speed design
------------
1. Files seek directly to each sample point instead of decoding throwaway
   frames in between. A 60fps video sampled at 1Hz used to decode 60 frames
   per sample; it now decodes one.
2. Frames are resized to ``cfg.embed_max_width`` before embedding and cached
   at ``cfg.cache_thumb_width`` before writing to disk.
3. Embedders are called in batches of ``cfg.batch_size`` (file sources) so
   the GPU stays busy. Stream sources stay at batch=1 to keep latency low.
4. Cache writes run on a background thread when ``cfg.async_disk_writes``
   is set, so disk IO does not block the next embed call.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .captioner import caption_keyframes
from .config import PipelineConfig, load_dotenv, resolve_device
from .embedders import build_embedder
from .ffmpeg_source import FfmpegSampledSource, ParallelFfmpegSource, ffmpeg_available
from .io_utils import (
    append_jsonl, imwrite_unicode, laplacian_sharpness,
    resize_max_width, write_csv, write_json,
)
from .logging_setup import get_logger
from .segmenter import SampledFrame, Segment, StreamingSegmenter
from .selector import Keyframe, select_all_keyframes, select_segment_keyframes
from .source import VideoSource, subsample_by_time
from .visualize import (
    draw_film_strip, draw_keyframes_grid, draw_similarity_curve, draw_timeline,
)

log = get_logger("pipeline")


@dataclass
class RunPaths:
    root: Path
    frames_dir: Path
    keyframes_dir: Path
    viz_dir: Path
    captions_dir: Path
    logs_dir: Path

    @classmethod
    def create(cls, output_root: Path, source_label: str) -> "RunPaths":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in source_label)
        root = output_root / f"{safe_label}_{ts}"
        paths = cls(
            root=root,
            frames_dir=root / "frames",
            keyframes_dir=root / "keyframes",
            viz_dir=root / "viz",
            captions_dir=root / "caption",
            logs_dir=root / "logs",
        )
        for d in (paths.root, paths.frames_dir, paths.keyframes_dir,
                  paths.viz_dir, paths.captions_dir, paths.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass
class PipelineResult:
    run_dir: Path
    embedder: str
    duration_sec: float
    frames_analyzed: int
    segments: int
    keyframes: int
    caption_text: str
    caption_latency_sec: float
    embed_latency_ms_avg: float
    embed_latency_ms_p95: float
    wall_time_sec: float


@dataclass
class ProgressEvent:
    """One streamed update to subscribers (Gradio UI, notebooks, logs)."""
    stage: str
    message: str
    run_dir: Optional[Path] = None

    analyzed_idx: int = -1
    source_idx: int = -1
    timestamp_sec: float = 0.0
    sim_to_shot_mean: float = 0.0
    sharpness: float = 0.0
    is_shot_start: bool = False
    embed_ms: float = 0.0
    current_segment_id: int = -1
    bgr_path: Optional[str] = None

    frames_analyzed: int = 0
    segments_so_far: int = 0
    keyframes_so_far: int = 0
    embed_latency_ms_avg: float = 0.0

    timeline_path: Optional[Path] = None
    similarity_curve_path: Optional[Path] = None
    keyframes_grid_path: Optional[Path] = None
    film_strip_path: Optional[Path] = None
    """Single hero visualization: thumbnails of every analyzed frame in a row,
    coloured segment band below, keyframe markers above. Web UI displays this
    instead of the three separate viz files."""
    keyframes: list[Keyframe] = field(default_factory=list)

    final_result: Optional[PipelineResult] = None
    caption_text: str = ""


def _select_partial(segments: list[Segment], cfg) -> list[Keyframe]:
    if not segments:
        return []
    out: list[Keyframe] = []
    next_id = 1
    for seg in segments:
        chosen = select_segment_keyframes(seg, cfg, next_id)
        out.extend(chosen)
        next_id += len(chosen)
    return out


class _AsyncWriter:
    """Background thread that drains a queue of (path, bgr) cache writes."""

    def __init__(self, max_queue: int = 64) -> None:
        self._q: "queue.Queue[tuple[Path, np.ndarray] | None]" = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, path: Path, bgr: np.ndarray) -> None:
        self._q.put((path, bgr))

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            path, bgr = item
            try:
                imwrite_unicode(path, bgr)
            except Exception as exc:  # pragma: no cover
                log.warning("async write failed for %s: %s", path, exc)


def _prepare_frame(bgr: np.ndarray, embed_w: int, cache_w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(embed_input, cache_input)``. Both BGR. Smaller of the two reuses memory."""
    embed_bgr = resize_max_width(bgr, embed_w) if embed_w > 0 else bgr
    if cache_w <= 0 or cache_w >= bgr.shape[1]:
        cache_bgr = bgr
    elif cache_w == embed_w:
        cache_bgr = embed_bgr
    else:
        cache_bgr = resize_max_width(bgr, cache_w)
    return embed_bgr, cache_bgr


def iter_pipeline(
    source_spec: str | int | Path,
    cfg: PipelineConfig,
    refresh_viz_every: int | None = None,
) -> Iterator[ProgressEvent]:
    load_dotenv()
    cfg.embedder.device = resolve_device(cfg.embedder.device)
    if refresh_viz_every is None:
        refresh_viz_every = cfg.viz_refresh_every

    # Pick source backend. ffmpeg subprocess is much faster on files; cv2 is
    # required for webcams and arbitrary URLs that ffmpeg cannot iterate over.
    spec_str = str(source_spec)
    is_file_like = (
        isinstance(source_spec, (str, Path)) and not spec_str.isdigit()
        and not spec_str.startswith(("rtsp://", "rtmp://", "http://", "https://"))
        and Path(spec_str).exists()
    )
    use_ffmpeg = cfg.use_ffmpeg and is_file_like and ffmpeg_available() and not cfg.realtime_pacing

    src: object
    if use_ffmpeg:
        if cfg.num_decode_workers > 1:
            src = ParallelFfmpegSource(
                source_spec, interval_sec=cfg.sampler.interval_sec,
                max_width=cfg.embed_max_width,
                hwaccel=cfg.ffmpeg_hwaccel,
                num_workers=cfg.num_decode_workers,
            )
        else:
            src = FfmpegSampledSource(
                source_spec, interval_sec=cfg.sampler.interval_sec,
                max_width=cfg.embed_max_width,
                hwaccel=cfg.ffmpeg_hwaccel,
            )
        src_label = Path(spec_str).stem
        is_file = True
        duration_sec_hint = src.duration_sec  # type: ignore[union-attr]
    else:
        src = VideoSource(source_spec, realtime=cfg.realtime_pacing)
        src_label = (Path(spec_str).stem if isinstance(source_spec, (str, Path)) else
                     f"cam{source_spec}")
        is_file = src.is_file
        duration_sec_hint = src.duration_sec or 0.0

    paths = RunPaths.create(cfg.output_root, src_label)
    log.info("[run] output dir: %s", paths.root)
    write_json(cfg.to_dict(), paths.logs_dir / "config.json")

    yield ProgressEvent(
        stage="init",
        message=(f"Loading embedder '{cfg.embedder.name}' on {cfg.embedder.device}; "
                 f"decoder={'ffmpeg(' + cfg.ffmpeg_hwaccel + ')' if use_ffmpeg else 'cv2'}..."),
        run_dir=paths.root,
    )

    embedder = build_embedder(cfg.embedder.name, device=cfg.embedder.device)
    log.info("[embed] %s on %s, dim=%d",
             embedder.name, getattr(embedder, "device", "?"), embedder.dim)

    effective_batch = max(1, cfg.batch_size if is_file else 1)
    yield ProgressEvent(
        stage="init",
        message=(f"Embedder ready: {embedder.name} ({embedder.dim}-d) on "
                 f"{cfg.embedder.device}.  fast_path={'on' if is_file else 'off (stream)'}  "
                 f"batch={effective_batch}  cache@{cfg.cache_thumb_width}px"),
        run_dir=paths.root,
    )

    segmenter = StreamingSegmenter(cfg.segmenter)
    samples: list[SampledFrame] = []
    embed_latencies_ms: list[float] = []
    events_path = paths.logs_dir / "stream_events.jsonl"
    wall_start = time.perf_counter()
    analyzed_count = 0
    current_kf: list[Keyframe] = []

    writer: Optional[_AsyncWriter] = _AsyncWriter() if cfg.async_disk_writes else None

    total_frames_hint: int | None = None
    if is_file and duration_sec_hint and cfg.sampler.interval_sec > 0:
        total_frames_hint = max(1, int(duration_sec_hint / cfg.sampler.interval_sec))
    progress = tqdm(
        total=total_frames_hint, unit="sample", desc="streaming",
        dynamic_ncols=True, disable=not is_file,
    )

    pending: list[tuple] = []  # (frame, embed_bgr, cache_bgr, sharpness, bgr_path)

    def flush_batch() -> Iterator[ProgressEvent]:
        nonlocal analyzed_count, current_kf
        if not pending:
            return
        embed_inputs = [p[1] for p in pending]
        t_embed = time.perf_counter()
        vectors = embedder.embed_batch(embed_inputs)
        batch_ms = (time.perf_counter() - t_embed) * 1000.0
        per_frame_ms = batch_ms / max(len(pending), 1)

        for i, (frame, _ebgr, cache_bgr, sharpness, bgr_path) in enumerate(pending):
            embed_latencies_ms.append(per_frame_ms)
            embedding = vectors[i]

            if writer is not None:
                writer.submit(bgr_path, cache_bgr)
            else:
                imwrite_unicode(bgr_path, cache_bgr)

            result = segmenter.ingest(
                source_index=frame.index,
                timestamp_sec=frame.timestamp_sec,
                embedding=embedding,
                bgr_path=str(bgr_path),
                sharpness=sharpness,
            )
            sampled = result.sampled
            closed = result.closed
            samples.append(sampled)
            append_jsonl({
                "event": "frame",
                "analyzed_idx": sampled.index,
                "source_idx": sampled.source_index,
                "timestamp_sec": round(sampled.timestamp_sec, 3),
                "sim": round(sampled.sim_to_shot_mean, 4),
                "sharpness": round(sampled.sharpness, 1),
                "is_shot_start": sampled.is_shot_start,
                "embed_ms": round(per_frame_ms, 2),
            }, events_path)

            timeline_p: Optional[Path] = None
            sim_p: Optional[Path] = None
            grid_p: Optional[Path] = None
            film_p: Optional[Path] = None
            need_redraw = (analyzed_count % refresh_viz_every == 0
                           or sampled.is_shot_start)
            if samples and need_redraw:
                timeline_p = paths.viz_dir / "timeline.jpg"
                sim_p = paths.viz_dir / "similarity_curve.jpg"
                film_p = paths.viz_dir / "film_strip.jpg"
                kf_ts = [k.timestamp_sec for k in current_kf]
                draw_timeline(samples, kf_ts, timeline_p)
                draw_similarity_curve(samples, cfg.segmenter.sim_threshold, kf_ts, sim_p)
                draw_film_strip(samples, current_kf, film_p)

            if closed is not None:
                append_jsonl({
                    "event": "segment_closed",
                    "segment_id": closed.segment_id,
                    "start_sec": round(closed.start_sec, 3),
                    "end_sec": round(closed.end_sec, 3),
                    "frames": closed.num_frames,
                }, events_path)
                current_kf = _select_partial(segmenter.closed_segments, cfg.selector)
                grid_p = paths.viz_dir / "keyframes_grid.jpg"
                draw_keyframes_grid(current_kf, grid_p)
                if film_p is None:
                    film_p = paths.viz_dir / "film_strip.jpg"
                    draw_film_strip(samples, current_kf, film_p)
                yield ProgressEvent(
                    stage="segment_closed",
                    message=(f"Segment #{closed.segment_id} closed at t={closed.end_sec:.1f}s "
                             f"({closed.duration_sec:.1f}s, {closed.num_frames} frames)"),
                    run_dir=paths.root,
                    current_segment_id=closed.segment_id,
                    segments_so_far=len(segmenter.closed_segments),
                    keyframes_so_far=len(current_kf),
                    timeline_path=timeline_p,
                    similarity_curve_path=sim_p,
                    keyframes_grid_path=grid_p,
                    film_strip_path=film_p,
                    keyframes=list(current_kf),
                )

            yield ProgressEvent(
                stage="frame",
                message=(f"Frame {analyzed_count + 1}: t={frame.timestamp_sec:.1f}s  "
                         f"sim={sampled.sim_to_shot_mean:.3f}  "
                         f"sharp={sharpness:.0f}  embed={per_frame_ms:.1f}ms"
                         + ("  [SHOT START]" if sampled.is_shot_start else "")),
                run_dir=paths.root,
                analyzed_idx=analyzed_count,
                source_idx=frame.index,
                timestamp_sec=frame.timestamp_sec,
                sim_to_shot_mean=sampled.sim_to_shot_mean,
                sharpness=sharpness,
                is_shot_start=sampled.is_shot_start,
                embed_ms=per_frame_ms,
                current_segment_id=segmenter.current_segment_id,
                bgr_path=str(bgr_path),
                frames_analyzed=analyzed_count + 1,
                segments_so_far=segmenter.total_segments_so_far,
                keyframes_so_far=len(current_kf),
                embed_latency_ms_avg=float(np.mean(embed_latencies_ms)),
                timeline_path=timeline_p,
                similarity_curve_path=sim_p,
                keyframes_grid_path=grid_p,
                film_strip_path=film_p,
                keyframes=list(current_kf),
            )
            analyzed_count += 1
            progress.update(1)
        pending.clear()

    try:
        if use_ffmpeg:
            # ffmpeg returns frames already pre-scaled to embed_max_width
            frame_iter = iter(src)
        else:
            frame_iter = src.iter_sampled(cfg.sampler.interval_sec)  # type: ignore[union-attr]
        for frame in frame_iter:
            if use_ffmpeg:
                # already pre-scaled by ffmpeg; reuse for both embed and cache
                embed_bgr = frame.bgr
                cache_bgr = (resize_max_width(frame.bgr, cfg.cache_thumb_width)
                             if cfg.cache_thumb_width and cfg.cache_thumb_width < frame.bgr.shape[1]
                             else frame.bgr)
            else:
                embed_bgr, cache_bgr = _prepare_frame(
                    frame.bgr, cfg.embed_max_width, cfg.cache_thumb_width,
                )
            sharpness = laplacian_sharpness(embed_bgr)
            bgr_path = paths.frames_dir / f"f{analyzed_count + len(pending):05d}_t{frame.timestamp_sec:07.2f}s.jpg"
            pending.append((frame, embed_bgr, cache_bgr, sharpness, bgr_path))
            if len(pending) >= effective_batch:
                yield from flush_batch()
        yield from flush_batch()
    finally:
        progress.close()
        last_ts = samples[-1].timestamp_sec if samples else 0.0
        segmenter.finalise(end_sec=last_ts)
        release = getattr(src, "release", None)
        if callable(release):
            try:
                release()
            except Exception as exc:
                log.warning("source release failed: %s", exc)
        if writer is not None:
            writer.close()

    yield ProgressEvent(
        stage="select", message="Selecting keyframes from all segments...",
        run_dir=paths.root,
    )
    segments = segmenter.closed_segments
    keyframes = select_all_keyframes(segments, cfg.selector)

    saved_kf_paths: list[Path] = []
    for kf in keyframes:
        src_path = Path(kf.bgr_path)
        dst = paths.keyframes_dir / (
            f"kf_{kf.keyframe_id:03d}_seg{kf.segment_id:02d}"
            f"_t{kf.timestamp_sec:07.2f}s.jpg"
        )
        if not dst.exists() and src_path.exists():
            dst.write_bytes(src_path.read_bytes())
        saved_kf_paths.append(dst)

    write_csv(
        [{
            "analyzed_idx": s.index,
            "source_idx": s.source_index,
            "timestamp_sec": round(s.timestamp_sec, 3),
            "sim_to_shot_mean": round(s.sim_to_shot_mean, 4),
            "sharpness": round(s.sharpness, 1),
            "is_shot_start": int(s.is_shot_start),
        } for s in samples],
        paths.logs_dir / "analyzed_frames.csv",
    )
    write_csv(
        [{
            "segment_id": seg.segment_id,
            "start_sec": round(seg.start_sec, 3),
            "end_sec": round(seg.end_sec, 3),
            "duration_sec": round(seg.duration_sec, 3),
            "num_frames": seg.num_frames,
        } for seg in segments],
        paths.logs_dir / "segments.csv",
    )
    write_csv(
        [{
            "keyframe_id": k.keyframe_id,
            "segment_id": k.segment_id,
            "timestamp_sec": round(k.timestamp_sec, 3),
            "sharpness": round(k.sharpness, 1),
            "representativeness": round(k.representativeness, 4),
            "composite_score": round(k.composite_score, 4),
            "rank_in_segment": k.rank_in_segment,
            "saved_path": str(saved_kf_paths[i]),
        } for i, k in enumerate(keyframes)],
        paths.keyframes_dir / "keyframes.csv",
    )

    yield ProgressEvent(
        stage="visualize", message="Rendering final visualizations...",
        run_dir=paths.root,
    )
    timeline_p = paths.viz_dir / "timeline.jpg"
    sim_p = paths.viz_dir / "similarity_curve.jpg"
    grid_p = paths.viz_dir / "keyframes_grid.jpg"
    film_p = paths.viz_dir / "film_strip.jpg"
    draw_timeline(samples, [k.timestamp_sec for k in keyframes], timeline_p)
    draw_similarity_curve(samples, cfg.segmenter.sim_threshold,
                          [k.timestamp_sec for k in keyframes], sim_p)
    draw_keyframes_grid(keyframes, grid_p)
    draw_film_strip(samples, keyframes, film_p)

    duration_sec = (samples[-1].timestamp_sec - samples[0].timestamp_sec) if samples else 0.0

    caption_text = ""
    caption_latency = 0.0
    yield ProgressEvent(
        stage="caption",
        message=(f"Captioning {len(keyframes)} keyframes with {cfg.captioner.model}..."
                 if cfg.captioner.enabled else "Skipping LLM caption (--no-caption)."),
        run_dir=paths.root,
        timeline_path=timeline_p,
        similarity_curve_path=sim_p,
        keyframes_grid_path=grid_p,
        film_strip_path=film_p,
        keyframes=keyframes,
        segments_so_far=len(segments),
        keyframes_so_far=len(keyframes),
    )
    try:
        result = caption_keyframes(keyframes, duration_sec, cfg.captioner)
        caption_text = result.text
        caption_latency = result.latency_sec
        (paths.captions_dir / "caption.md").write_text(
            "# Video Caption\n\n"
            f"- Source: `{source_spec}`\n"
            f"- Model: `{result.model}`\n"
            f"- Keyframes sent: {result.frames_sent}\n"
            f"- LLM latency: {result.latency_sec:.2f}s\n"
            f"- Embedder: `{cfg.embedder.name}` on `{cfg.embedder.device}`\n\n"
            "---\n\n"
            f"{caption_text}\n",
            encoding="utf-8",
        )
        if result.raw is not None:
            write_json(result.raw, paths.captions_dir / "raw_response.json")
    except Exception as exc:
        log.exception("captioner failed: %s", exc)
        caption_text = f"(captioner failed: {exc})"
        (paths.captions_dir / "caption.md").write_text(caption_text, encoding="utf-8")

    embed_p50 = float(np.percentile(embed_latencies_ms, 50)) if embed_latencies_ms else 0.0
    embed_p95 = float(np.percentile(embed_latencies_ms, 95)) if embed_latencies_ms else 0.0
    summary = PipelineResult(
        run_dir=paths.root,
        embedder=cfg.embedder.name,
        duration_sec=duration_sec,
        frames_analyzed=len(samples),
        segments=len(segments),
        keyframes=len(keyframes),
        caption_text=caption_text,
        caption_latency_sec=caption_latency,
        embed_latency_ms_avg=round(float(np.mean(embed_latencies_ms)) if embed_latencies_ms else 0.0, 2),
        embed_latency_ms_p95=round(embed_p95, 2),
        wall_time_sec=round(time.perf_counter() - wall_start, 2),
    )
    write_json({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(summary).items()},
        "embed_latency_ms_p50": round(embed_p50, 2),
    }, paths.root / "summary.json")

    _write_run_readme(paths, cfg, summary, segments, keyframes)
    log.info("[done] run dir: %s", paths.root)

    yield ProgressEvent(
        stage="done",
        message=(f"Complete: {summary.frames_analyzed} frames -> {summary.segments} segments "
                 f"-> {summary.keyframes} keyframes  ({summary.wall_time_sec:.1f}s wall)"),
        run_dir=paths.root,
        frames_analyzed=summary.frames_analyzed,
        segments_so_far=summary.segments,
        keyframes_so_far=summary.keyframes,
        embed_latency_ms_avg=summary.embed_latency_ms_avg,
        timeline_path=timeline_p,
        similarity_curve_path=sim_p,
        keyframes_grid_path=grid_p,
        film_strip_path=film_p,
        keyframes=keyframes,
        final_result=summary,
        caption_text=caption_text,
    )


def run_pipeline(source_spec: str | int | Path, cfg: PipelineConfig) -> PipelineResult:
    last_result: Optional[PipelineResult] = None
    for event in iter_pipeline(source_spec, cfg):
        if event.final_result is not None:
            last_result = event.final_result
    assert last_result is not None
    return last_result


def _write_run_readme(
    paths: RunPaths,
    cfg: PipelineConfig,
    summary: PipelineResult,
    segments: list[Segment],
    keyframes: list[Keyframe],
) -> None:
    md: list[str] = []
    md += [
        f"# Run: {paths.root.name}",
        "",
        "## Pipeline summary",
        "",
        f"- Embedder: `{summary.embedder}` ({cfg.embedder.device})",
        f"- Analyzed frames: **{summary.frames_analyzed}** "
        f"(sampling = every {cfg.sampler.interval_sec:.2f}s)",
        f"- Segments detected: **{summary.segments}** "
        f"(sim threshold {cfg.segmenter.sim_threshold}, "
        f"min shot {cfg.segmenter.min_shot_sec:.1f}s, "
        f"{'EMA' if cfg.segmenter.use_ema else 'arithmetic'} mean)",
        f"- Keyframes selected: **{summary.keyframes}** "
        f"(weights sharp/repr = {cfg.selector.sharpness_weight}/{cfg.selector.representativeness_weight})",
        f"- Pipeline wall time: {summary.wall_time_sec:.2f}s",
        f"- Embed latency: mean {summary.embed_latency_ms_avg:.2f}ms / p95 {summary.embed_latency_ms_p95:.2f}ms",
        f"- LLM caption latency: {summary.caption_latency_sec:.2f}s",
        f"- Speed knobs: batch={cfg.batch_size}, embed_max_w={cfg.embed_max_width}, "
        f"cache_w={cfg.cache_thumb_width}, async_writes={cfg.async_disk_writes}",
        "",
        "## Files",
        "",
        "| Path | Purpose |",
        "| --- | --- |",
        "| `frames/` | Every analyzed frame, cached at thumbnail size. |",
        "| `keyframes/` | Selected keyframes + `keyframes.csv` metadata. |",
        "| `viz/timeline.jpg` | Segment-coloured timeline with keyframe markers. |",
        "| `viz/similarity_curve.jpg` | Per-frame cosine similarity vs running shot mean. |",
        "| `viz/keyframes_grid.jpg` | Contact sheet of selected keyframes. |",
        "| `caption/caption.md` | LLM narration of the video. |",
        "| `logs/stream_events.jsonl` | Per-frame decisions as they were emitted (real-time log). |",
        "| `logs/analyzed_frames.csv` | Tabular view of analyzed frames. |",
        "| `logs/segments.csv` | Segment boundaries. |",
        "| `logs/config.json` | Exact pipeline config used for this run. |",
        "| `summary.json` | Machine-readable summary (durations, counts, latencies). |",
        "",
        "## Segments",
        "",
        "| ID | Start (s) | End (s) | Duration (s) | Frames |",
        "| --- | --- | --- | --- | --- |",
    ]
    for seg in segments:
        md.append(
            f"| {seg.segment_id} | {seg.start_sec:.2f} | {seg.end_sec:.2f} "
            f"| {seg.duration_sec:.2f} | {seg.num_frames} |"
        )

    md += [
        "",
        "## Keyframes",
        "",
        "| ID | Segment | Time (s) | Sharpness | Repr (cos to centroid) | Composite |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for kf in keyframes:
        md.append(
            f"| KF{kf.keyframe_id:02d} | #{kf.segment_id} | {kf.timestamp_sec:.2f} "
            f"| {kf.sharpness:.0f} | {kf.representativeness:.3f} | {kf.composite_score:+.2f} |"
        )

    md += ["", "## Caption", "", summary.caption_text or "(no caption)"]
    (paths.root / "README.md").write_text("\n".join(md), encoding="utf-8")
