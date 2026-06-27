"""Gradio web UI for the keyframe pipeline.

Six tabs:

- Run               upload or pick a local video, watch the timeline build
                    live with streaming per-shot captions + final synthesis.
- Live              point the pipeline at a webcam index or RTSP/HTTP URL
                    for a bounded capture window. Server-side capture, so the
                    camera must be reachable from the host running Gradio.
- Compare           same video through N embedders side-by-side: one column
                    per model with stats, film strip and keyframes gallery.
                    Each compare run gets its own timestamped output folder.
- Caption bench     run the keyframe pipeline once, then score multiple LLM
                    caption models on the same keyframes. Side-by-side text.
- History           browse past runs in outputs/ and reload their artefacts.
- About             architecture + recommended parameters.

Everything writes to outputs/ so the CLI, Compare, Caption bench and History
tabs all share the same on-disk format.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

# macOS system proxies sometimes 502 the localhost self-probe gradio makes
# at launch. Force httpx to bypass proxies for the loopback. Must run before
# gradio import.
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import gradio as gr

from .captioner import caption_keyframes
from .config import (
    CaptionerConfig, EmbedderConfig, PipelineConfig, SamplerConfig,
    SegmenterConfig, SelectorConfig,
)
from .embedders import list_embedders
from .logging_setup import setup_logging
from .pipeline import ProgressEvent, iter_pipeline
from .selector import Keyframe


DEFAULT_EMBEDDER = "yolov8n"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_VIDEO_DIR = PROJECT_ROOT / "video"


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #

def _local_video_choices() -> list[str]:
    """Return paths of files in video/ so the Run tab can skip the upload step."""
    if not LOCAL_VIDEO_DIR.exists():
        return []
    out: list[str] = []
    for ext in (".MOV", ".mov", ".mp4", ".mkv", ".avi", ".webm"):
        out.extend(str(p) for p in sorted(LOCAL_VIDEO_DIR.glob(f"*{ext}")))
    return out


def _build_config(
    embedder_name: str,
    sample_interval: float,
    sim_threshold: float,
    min_shot_sec: float,
    diversity_thr: float,
    max_per_segment: int,
    enable_caption: bool,
    caption_model: str,
    output_root: str,
    *,
    segment_model: str = "",
    realtime_pacing: bool = False,
) -> PipelineConfig:
    return PipelineConfig(
        sampler=SamplerConfig(interval_sec=float(sample_interval)),
        segmenter=SegmenterConfig(
            sim_threshold=float(sim_threshold),
            min_shot_sec=float(min_shot_sec),
        ),
        selector=SelectorConfig(
            max_frames_per_segment=int(max_per_segment),
            diversity_sim_threshold=float(diversity_thr),
        ),
        captioner=CaptionerConfig(
            enabled=bool(enable_caption),
            model=str(caption_model),
            segment_model=str(segment_model or ""),
        ),
        embedder=EmbedderConfig(name=str(embedder_name), device="auto"),
        output_root=Path(output_root),
        realtime_pacing=realtime_pacing,
    )


def _stage_icon(stage: str) -> str:
    return {
        "init": "[init]",
        "frame": "[frame]",
        "segment_closed": "[shot]",
        "segment_caption": "[say]",
        "select": "[pick]",
        "visualize": "[viz]",
        "caption": "[llm]",
        "done": "[done]",
    }.get(stage, "[?]")


def _stats_md(ev: ProgressEvent) -> str:
    parts = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Frames analyzed | **{ev.frames_analyzed}** |",
        f"| Segments | **{ev.segments_so_far}** |",
        f"| Keyframes | **{ev.keyframes_so_far}** |",
        f"| Embed latency (mean) | **{ev.embed_latency_ms_avg:.1f} ms** |",
    ]
    if ev.final_result is not None:
        fr = ev.final_result
        parts += [
            f"| Wall time | **{fr.wall_time_sec:.2f} s** |",
            f"| Embed p95 | {fr.embed_latency_ms_p95:.1f} ms |",
            f"| Caption latency | {fr.caption_latency_sec:.2f} s |",
        ]
    return "\n".join(parts)


def _live_narration_md(captions: list) -> str:
    if not captions:
        return "_(narration will stream here as each shot is captioned.)_"
    lines = ["### Live narration", ""]
    for cap in captions:
        lines.append(
            f"**Shot {cap.segment_id} · {cap.start_sec:.1f}–{cap.end_sec:.1f}s** — "
            f"{cap.text}"
        )
        lines.append("")
    return "\n".join(lines)


def _gallery_items(keyframes: list) -> list[tuple[str, str]]:
    return [
        (k.bgr_path,
         f"KF{k.keyframe_id:02d}   t={k.timestamp_sec:.1f}s   seg #{k.segment_id}")
        for k in keyframes
    ]


# --------------------------------------------------------------------------- #
# Tab: Run                                                                    #
# --------------------------------------------------------------------------- #

def _resolve_video_input(uploaded: Optional[str], local: Optional[str]) -> Optional[str]:
    """Prefer an explicit local pick over an upload (uploads are temp paths)."""
    if local:
        return local
    if uploaded:
        return uploaded
    return None


def _run_single(
    uploaded_path: str,
    local_path: str,
    embedder_name: str,
    sample_interval: float,
    sim_threshold: float,
    min_shot_sec: float,
    diversity_thr: float,
    max_per_segment: int,
    enable_caption: bool,
    caption_model: str,
    segment_model: str,
    output_root: str,
) -> Iterator[tuple]:
    video_path = _resolve_video_input(uploaded_path, local_path)
    if not video_path:
        yield (None, "_(idle: upload or pick a video.)_", "_(no stats yet.)_",
               [], "_(narration will stream here as each shot is captioned.)_",
               "_(no caption yet.)_", "")
        return

    cfg = _build_config(
        embedder_name, sample_interval, sim_threshold, min_shot_sec,
        diversity_thr, max_per_segment, enable_caption, caption_model,
        output_root, segment_model=segment_model,
    )

    status_log: list[str] = []
    last_film: str | None = None
    last_gallery: list[tuple[str, str]] = []
    last_stats = "_(starting...)_"
    caption = "_(captioning runs after pipeline finishes.)_"
    narration = "_(narration will stream here as each shot is captioned.)_"
    run_dir = ""

    for ev in iter_pipeline(video_path, cfg, refresh_viz_every=4):
        status_log.append(f"{_stage_icon(ev.stage)} {ev.message}")
        if len(status_log) > 60:
            status_log = status_log[-60:]
        if ev.run_dir is not None:
            run_dir = str(ev.run_dir)
        if ev.film_strip_path:
            last_film = str(ev.film_strip_path)
        if ev.keyframes:
            last_gallery = _gallery_items(ev.keyframes)
        last_stats = _stats_md(ev)
        if ev.segment_captions:
            narration = _live_narration_md(ev.segment_captions)
        if ev.caption_text:
            caption = ev.caption_text
        yield (last_film,
               "\n".join(status_log[-12:]),
               last_stats,
               last_gallery,
               narration,
               caption,
               f"`{run_dir}`")


# --------------------------------------------------------------------------- #
# Tab: Live (webcam / RTSP)                                                   #
# --------------------------------------------------------------------------- #

def _resolve_live_source(spec: str) -> str | int:
    s = (spec or "").strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    return s


def _run_live(
    source_spec: str,
    embedder_name: str,
    sample_interval: float,
    sim_threshold: float,
    max_duration_sec: float,
    enable_caption: bool,
    caption_model: str,
    output_root: str,
) -> Iterator[tuple]:
    src = _resolve_live_source(source_spec)
    cfg = _build_config(
        embedder_name, sample_interval, sim_threshold, 2.0, 0.97, 3,
        enable_caption, caption_model, output_root,
        realtime_pacing=False,
    )

    status_log: list[str] = []
    last_film: str | None = None
    last_gallery: list[tuple[str, str]] = []
    last_stats = "_(starting...)_"
    narration = "_(narration will stream as each shot is captioned.)_"
    run_dir = ""
    t0 = time.perf_counter()
    stopped = False

    yield (last_film,
           f"[live] opening source: `{src}` ...",
           last_stats, last_gallery, narration, "")

    try:
        for ev in iter_pipeline(src, cfg, refresh_viz_every=4):
            elapsed = time.perf_counter() - t0
            status_log.append(f"{_stage_icon(ev.stage)} t+{elapsed:5.1f}s  {ev.message}")
            if len(status_log) > 80:
                status_log = status_log[-80:]
            if ev.run_dir is not None:
                run_dir = str(ev.run_dir)
            if ev.film_strip_path:
                last_film = str(ev.film_strip_path)
            if ev.keyframes:
                last_gallery = _gallery_items(ev.keyframes)
            last_stats = _stats_md(ev)
            if ev.segment_captions:
                narration = _live_narration_md(ev.segment_captions)

            yield (last_film,
                   "\n".join(status_log[-14:]),
                   last_stats,
                   last_gallery,
                   narration,
                   f"`{run_dir}`")

            if elapsed >= max_duration_sec and not stopped:
                stopped = True
                status_log.append(
                    f"[live] reached max-duration cap ({max_duration_sec:.0f}s); "
                    "ending capture.")
                yield (last_film, "\n".join(status_log[-14:]),
                       last_stats, last_gallery, narration, f"`{run_dir}`")
                break
    except Exception as exc:  # noqa: BLE001
        status_log.append(f"[live] error: {type(exc).__name__}: {exc}")
        yield (last_film, "\n".join(status_log[-14:]),
               last_stats, last_gallery, narration, f"`{run_dir}`")


# --------------------------------------------------------------------------- #
# Tab: Compare embedders                                                      #
# --------------------------------------------------------------------------- #

def _compare_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _compare_table_md(rows: list[dict]) -> str:
    md = ["| Embedder | Segments | Keyframes | Embed mean | Wall (s) |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        md.append(
            f"| `{r['embedder']}` | {r['segments']} | {r['keyframes']} | "
            f"{r['embed_ms']:.1f} ms | {r['wall_s']:.1f} |"
        )
    return "\n".join(md)


def _run_compare(
    uploaded_path: str,
    local_path: str,
    embedders: list[str],
    sample_interval: float,
    sim_threshold: float,
    min_shot_sec: float,
    output_root: str,
) -> Iterator[tuple]:
    video_path = _resolve_video_input(uploaded_path, local_path)
    empty_film: str | None = None
    if not video_path:
        yield ("_(upload or pick a video first.)_", empty_film,
               [], [], "_(no log yet.)_", "")
        return
    if not embedders:
        yield ("_(pick at least one embedder.)_", empty_film,
               [], [], "_(no log yet.)_", "")
        return

    compare_dir = Path(output_root) / _compare_run_id()
    compare_dir.mkdir(parents=True, exist_ok=True)
    src_name = Path(video_path).stem
    (compare_dir / "compare.json").write_text(json.dumps({
        "video": video_path,
        "embedders": list(embedders),
        "sample_interval": sample_interval,
        "sim_threshold": sim_threshold,
        "min_shot_sec": min_shot_sec,
    }, indent=2), encoding="utf-8")

    table_rows: list[dict] = []
    film_strips: list[tuple[str, str]] = []
    keyframes_gallery: list[tuple[str, str]] = []
    status: list[str] = [f"[compare] writing into `{compare_dir}`"]

    for emb in embedders:
        status.append(f"[compare] running `{emb}` ...")
        yield (_compare_table_md(table_rows), film_strips[-1][0] if film_strips else empty_film,
               film_strips, keyframes_gallery, "\n".join(status[-12:]),
               f"`{compare_dir}`")
        cfg = _build_config(
            emb, sample_interval, sim_threshold, min_shot_sec, 0.97, 3,
            enable_caption=False, caption_model="gpt-5.4",
            output_root=str(compare_dir),
        )
        t0 = time.perf_counter()
        last: ProgressEvent | None = None
        try:
            for ev in iter_pipeline(video_path, cfg, refresh_viz_every=20):
                last = ev
        except Exception as exc:  # noqa: BLE001
            status.append(f"[compare] `{emb}` FAILED: {type(exc).__name__}: {exc}")
            yield (_compare_table_md(table_rows),
                   film_strips[-1][0] if film_strips else empty_film,
                   film_strips, keyframes_gallery,
                   "\n".join(status[-12:]), f"`{compare_dir}`")
            continue

        wall = time.perf_counter() - t0
        if last is None or last.final_result is None:
            status.append(f"[compare] `{emb}` produced no result")
            continue
        fr = last.final_result
        table_rows.append({
            "embedder": emb,
            "segments": fr.segments,
            "keyframes": fr.keyframes,
            "embed_ms": fr.embed_latency_ms_avg,
            "wall_s": round(wall, 2),
            "run_dir": str(fr.run_dir),
        })
        if last.film_strip_path and Path(last.film_strip_path).exists():
            film_strips.append((
                str(last.film_strip_path),
                f"{emb}  ·  {fr.segments} segs  ·  {fr.keyframes} KFs  ·  "
                f"{fr.embed_latency_ms_avg:.0f} ms/frame  ·  {wall:.1f}s wall",
            ))
        for k in last.keyframes:
            keyframes_gallery.append(
                (k.bgr_path,
                 f"{emb}  KF{k.keyframe_id:02d}  t={k.timestamp_sec:.1f}s")
            )
        status.append(
            f"[compare] `{emb}` done: {fr.segments} segs, {fr.keyframes} KFs, "
            f"{wall:.1f}s wall"
        )
        yield (_compare_table_md(table_rows),
               film_strips[-1][0],
               film_strips, keyframes_gallery,
               "\n".join(status[-12:]), f"`{compare_dir}`")

    # Final summary md sidecar
    summary_md = ["# Compare run", "",
                  f"- Video: `{video_path}`",
                  f"- Sample interval: {sample_interval}s",
                  f"- Sim threshold: {sim_threshold}",
                  f"- Min shot: {min_shot_sec}s",
                  "",
                  _compare_table_md(table_rows)]
    (compare_dir / "README.md").write_text("\n".join(summary_md) + "\n",
                                            encoding="utf-8")
    status.append(f"[compare] complete. report: {compare_dir / 'README.md'}")
    yield (_compare_table_md(table_rows),
           film_strips[-1][0] if film_strips else empty_film,
           film_strips, keyframes_gallery,
           "\n".join(status[-14:]), f"`{compare_dir}`")


# --------------------------------------------------------------------------- #
# Tab: Caption bench                                                          #
# --------------------------------------------------------------------------- #

def _parse_model_list(text: str) -> list[str]:
    raw = (text or "").replace(",", "\n")
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _run_caption_bench(
    uploaded_path: str,
    local_path: str,
    embedder_name: str,
    sample_interval: float,
    sim_threshold: float,
    models_text: str,
    detail: str,
    max_keyframes: int,
    output_root: str,
) -> Iterator[tuple]:
    video_path = _resolve_video_input(uploaded_path, local_path)
    if not video_path:
        yield ("_(upload or pick a video first.)_", [], "_(no captions yet.)_",
               "_(no log yet.)_", "")
        return
    models = _parse_model_list(models_text)
    if not models:
        yield ("_(enter at least one model id, one per line.)_", [],
               "_(no captions yet.)_", "_(no log yet.)_", "")
        return

    out_root = Path(output_root) / _compare_run_id()
    out_root.mkdir(parents=True, exist_ok=True)
    status: list[str] = [f"[capbench] writing into `{out_root}`",
                         f"[capbench] running keyframe pipeline once with `{embedder_name}` ..."]
    yield ("_(running pipeline first ...)_", [], "_(running pipeline ...)_",
           "\n".join(status[-12:]), f"`{out_root}`")

    pipe_cfg = _build_config(
        embedder_name, sample_interval, sim_threshold, 2.0, 0.97, 3,
        enable_caption=False, caption_model="gpt-5.4",
        output_root=str(out_root / "_pipeline"),
    )
    last: ProgressEvent | None = None
    for ev in iter_pipeline(video_path, pipe_cfg, refresh_viz_every=20):
        last = ev
    if last is None or last.final_result is None:
        status.append("[capbench] pipeline failed to produce keyframes")
        yield ("_(pipeline failed)_", [], "_(no captions)_",
               "\n".join(status[-12:]), f"`{out_root}`")
        return

    keyframes = last.keyframes
    gallery = _gallery_items(keyframes)
    duration_sec = last.final_result.duration_sec
    status.append(
        f"[capbench] pipeline done: {len(keyframes)} keyframes, "
        f"duration {duration_sec:.1f}s")
    yield (_caption_rows_md([]), gallery, "_(captioning ...)_",
           "\n".join(status[-12:]), f"`{out_root}`")

    rows: list[dict] = []
    for model_id in models:
        status.append(f"[capbench] captioning with `{model_id}` ...")
        yield (_caption_rows_md(rows), gallery, _captions_md(rows),
               "\n".join(status[-12:]), f"`{out_root}`")
        cap_cfg = CaptionerConfig(
            enabled=True, model=model_id, detail=detail,
            max_keyframes=int(max_keyframes), fallback_models=(),
        )
        try:
            res = caption_keyframes(keyframes, duration_sec, cap_cfg)
            rows.append({
                "model": model_id,
                "ok": True,
                "latency": round(res.latency_sec, 2),
                "chars": len(res.text),
                "words": len(res.text.split()),
                "text": res.text,
                "error": "",
            })
            status.append(
                f"[capbench] `{model_id}` -> {len(res.text)} chars, "
                f"{res.latency_sec:.2f}s")
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "model": model_id, "ok": False, "latency": 0.0,
                "chars": 0, "words": 0, "text": "",
                "error": f"{type(exc).__name__}: {exc}",
            })
            status.append(
                f"[capbench] `{model_id}` FAILED: {type(exc).__name__}: {exc}")
        yield (_caption_rows_md(rows), gallery, _captions_md(rows),
               "\n".join(status[-12:]), f"`{out_root}`")

    (out_root / "caption_bench.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# Caption bench\n\n"
        f"- Video: `{video_path}`\n"
        f"- Embedder: `{embedder_name}`\n"
        f"- Keyframes: {len(keyframes)}\n"
        f"- Detail: `{detail}`\n\n"
        + _caption_rows_md(rows) + "\n\n## Captions\n\n"
        + _captions_md(rows) + "\n",
        encoding="utf-8",
    )
    status.append(f"[capbench] complete. report: {out_root / 'README.md'}")
    yield (_caption_rows_md(rows), gallery, _captions_md(rows),
           "\n".join(status[-14:]), f"`{out_root}`")


def _caption_rows_md(rows: list[dict]) -> str:
    md = ["| Model | Latency (s) | Chars | Words | Status |",
          "| --- | ---: | ---: | ---: | --- |"]
    for r in rows:
        status = "ok" if r["ok"] else f"`{r['error'][:50]}`"
        md.append(
            f"| `{r['model']}` | {r['latency']:.2f} | "
            f"{r['chars']} | {r['words']} | {status} |"
        )
    return "\n".join(md)


def _captions_md(rows: list[dict]) -> str:
    if not rows:
        return "_(no captions yet.)_"
    parts: list[str] = []
    for r in rows:
        parts.append(f"### `{r['model']}`")
        if not r["ok"]:
            parts.append(f"_failed: {r['error']}_")
        else:
            parts.append(
                f"_{r['latency']:.2f}s · {r['chars']} chars · {r['words']} words_")
            parts.append("")
            parts.append(r["text"])
        parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Tab: History                                                                #
# --------------------------------------------------------------------------- #

def _list_runs(output_root: str) -> list[str]:
    """Return run-dir paths in outputs/, newest first.

    A directory is considered a run if it contains a ``summary.json`` or
    ``compare.json`` or ``caption_bench.json`` marker.
    """
    root = Path(output_root)
    if not root.exists():
        return []
    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        # the keyframe run dirs include summary.json
        if (child / "summary.json").exists() or \
           (child / "compare.json").exists() or \
           (child / "caption_bench.json").exists():
            candidates.append(child)
        else:
            # recurse one level (compare/_compareid/embedder/run/)
            for sub in child.iterdir():
                if sub.is_dir() and (sub / "summary.json").exists():
                    candidates.append(sub)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(c) for c in candidates[:200]]


def _load_run(run_dir: str) -> tuple[str, str | None, list[tuple[str, str]], str, str]:
    """Return (overview_md, film_strip_path, gallery_items, caption_md, raw_json)."""
    if not run_dir or not Path(run_dir).exists():
        return ("_(pick a run from the dropdown.)_", None, [],
                "_(no caption.)_", "")
    p = Path(run_dir)
    summary_path = p / "summary.json"
    overview = [f"# {p.name}", "", f"`{p}`", ""]
    raw_json = ""
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            raw_json = json.dumps(data, indent=2, ensure_ascii=False)
            overview += [
                f"- Embedder: `{data.get('embedder')}`",
                f"- Duration: {data.get('duration_sec', 0):.1f}s",
                f"- Frames analyzed: {data.get('frames_analyzed')}",
                f"- Segments: {data.get('segments')}",
                f"- Keyframes: {data.get('keyframes')}",
                f"- Embed mean: {data.get('embed_latency_ms_avg')} ms",
                f"- Embed p95: {data.get('embed_latency_ms_p95')} ms",
                f"- Wall time: {data.get('wall_time_sec')} s",
                f"- Caption latency: {data.get('caption_latency_sec')} s",
            ]
        except Exception as exc:  # noqa: BLE001
            overview.append(f"_(failed to parse summary.json: {exc})_")
    elif (p / "compare.json").exists():
        try:
            data = json.loads((p / "compare.json").read_text(encoding="utf-8"))
            raw_json = json.dumps(data, indent=2, ensure_ascii=False)
            overview += [
                "**Compare run**", "",
                f"- Video: `{data.get('video')}`",
                f"- Embedders: " + ", ".join(f"`{e}`" for e in data.get("embedders", [])),
                f"- Sample interval: {data.get('sample_interval')}s",
                f"- Sim threshold: {data.get('sim_threshold')}",
            ]
        except Exception as exc:  # noqa: BLE001
            overview.append(f"_(failed to parse compare.json: {exc})_")

    film_strip = p / "viz" / "film_strip.jpg"
    film_path = str(film_strip) if film_strip.exists() else None

    gallery: list[tuple[str, str]] = []
    kf_dir = p / "keyframes"
    if kf_dir.exists():
        for kf in sorted(kf_dir.glob("kf_*.jpg")):
            gallery.append((str(kf), kf.stem))

    caption = "_(no caption written.)_"
    cap_md = p / "caption" / "caption.md"
    if cap_md.exists():
        caption = cap_md.read_text(encoding="utf-8")
    elif (p / "README.md").exists():
        caption = (p / "README.md").read_text(encoding="utf-8")

    return ("\n".join(overview), film_path, gallery, caption, raw_json)


def _refresh_runs(output_root: str) -> dict:
    runs = _list_runs(output_root)
    if not runs:
        return gr.update(choices=[], value=None)
    return gr.update(choices=runs, value=runs[0])


# --------------------------------------------------------------------------- #
# CSS + layout                                                                #
# --------------------------------------------------------------------------- #

CSS = """
.gradio-container { max-width: 1700px !important; }
#film-strip img, #compare-strip img, #history-strip img {
    max-width: 100% !important; height: auto !important;
    image-rendering: -webkit-optimize-contrast;
}
#caption-card { background: #f5f7fb; color: #1a1d24 !important;
                padding: 18px 22px; border-radius: 10px;
                border-left: 4px solid #6c8cff; line-height: 1.65;
                font-size: 15px; }
#caption-card * { color: #1a1d24 !important; }
#narration-card { background: #fffaf0; color: #2a2418 !important;
                  padding: 14px 18px; border-radius: 10px;
                  border-left: 4px solid #d8a64d; line-height: 1.6;
                  font-size: 14px; min-height: 80px; }
#narration-card * { color: #2a2418 !important; }
#stats-card { background: #1f2330; color: #e8eaf2 !important;
              padding: 14px 18px; border-radius: 10px;
              font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
#stats-card *, #stats-card table, #stats-card td, #stats-card th,
#stats-card p, #stats-card strong { color: #e8eaf2 !important; }
#run-dir { font-size: 12px; opacity: 0.75; }
"""


ABOUT_MD = """
## What this tool does

`keyframe` watches a video, picks the small set of frames that actually carry
the narrative, and asks a multimodal LLM to write a factual narration of what
happens. The same code path runs offline (file → caption) and online (webcam
or RTSP → live keyframes + streaming captions).

## Five stages

1. **Sample** the source at a fixed interval (default 1 frame/sec).
2. **Embed** each frame into a vector with the selected model.
3. **Segment** the stream online: open a new shot when cosine similarity to
   the running EMA falls below the threshold.
4. **Select** keyframes per segment using a weighted z-score of sharpness +
   centroid similarity, then drop cross-segment duplicates.
5. **Caption** every shot as it closes (streaming), and write a final
   synthesis paragraph over the full keyframe set at the end.

## Capabilities

- **File mode** — any video ffmpeg can read. Hardware-accelerated decode on
  macOS / NVIDIA / Linux.
- **Live mode** — webcam index (e.g. `0`) or RTSP / HTTP camera URL. Pipeline
  streams keyframes + captions as the camera produces them.
- **Compare mode** — run the same video through N embedders side-by-side.
- **Caption bench** — score multiple LLM caption models on the same keyframe
  set.
- **History** — every run is self-contained in `outputs/<stem>_<timestamp>/`
  with a film strip, keyframe gallery, caption, raw events, and CSVs.

## Embedder picks (from this build's benchmarks)

| Workload | Recommended embedder | Why |
| --- | --- | --- |
| Realtime (webcam, low end) | `mobileclip-s0` | ~3 ms / frame, semantic |
| Offline best quality       | `siglip-b16`    | best retrieval scores |
| Default / general          | `yolov8n`       | 24 ms on MPS, no extra dep |
| CPU-only / no GPU          | `phash`         | <1 ms, gross changes only |

## Threshold cheat sheet

| Embedder family | `sim-threshold` | Why |
| --- | ---: | --- |
| YOLO (layout-based) | 0.96 – 0.97 | similarities run hot |
| CLIP / SigLIP / MobileCLIP (semantic) | 0.92 – 0.94 | semantic shifts produce lower similarity |
| phash / hsv | 0.90 – 0.92 | low-frequency only |

`min-shot-sec`: 2 s for tripod / stable footage, 3 – 4 s for handheld phone
video to suppress autofocus and shake flicker.
"""


def build_app() -> gr.Blocks:
    embedder_choices = list_embedders()
    local_videos = _local_video_choices()
    with gr.Blocks(title="Keyframe Pipeline", css=CSS) as demo:
        gr.Markdown(
            "# Keyframe Pipeline\n"
            "_Upload a video. Watch the timeline build live. "
            "Read what the model thinks happens, shot by shot._"
        )

        # ------------------ Tab: Run ------------------------------------- #
        with gr.Tab("Run"):
            with gr.Row():
                with gr.Column(scale=1, min_width=340):
                    video_in = gr.Video(label="Upload", sources=["upload"])
                    local_pick = gr.Dropdown(
                        local_videos,
                        value=local_videos[0] if local_videos else None,
                        label=f"…or pick from video/ ({len(local_videos)} files)",
                        allow_custom_value=True, interactive=True,
                    )
                    embedder = gr.Dropdown(
                        embedder_choices, value=DEFAULT_EMBEDDER, label="Embedder",
                    )
                    sample_interval = gr.Slider(0.25, 5.0, value=1.0, step=0.25,
                                                label="Sample every N seconds")
                    sim_threshold = gr.Slider(0.80, 0.99, value=0.96, step=0.005,
                                              label="Shot-cut threshold (lower → more cuts)")
                    with gr.Accordion("Advanced", open=False):
                        min_shot_sec = gr.Slider(0.0, 8.0, value=2.0, step=0.5,
                                                 label="Anti-flicker min shot (s)")
                        diversity_thr = gr.Slider(0.80, 1.00, value=0.97, step=0.01,
                                                  label="Drop duplicates above")
                        max_per_segment = gr.Slider(1, 6, value=3, step=1,
                                                    label="Max keyframes per segment")
                        enable_caption = gr.Checkbox(value=True, label="LLM caption")
                        caption_model = gr.Textbox(value="gpt-5.4",
                                                   label="Final-synthesis model")
                        segment_model = gr.Textbox(
                            value="",
                            label="Per-shot streaming model (blank = reuse final model)",
                        )
                        output_root = gr.Textbox(value="outputs", label="Output dir")
                    run_btn = gr.Button("Run", variant="primary", size="lg")

                with gr.Column(scale=2):
                    stats_md = gr.Markdown("_(no run yet.)_", elem_id="stats-card")
                    status_log = gr.Textbox(label="Status", lines=12, max_lines=12,
                                            interactive=False, autoscroll=True,
                                            value="_(idle.)_")
                    run_dir_md = gr.Markdown("", elem_id="run-dir")

            gr.Markdown("### Timeline")
            film_strip = gr.Image(
                show_label=False, type="filepath",
                interactive=False, elem_id="film-strip", height=240,
            )

            gr.Markdown("### Keyframes")
            keyframes_gallery = gr.Gallery(
                show_label=False, columns=4, rows=2, height=560,
                object_fit="contain", allow_preview=True,
            )

            gr.Markdown("### Live narration · streamed shot-by-shot")
            narration_md = gr.Markdown(
                "_(narration will stream here as each shot is captioned.)_",
                elem_id="narration-card",
            )

            gr.Markdown("### Final synthesis")
            caption_md = gr.Markdown(
                "_(full-video caption appears after all keyframes are picked.)_",
                elem_id="caption-card",
            )

            run_btn.click(
                fn=_run_single,
                inputs=[
                    video_in, local_pick, embedder, sample_interval, sim_threshold,
                    min_shot_sec, diversity_thr, max_per_segment,
                    enable_caption, caption_model, segment_model, output_root,
                ],
                outputs=[film_strip, status_log, stats_md,
                         keyframes_gallery, narration_md, caption_md, run_dir_md],
                show_progress="minimal",
            )

        # ------------------ Tab: Live ------------------------------------ #
        with gr.Tab("Live"):
            gr.Markdown(
                "**Server-side capture.** Webcam (index `0`, `1`, …) or any "
                "`rtsp://` / `http://` camera URL reachable from the host "
                "running this UI. The capture stops automatically when "
                "*Max duration* is reached.\n\n"
                "Tip: streaming captions need `OPENAI_API_KEY` set in `.env`. "
                "Without it the keyframe stream still works; only the LLM "
                "narration is skipped."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=340):
                    live_source = gr.Textbox(
                        value="0",
                        label="Webcam index or RTSP / HTTP URL",
                    )
                    live_embedder = gr.Dropdown(
                        embedder_choices, value="mobileclip-s0",
                        label="Embedder (recommend mobileclip-s0 for live)",
                    )
                    live_interval = gr.Slider(0.25, 3.0, value=1.0, step=0.25,
                                              label="Sample every N seconds")
                    live_sim = gr.Slider(0.80, 0.99, value=0.93, step=0.005,
                                         label="Shot-cut threshold")
                    live_duration = gr.Slider(10, 600, value=60, step=10,
                                              label="Max duration (s)")
                    live_caption = gr.Checkbox(value=True, label="Stream captions")
                    live_model = gr.Textbox(value="gpt-5.4", label="Caption model")
                    live_output = gr.Textbox(value="outputs", label="Output dir")
                    live_btn = gr.Button("Start capture", variant="primary",
                                         size="lg")
                with gr.Column(scale=2):
                    live_stats = gr.Markdown("_(idle.)_", elem_id="stats-card")
                    live_log = gr.Textbox(label="Live log", lines=14,
                                          max_lines=14, interactive=False,
                                          autoscroll=True, value="_(idle.)_")
                    live_run_dir = gr.Markdown("", elem_id="run-dir")

            gr.Markdown("### Live timeline")
            live_film = gr.Image(
                show_label=False, type="filepath",
                interactive=False, elem_id="film-strip", height=240,
            )

            gr.Markdown("### Keyframes as they're picked")
            live_gallery = gr.Gallery(
                show_label=False, columns=4, rows=2, height=480,
                object_fit="contain", allow_preview=True,
            )

            gr.Markdown("### Live narration")
            live_narration = gr.Markdown(
                "_(narration will stream as each shot is captioned.)_",
                elem_id="narration-card",
            )

            live_btn.click(
                fn=_run_live,
                inputs=[live_source, live_embedder, live_interval, live_sim,
                        live_duration, live_caption, live_model, live_output],
                outputs=[live_film, live_log, live_stats,
                         live_gallery, live_narration, live_run_dir],
                show_progress="minimal",
            )

        # ------------------ Tab: Compare embedders ----------------------- #
        with gr.Tab("Compare embedders"):
            gr.Markdown(
                "Run the same video through several embedders. Each gets its "
                "own row in the table, its own film strip, and its own slice "
                "of the keyframes gallery (labelled with the model name). "
                "Output goes to `outputs/compare/<timestamp>/<embedder>/...` "
                "so every comparison is a single browsable folder."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=340):
                    cmp_upload = gr.Video(label="Upload", sources=["upload"])
                    cmp_local = gr.Dropdown(
                        local_videos,
                        value=local_videos[0] if local_videos else None,
                        label="…or pick from video/",
                        allow_custom_value=True, interactive=True,
                    )
                    cmp_embedders = gr.CheckboxGroup(
                        embedder_choices,
                        value=["yolov8n", "mobileclip-s0", "siglip-b16"],
                        label="Embedders",
                    )
                    cmp_interval = gr.Slider(0.25, 5.0, value=1.0, step=0.25,
                                             label="Sample every N seconds")
                    cmp_sim = gr.Slider(0.80, 0.99, value=0.94, step=0.005,
                                        label="Shot-cut threshold")
                    cmp_min_shot = gr.Slider(0.0, 8.0, value=2.0, step=0.5,
                                             label="Min shot (s)")
                    cmp_output = gr.Textbox(value="outputs/compare",
                                            label="Output dir")
                    cmp_btn = gr.Button("Compare", variant="primary", size="lg")
                with gr.Column(scale=2):
                    cmp_table = gr.Markdown("_(no run yet.)_")
                    cmp_log = gr.Textbox(label="Log", lines=10, max_lines=10,
                                         interactive=False, autoscroll=True,
                                         value="_(idle.)_")
                    cmp_run_dir = gr.Markdown("", elem_id="run-dir")

            gr.Markdown("### Latest film strip (most recent model)")
            cmp_film = gr.Image(
                show_label=False, type="filepath",
                interactive=False, elem_id="compare-strip", height=240,
            )

            gr.Markdown("### All film strips, stacked")
            cmp_strips = gr.Gallery(
                show_label=True, columns=1, rows=1, height=700,
                object_fit="contain",
            )

            gr.Markdown("### Keyframes per model")
            cmp_kf_gallery = gr.Gallery(
                show_label=False, columns=4, rows=2, height=560,
                object_fit="contain", allow_preview=True,
            )

            cmp_btn.click(
                fn=_run_compare,
                inputs=[cmp_upload, cmp_local, cmp_embedders, cmp_interval,
                        cmp_sim, cmp_min_shot, cmp_output],
                outputs=[cmp_table, cmp_film, cmp_strips, cmp_kf_gallery,
                         cmp_log, cmp_run_dir],
                show_progress="minimal",
            )

        # ------------------ Tab: Caption bench --------------------------- #
        with gr.Tab("Caption bench"):
            gr.Markdown(
                "Score multiple LLM caption models on the **same** set of "
                "keyframes. The pipeline runs once with your chosen embedder; "
                "then each model captions the same frames so you can compare "
                "writing quality and latency apples-to-apples."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=340):
                    cb_upload = gr.Video(label="Upload", sources=["upload"])
                    cb_local = gr.Dropdown(
                        local_videos,
                        value=local_videos[0] if local_videos else None,
                        label="…or pick from video/",
                        allow_custom_value=True, interactive=True,
                    )
                    cb_embedder = gr.Dropdown(
                        embedder_choices, value=DEFAULT_EMBEDDER,
                        label="Embedder (for keyframe pass)",
                    )
                    cb_interval = gr.Slider(0.25, 5.0, value=1.0, step=0.25,
                                            label="Sample every N seconds")
                    cb_sim = gr.Slider(0.80, 0.99, value=0.96, step=0.005,
                                       label="Shot-cut threshold")
                    cb_models = gr.Textbox(
                        value=("gpt-5.4\n"
                               "google/gemini-2.5-flash\n"
                               "anthropic/claude-3.5-sonnet\n"
                               "qwen/qwen2.5-vl-72b-instruct"),
                        lines=6, label="LLM model ids (one per line)",
                    )
                    cb_detail = gr.Radio(["low", "high"], value="low",
                                         label="Image detail")
                    cb_max_kf = gr.Slider(4, 32, value=12, step=1,
                                          label="Max keyframes per call")
                    cb_output = gr.Textbox(value="outputs/caption-bench",
                                           label="Output dir")
                    cb_btn = gr.Button("Caption bench", variant="primary",
                                       size="lg")
                with gr.Column(scale=2):
                    cb_table = gr.Markdown("_(no run yet.)_")
                    cb_log = gr.Textbox(label="Log", lines=10, max_lines=10,
                                        interactive=False, autoscroll=True,
                                        value="_(idle.)_")
                    cb_run_dir = gr.Markdown("", elem_id="run-dir")

            gr.Markdown("### Keyframes (shared across all models)")
            cb_gallery = gr.Gallery(
                show_label=False, columns=6, rows=2, height=320,
                object_fit="contain",
            )

            gr.Markdown("### Captions, side by side")
            cb_captions = gr.Markdown("_(no captions yet.)_",
                                      elem_id="caption-card")

            cb_btn.click(
                fn=_run_caption_bench,
                inputs=[cb_upload, cb_local, cb_embedder, cb_interval, cb_sim,
                        cb_models, cb_detail, cb_max_kf, cb_output],
                outputs=[cb_table, cb_gallery, cb_captions, cb_log, cb_run_dir],
                show_progress="minimal",
            )

        # ------------------ Tab: History --------------------------------- #
        with gr.Tab("History"):
            gr.Markdown(
                "Every run leaves a self-contained folder under `outputs/`. "
                "Click **Refresh** to rescan, pick a run from the dropdown, "
                "and reload its artefacts here."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    hist_root = gr.Textbox(value="outputs",
                                           label="Outputs root")
                    hist_refresh = gr.Button("Refresh", size="sm")
                    initial_runs = _list_runs("outputs")
                    hist_pick = gr.Dropdown(
                        initial_runs,
                        value=initial_runs[0] if initial_runs else None,
                        label=f"Runs ({len(initial_runs)} found)",
                    )
                    hist_load = gr.Button("Load", variant="primary")
                with gr.Column(scale=2):
                    hist_overview = gr.Markdown("_(pick a run.)_")
                    hist_json = gr.Code(label="summary.json", language="json",
                                        lines=12, interactive=False)

            gr.Markdown("### Timeline")
            hist_strip = gr.Image(
                show_label=False, type="filepath",
                interactive=False, elem_id="history-strip", height=240,
            )

            gr.Markdown("### Keyframes")
            hist_gallery = gr.Gallery(
                show_label=False, columns=5, rows=2, height=480,
                object_fit="contain", allow_preview=True,
            )

            gr.Markdown("### Caption / report")
            hist_caption = gr.Markdown("_(no caption.)_",
                                       elem_id="caption-card")

            hist_refresh.click(fn=_refresh_runs, inputs=[hist_root],
                               outputs=[hist_pick])
            hist_load.click(
                fn=_load_run, inputs=[hist_pick],
                outputs=[hist_overview, hist_strip, hist_gallery,
                         hist_caption, hist_json],
            )

        # ------------------ Tab: About ----------------------------------- #
        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

    return demo


def main(argv: list[str] | None = None) -> int:
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Keyframe pipeline web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



