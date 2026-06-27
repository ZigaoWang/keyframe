"""Gradio web UI for the keyframe pipeline.

Layout, top to bottom:

1.  Controls bar           upload + embedder dropdown + a couple of sliders.
2.  Hero film strip        the only timeline you need. One thumbnail per
                           analyzed frame, coloured segment band below,
                           gold keyframe markers above, time labels.
                           Updates live as the pipeline runs.
3.  Stats card             counters and latency.
4.  Live event log         scrolling status, last few lines.
5.  Keyframes              big gallery, each tile is one picked frame.
6.  Caption                the LLM's narration of the whole video.

Comparison tab runs the same pipeline with N embedders on the same upload
and stacks their film strips so you can see which model picks better cuts.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterator

# macOS system proxies sometimes 502 the localhost self-probe gradio makes
# at launch. Force httpx to bypass proxies for the loopback. Must run before
# gradio import.
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import gradio as gr

from .config import (
    CaptionerConfig, EmbedderConfig, PipelineConfig, SamplerConfig,
    SegmenterConfig, SelectorConfig,
)
from .embedders import list_embedders
from .logging_setup import setup_logging
from .pipeline import ProgressEvent, iter_pipeline


DEFAULT_EMBEDDER = "yolov8n"


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
        captioner=CaptionerConfig(enabled=bool(enable_caption), model=str(caption_model)),
        embedder=EmbedderConfig(name=str(embedder_name), device="auto"),
        output_root=Path(output_root),
    )


def _stage_icon(stage: str) -> str:
    return {
        "init": "[init]",
        "frame": "[frame]",
        "segment_closed": "[shot]",
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


def _gallery_items(ev: ProgressEvent) -> list[tuple[str, str]]:
    return [
        (k.bgr_path,
         f"KF{k.keyframe_id:02d}   t={k.timestamp_sec:.1f}s   seg #{k.segment_id}")
        for k in ev.keyframes
    ]


def _run_single(
    video_path: str,
    embedder_name: str,
    sample_interval: float,
    sim_threshold: float,
    min_shot_sec: float,
    diversity_thr: float,
    max_per_segment: int,
    enable_caption: bool,
    caption_model: str,
    output_root: str,
) -> Iterator[tuple]:
    if not video_path:
        yield (None, "_(idle: drop a video first.)_", "_(no stats yet.)_",
               [], "_(no caption yet.)_")
        return

    cfg = _build_config(
        embedder_name, sample_interval, sim_threshold, min_shot_sec,
        diversity_thr, max_per_segment, enable_caption, caption_model, output_root,
    )

    status_log: list[str] = []
    last_film: str | None = None
    last_gallery: list[tuple[str, str]] = []
    last_stats = "_(starting...)_"
    caption = "_(captioning runs after pipeline finishes.)_"

    for ev in iter_pipeline(video_path, cfg, refresh_viz_every=4):
        status_log.append(f"{_stage_icon(ev.stage)} {ev.message}")
        if len(status_log) > 60:
            status_log = status_log[-60:]
        if ev.film_strip_path:
            last_film = str(ev.film_strip_path)
        if ev.keyframes:
            last_gallery = _gallery_items(ev)
        last_stats = _stats_md(ev)
        if ev.caption_text:
            caption = ev.caption_text
        yield (last_film,
               "\n".join(status_log[-12:]),
               last_stats,
               last_gallery,
               caption)


def _run_comparison(
    video_path: str,
    embedders: list[str],
    sample_interval: float,
    sim_threshold: float,
    output_root: str,
) -> Iterator[tuple]:
    if not video_path:
        yield ("_(upload a video first.)_", [])
        return
    if not embedders:
        yield ("_(pick at least one embedder.)_", [])
        return

    table_rows: list[dict] = []
    film_strips: list[tuple[str, str]] = []

    for emb in embedders:
        cfg = _build_config(
            emb, sample_interval, sim_threshold, 2.0, 0.92, 3,
            enable_caption=False, caption_model="gpt-5.4", output_root=output_root,
        )
        t0 = time.perf_counter()
        last: ProgressEvent | None = None
        for ev in iter_pipeline(video_path, cfg, refresh_viz_every=20):
            last = ev
        wall = time.perf_counter() - t0
        if last is None or last.final_result is None:
            continue
        fr = last.final_result
        table_rows.append({
            "embedder": emb,
            "segments": fr.segments,
            "keyframes": fr.keyframes,
            "embed_ms": fr.embed_latency_ms_avg,
            "wall_s": round(wall, 2),
        })
        if last.film_strip_path and Path(last.film_strip_path).exists():
            film_strips.append((str(last.film_strip_path),
                                f"{emb} - {fr.segments} segs, {fr.keyframes} KFs, "
                                f"{fr.embed_latency_ms_avg:.0f}ms embed"))

        md = ["| Embedder | Segments | Keyframes | Embed mean | Wall |",
              "| --- | ---: | ---: | ---: | ---: |"]
        for r in table_rows:
            md.append(
                f"| `{r['embedder']}` | {r['segments']} | {r['keyframes']} | "
                f"{r['embed_ms']:.1f} ms | {r['wall_s']:.1f} s |"
            )
        yield ("\n".join(md), film_strips)


CSS = """
.gradio-container { max-width: 1600px !important; }
#film-strip img { max-width: 100% !important; height: auto !important;
                  image-rendering: -webkit-optimize-contrast; }
#caption-card { background: #f5f7fb; color: #1a1d24 !important;
                padding: 18px 22px; border-radius: 10px;
                border-left: 4px solid #6c8cff; line-height: 1.65;
                font-size: 15px; }
#caption-card * { color: #1a1d24 !important; }
#stats-card { background: #1f2330; color: #e8eaf2 !important;
              padding: 14px 18px; border-radius: 10px;
              font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
#stats-card *, #stats-card table, #stats-card td, #stats-card th,
#stats-card p, #stats-card strong { color: #e8eaf2 !important; }
"""


def build_app() -> gr.Blocks:
    embedder_choices = list_embedders()
    with gr.Blocks(title="Keyframe Pipeline", css=CSS) as demo:
        gr.Markdown(
            "# Keyframe Pipeline\n"
            "_Upload a video. Watch the timeline build as each frame is "
            "analyzed. Read what the LLM thinks happens._"
        )

        with gr.Tab("Run"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    video_in = gr.Video(label="Video", sources=["upload"])
                    embedder = gr.Dropdown(
                        embedder_choices, value=DEFAULT_EMBEDDER, label="Embedder",
                    )
                    sample_interval = gr.Slider(0.25, 5.0, value=1.0, step=0.25,
                                                label="Sample every N seconds")
                    sim_threshold = gr.Slider(0.80, 0.99, value=0.96, step=0.005,
                                              label="Shot cut sensitivity (lower = more cuts)")
                    with gr.Accordion("Advanced", open=False):
                        min_shot_sec = gr.Slider(0.0, 8.0, value=2.0, step=0.5,
                                                 label="Anti-flicker min shot (s)")
                        diversity_thr = gr.Slider(0.80, 1.00, value=0.97, step=0.01,
                                                  label="Drop near-duplicates above (lower = fewer keyframes)")
                        max_per_segment = gr.Slider(1, 6, value=3, step=1,
                                                    label="Max keyframes per segment")
                        enable_caption = gr.Checkbox(value=True, label="LLM caption")
                        caption_model = gr.Textbox(value="gpt-5.4", label="LLM model")
                        output_root = gr.Textbox(value="outputs", label="Output dir")
                    run_btn = gr.Button("Run", variant="primary", size="lg")

                with gr.Column(scale=2):
                    stats_md = gr.Markdown("_(no run yet.)_", elem_id="stats-card")
                    status_log = gr.Textbox(label="Status", lines=10, max_lines=10,
                                            interactive=False, autoscroll=True,
                                            value="_(idle.)_")

            gr.Markdown("### Timeline")
            gr.Markdown(
                "_Every analyzed frame as a thumbnail in time order. "
                "Coloured band below = segment ID. Gold marker above = picked keyframe._",
            )
            film_strip = gr.Image(
                label=None, show_label=False, type="filepath",
                interactive=False, elem_id="film-strip", height=240,
            )

            gr.Markdown("### Keyframes")
            keyframes_gallery = gr.Gallery(
                label=None, show_label=False, columns=4, rows=2, height=560,
                object_fit="contain", allow_preview=True,
            )

            gr.Markdown("### LLM caption")
            caption_md = gr.Markdown(
                "_(caption appears after all keyframes are picked.)_",
                elem_id="caption-card",
            )

            run_btn.click(
                fn=_run_single,
                inputs=[
                    video_in, embedder, sample_interval, sim_threshold,
                    min_shot_sec, diversity_thr, max_per_segment,
                    enable_caption, caption_model, output_root,
                ],
                outputs=[film_strip, status_log, stats_md,
                         keyframes_gallery, caption_md],
                show_progress="minimal",
            )

        with gr.Tab("Compare models"):
            gr.Markdown(
                "Run the same video through multiple embedders. "
                "LLM caption is disabled in this tab. Look at the stacked "
                "film strips: a good embedder shows clear segment colour blocks "
                "aligned with real shot cuts; a bad one shows random colour noise."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    cmp_video = gr.Video(label="Video", sources=["upload"])
                    cmp_embedders = gr.CheckboxGroup(
                        embedder_choices,
                        value=[DEFAULT_EMBEDDER, "yolo11n", "mobile_sam"],
                        label="Embedders",
                    )
                    cmp_sample = gr.Slider(0.25, 5.0, value=1.0, step=0.25,
                                           label="Sample every N seconds")
                    cmp_sim = gr.Slider(0.80, 0.99, value=0.94, step=0.005,
                                        label="Shot cut threshold")
                    cmp_output = gr.Textbox(value="outputs/compare",
                                            label="Output dir")
                    cmp_btn = gr.Button("Compare", variant="primary", size="lg")
                with gr.Column(scale=2):
                    cmp_table = gr.Markdown("_(no run yet.)_")
                    cmp_strips = gr.Gallery(
                        label="Film strips", show_label=True, columns=1,
                        rows=1, height=700, object_fit="contain",
                    )
            cmp_btn.click(
                fn=_run_comparison,
                inputs=[cmp_video, cmp_embedders, cmp_sample, cmp_sim, cmp_output],
                outputs=[cmp_table, cmp_strips],
                show_progress="minimal",
            )

        with gr.Tab("About"):
            gr.Markdown(
                "## How it works\n"
                "1. **Sample** one frame every N seconds from the video.\n"
                "2. **Embed** each sample into a 256-d (or similar) vector "
                "using the selected model.\n"
                "3. **Segment** the stream online: open a new shot whenever the "
                "cosine similarity to the running EMA of the current shot's "
                "embedding falls below the threshold. An anti-flicker guard "
                "rejects sub-second shots.\n"
                "4. **Select** keyframes inside each segment by sharpness + "
                "centroid similarity, then **filter cross-segment duplicates** "
                "using cosine similarity so a long static scene split by EMA "
                "drift only contributes one keyframe.\n"
                "5. **Caption** all selected keyframes in a single multi-image "
                "request to the chat model.\n"
                "\n"
                "## Embedders\n"
                "| Name | Backbone | Dim |\n"
                "| --- | --- | ---: |\n"
                "| yolov8n / s / m | YOLOv8 detector backbone | 256 / 512 / 576 |\n"
                "| yolov8n-seg | YOLOv8 segmentation backbone | 256 |\n"
                "| yolo11n / s | YOLO11 detector backbone | 256 / 512 |\n"
                "| yolo26n | YOLO26 NMS-free detector backbone | 256 |\n"
                "| mobile_sam | MobileSAM TinyViT encoder (global-pooled) | 256 |\n"
                "| phash | 8x8 DCT perceptual hash | 64 |\n"
                "| hsv | HSV histogram | 96 |\n"
            )
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
