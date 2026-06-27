"""Command-line entrypoint for the keyframe pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    CaptionerConfig, EmbedderConfig, PipelineConfig, SamplerConfig,
    SegmenterConfig, SelectorConfig,
)
from .embedders import list_embedders
from .logging_setup import setup_logging
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="keyframe",
        description=(
            "Stream-friendly keyframe extraction + LLM video understanding. "
            "Runs the same algorithm offline (file in -> caption out) and online "
            "(webcam / RTSP -> live keyframe events)."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- run ------------------------------------------------------------- #
    run = sub.add_parser("run", help="Run the full pipeline on one source.")
    run.add_argument("source", help="Video file path, integer webcam index, or rtsp:// url.")
    run.add_argument("--output-root", default="outputs",
                     help="Base directory for per-run output folders. Default: outputs/")
    run.add_argument("--realtime", action="store_true",
                     help="When source is a file, sleep so the loop ticks at the source fps.")
    run.add_argument("--no-caption", action="store_true",
                     help="Skip the LLM caption stage. Pipeline still produces keyframes + viz.")

    sampler = run.add_argument_group("sampler")
    sampler.add_argument("--sample-interval", type=float, default=1.0,
                         help="Seconds between analyzed frames. Default 1.0.")

    seg = run.add_argument_group("segmenter")
    seg.add_argument("--sim-threshold", type=float, default=0.96,
                     help="cos_sim below this triggers a new segment. Default 0.96.")
    seg.add_argument("--min-shot-sec", type=float, default=2.0,
                     help="Reject segments shorter than this many seconds (anti-flicker). Default 2.0.")
    seg.add_argument("--ema-alpha", type=float, default=0.25,
                     help="EMA weight for the new frame in shot mean. Default 0.25.")
    seg.add_argument("--no-ema", action="store_true",
                     help="Use arithmetic mean instead of EMA for the shot mean.")

    sel = run.add_argument_group("selector")
    sel.add_argument("--max-per-segment", type=int, default=3,
                     help="Hard cap on keyframes per segment. Default 3.")
    sel.add_argument("--seconds-per-keyframe", type=float, default=8.0,
                     help="Roughly one keyframe per N seconds of segment. Default 8.0.")
    sel.add_argument("--min-spacing-sec", type=float, default=1.0,
                     help="Minimum temporal gap between picked frames inside one segment. Default 1.0.")
    sel.add_argument("--sharpness-weight", type=float, default=0.45,
                     help="Weight on sharpness in composite score. Default 0.45.")
    sel.add_argument("--repr-weight", type=float, default=0.55,
                     help="Weight on representativeness in composite score. Default 0.55.")

    emb = run.add_argument_group("embedder")
    emb.add_argument("--embedder", default="yolov8n",
                     choices=list_embedders(),
                     help="Which model to embed frames with. Default yolov8n.")
    emb.add_argument("--device", default="auto",
                     help="Torch device: auto | cpu | cuda | mps. Default auto.")

    cap = run.add_argument_group("captioner")
    cap.add_argument("--model", default="gpt-5.4",
                     help="LLM model name. Default gpt-5.4.")
    cap.add_argument("--detail", default="low", choices=("low", "high"),
                     help="OpenAI image detail level. Default low.")
    cap.add_argument("--max-caption-frames", type=int, default=24,
                     help="Hard cap on keyframes sent to the LLM. Default 24.")

    p_log = run.add_argument_group("logging")
    p_log.add_argument("--log-level", default="INFO",
                       choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    # ---- list-embedders -------------------------------------------------- #
    sub.add_parser("list-embedders", help="Print the registered embedder names.")

    # ---- web ------------------------------------------------------------- #
    web = sub.add_parser("web", help="Launch the Gradio web UI.")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=7860)
    web.add_argument("--share", action="store_true",
                     help="Expose via a Gradio public tunnel.")

    # ---- benchmark ------------------------------------------------------- #
    bench = sub.add_parser("benchmark",
                           help="Compare embedders on the same video. Writes a markdown report.")
    bench.add_argument("source", help="Video file path to use for benchmarking.")
    bench.add_argument("--embedders", nargs="+",
                       default=["yolov8n", "phash", "hsv"],
                       help="Embedder names to compare. Default: yolov8n phash hsv.")
    bench.add_argument("--output-root", default="outputs/benchmarks",
                       help="Directory for the benchmark report.")
    bench.add_argument("--no-caption", action="store_true",
                       help="Skip LLM captioning during benchmarks (default behaviour).")
    bench.add_argument("--sample-interval", type=float, default=1.0)
    bench.add_argument("--sim-threshold", type=float, default=0.96)
    bench.add_argument("--device", default="auto")

    # ---- caption-bench --------------------------------------------------- #
    capb = sub.add_parser(
        "caption-bench",
        help=("Score multiple LLM caption models on the same keyframe set. "
              "Writes outputs/caption-bench/caption_benchmark.md."),
    )
    capb.add_argument("source", help="Video file path.")
    capb.add_argument(
        "--models", nargs="+", required=True,
        help=("LLM model ids to benchmark. Example: --models gpt-5.4 "
              "google/gemini-2.5-flash anthropic/claude-3.5-haiku."),
    )
    capb.add_argument("--output-root", default="outputs/caption-bench",
                      help="Directory for the caption-bench report.")
    capb.add_argument("--embedder", default="yolov8n",
                      choices=list_embedders(),
                      help="Embedder for the keyframe pass. Default yolov8n.")
    capb.add_argument("--device", default="auto",
                      help="Torch device: auto | cpu | cuda | mps. Default auto.")
    capb.add_argument("--sample-interval", type=float, default=1.0)
    capb.add_argument("--sim-threshold", type=float, default=0.96)
    capb.add_argument("--detail", default="low", choices=("low", "high"),
                      help="Image detail forwarded to every model. Default low.")
    capb.add_argument("--max-keyframes", type=int, default=16,
                      help="Cap keyframes sent per model. Default 16.")

    return p


def _cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        sampler=SamplerConfig(interval_sec=args.sample_interval),
        segmenter=SegmenterConfig(
            sim_threshold=args.sim_threshold,
            use_ema=not args.no_ema,
            ema_alpha=args.ema_alpha,
            min_shot_sec=args.min_shot_sec,
        ),
        selector=SelectorConfig(
            max_frames_per_segment=args.max_per_segment,
            seconds_per_keyframe=args.seconds_per_keyframe,
            min_temporal_spacing_sec=args.min_spacing_sec,
            sharpness_weight=args.sharpness_weight,
            representativeness_weight=args.repr_weight,
        ),
        captioner=CaptionerConfig(
            enabled=not args.no_caption,
            model=args.model,
            detail=args.detail,
            max_keyframes=args.max_caption_frames,
        ),
        embedder=EmbedderConfig(name=args.embedder, device=args.device),
        output_root=Path(args.output_root),
        realtime_pacing=args.realtime,
    )
    return cfg


def _bench_cfg(args: argparse.Namespace, embedder_name: str) -> PipelineConfig:
    return PipelineConfig(
        sampler=SamplerConfig(interval_sec=args.sample_interval),
        segmenter=SegmenterConfig(sim_threshold=args.sim_threshold),
        captioner=CaptionerConfig(enabled=False),
        embedder=EmbedderConfig(name=embedder_name, device=args.device),
        output_root=Path(args.output_root) / embedder_name,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging("INFO" if args.command != "run" else getattr(args, "log_level", "INFO"))

    if args.command == "list-embedders":
        for name in list_embedders():
            print(name)
        return 0

    if args.command == "run":
        cfg = _cfg_from_args(args)
        result = run_pipeline(args.source, cfg)
        print()
        print(f"Run dir       : {result.run_dir}")
        print(f"Frames analyzed: {result.frames_analyzed}  (duration {result.duration_sec:.1f}s)")
        print(f"Segments       : {result.segments}")
        print(f"Keyframes      : {result.keyframes}")
        print(f"Embed latency  : mean {result.embed_latency_ms_avg:.2f}ms / "
              f"p95 {result.embed_latency_ms_p95:.2f}ms")
        print(f"Wall time      : {result.wall_time_sec:.2f}s")
        if result.caption_text:
            print()
            print("=== Caption ===")
            print(result.caption_text)
        return 0

    if args.command == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(args.source, args.embedders, _bench_cfg, args)
        return 0

    if args.command == "caption-bench":
        from .caption_bench import cli_run
        return cli_run(args)

    if args.command == "web":
        try:
            from .webapp import main as web_main
        except ModuleNotFoundError as exc:
            parser.error(
                f"web UI requires the optional 'gradio' dependency: {exc}.\n"
                "install it with: pip install 'keyframe[web]'  (or)  pip install gradio"
            )
        return web_main([f"--host={args.host}", f"--port={args.port}",
                         *(["--share"] if args.share else [])])

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
