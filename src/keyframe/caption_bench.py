"""Captioner benchmark.

Runs the keyframe selection pipeline once on a given video (no LLM), then
sends the same set of keyframes through every requested caption model and
writes a side-by-side markdown report comparing latency, output length, and
text. Useful for picking which model to default to in
:class:`CaptionerConfig.model` / ``.segment_model``.

The benchmark deliberately reuses a single keyframe set so every model is
graded on identical input. The pipeline cost (decode + embed + segment +
select) is measured once and reported as overhead.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .captioner import caption_keyframes
from .config import CaptionerConfig, PipelineConfig, load_dotenv
from .logging_setup import get_logger
from .pipeline import run_pipeline

log = get_logger("caption_bench")


@dataclass
class CaptionBenchRow:
    model: str
    frames_sent: int
    latency_sec: float
    char_count: int
    word_count: int
    text: str
    error: str = ""


def _shorten(text: str, limit: int = 220) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_caption_benchmark(
    source: str,
    models: list[str],
    pipeline_cfg: PipelineConfig,
    output_root: Path,
    detail: str = "low",
    max_keyframes: int = 16,
) -> Path:
    """Run pipeline once, then caption with each model. Writes a markdown report.

    Returns the path of the report.
    """
    load_dotenv()
    if not models:
        raise ValueError("caption-bench requires at least one model id")

    output_root.mkdir(parents=True, exist_ok=True)

    # Run pipeline once with captioning disabled so we have a stable keyframe set.
    bench_cfg = PipelineConfig(
        sampler=pipeline_cfg.sampler,
        segmenter=pipeline_cfg.segmenter,
        selector=pipeline_cfg.selector,
        embedder=pipeline_cfg.embedder,
        captioner=CaptionerConfig(enabled=False),
        output_root=output_root / "_pipeline",
        embed_max_width=pipeline_cfg.embed_max_width,
        cache_thumb_width=pipeline_cfg.cache_thumb_width,
        batch_size=pipeline_cfg.batch_size,
        use_ffmpeg=pipeline_cfg.use_ffmpeg,
        ffmpeg_hwaccel=pipeline_cfg.ffmpeg_hwaccel,
    )
    t0 = time.perf_counter()
    pipeline_result = run_pipeline(source, bench_cfg)
    pipeline_wall = time.perf_counter() - t0
    log.info("[caption-bench] pipeline produced %d keyframes in %.2fs",
             pipeline_result.keyframes, pipeline_wall)

    # Re-read the keyframes off disk via the pipeline's run dir; we need the
    # Keyframe dataclasses with their bgr_paths for caption_keyframes.
    from .selector import Keyframe
    import csv

    kf_csv = pipeline_result.run_dir / "keyframes" / "keyframes.csv"
    keyframes: list[Keyframe] = []
    with kf_csv.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            keyframes.append(Keyframe(
                keyframe_id=int(row["keyframe_id"]),
                segment_id=int(row["segment_id"]),
                source_index=0,
                timestamp_sec=float(row["timestamp_sec"]),
                bgr_path=row["saved_path"],
                sharpness=float(row["sharpness"]),
                representativeness=float(row["representativeness"]),
                composite_score=float(row["composite_score"]),
                rank_in_segment=int(row["rank_in_segment"]),
                embedding=None,
            ))
    duration_sec = pipeline_result.duration_sec

    rows: list[CaptionBenchRow] = []
    for model in models:
        cap_cfg = CaptionerConfig(
            enabled=True, model=model, detail=detail,
            max_keyframes=max_keyframes, fallback_models=(),
        )
        try:
            res = caption_keyframes(keyframes, duration_sec, cap_cfg)
            rows.append(CaptionBenchRow(
                model=model,
                frames_sent=res.frames_sent,
                latency_sec=round(res.latency_sec, 2),
                char_count=len(res.text),
                word_count=len(res.text.split()),
                text=res.text,
            ))
            log.info("[caption-bench] %s: %d chars, %.2fs",
                     model, len(res.text), res.latency_sec)
        except Exception as exc:  # noqa: BLE001
            rows.append(CaptionBenchRow(
                model=model, frames_sent=len(keyframes), latency_sec=0.0,
                char_count=0, word_count=0, text="",
                error=f"{type(exc).__name__}: {exc}",
            ))
            log.warning("[caption-bench] %s failed: %s", model, exc)

    report_path = output_root / "caption_benchmark.md"
    md: list[str] = [
        "# Captioner benchmark",
        "",
        f"- Source: `{source}`",
        f"- Embedder: `{pipeline_result.embedder}`",
        f"- Keyframes used: **{pipeline_result.keyframes}**",
        f"- Pipeline overhead (one-time): {pipeline_wall:.2f}s",
        f"- Image detail: `{detail}`",
        "",
        "| Model | Frames | Latency (s) | Chars | Words | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        status = "ok" if not row.error else f"`{row.error[:60]}`"
        md.append(
            f"| `{row.model}` | {row.frames_sent} | {row.latency_sec:.2f} "
            f"| {row.char_count} | {row.word_count} | {status} |"
        )
    md += ["", "## Captions", ""]
    for row in rows:
        md += [f"### `{row.model}`"]
        if row.error:
            md += ["", f"_failed: {row.error}_", ""]
            continue
        md += ["", row.text, ""]
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    json_path = output_root / "caption_benchmark.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== Caption benchmark summary ===")
    for row in rows:
        flag = "FAIL" if row.error else "ok"
        print(
            f"  {row.model:40s}  {row.latency_sec:6.2f}s  "
            f"{row.char_count:5d} chars  {row.word_count:4d} words  "
            f"[{flag}] {_shorten(row.text or row.error, 80)}"
        )
    print(f"\nReport: {report_path}")
    return report_path


def cli_run(args: argparse.Namespace) -> int:
    """argparse glue: build configs from CLI args, invoke the benchmark."""
    from .config import EmbedderConfig, SamplerConfig, SegmenterConfig, SelectorConfig

    pipeline_cfg = PipelineConfig(
        sampler=SamplerConfig(interval_sec=args.sample_interval),
        segmenter=SegmenterConfig(sim_threshold=args.sim_threshold),
        selector=SelectorConfig(),
        embedder=EmbedderConfig(name=args.embedder, device=args.device),
        captioner=CaptionerConfig(enabled=False),
    )
    run_caption_benchmark(
        source=args.source,
        models=list(args.models),
        pipeline_cfg=pipeline_cfg,
        output_root=Path(args.output_root),
        detail=args.detail,
        max_keyframes=args.max_keyframes,
    )
    return 0
