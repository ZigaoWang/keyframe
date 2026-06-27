"""Cross-embedder benchmark.

Runs the full pipeline (minus LLM captioning) once per requested embedder on
the same source video, then writes a side-by-side markdown report. Captures:

  * embed latency mean / p50 / p95
  * total wall time
  * segments detected
  * keyframes selected
  * keyframe timestamps overlap with the reference embedder (Jaccard at +/- 1s tolerance)

The first embedder in the list is the reference. Useful for asking "is the
cheap baseline (phash) anywhere near the YOLO embedder for shot detection?".
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import PipelineConfig
from .logging_setup import get_logger
from .pipeline import PipelineResult, run_pipeline

log = get_logger("benchmark")


@dataclass
class BenchmarkRow:
    embedder: str
    frames_analyzed: int
    segments: int
    keyframes: int
    embed_latency_ms_avg: float
    embed_latency_ms_p95: float
    wall_time_sec: float
    timestamps_jaccard_vs_ref: float
    run_dir: str


def _load_keyframe_timestamps(run_dir: Path) -> list[float]:
    summary = run_dir / "summary.json"
    if not summary.exists():
        return []
    kf_csv = run_dir / "keyframes" / "keyframes.csv"
    if not kf_csv.exists():
        return []
    import csv
    out: list[float] = []
    with kf_csv.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append(float(row["timestamp_sec"]))
            except (KeyError, ValueError):
                pass
    return sorted(out)


def _jaccard_with_tolerance(a: list[float], b: list[float], tolerance_sec: float) -> float:
    """Treat two timestamps as matching if they are within ``tolerance_sec``.
    Greedy bipartite match for simplicity (good enough for typical sizes)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    used_b = set()
    matches = 0
    for t in a:
        best = -1
        best_dist = tolerance_sec + 1e-9
        for j, u in enumerate(b):
            if j in used_b:
                continue
            d = abs(u - t)
            if d <= best_dist:
                best_dist = d
                best = j
        if best >= 0:
            used_b.add(best)
            matches += 1
    union = len(a) + len(b) - matches
    return matches / union if union > 0 else 1.0


def run_benchmark(
    source: str,
    embedders: list[str],
    cfg_factory: Callable[[argparse.Namespace, str], PipelineConfig],
    args: argparse.Namespace,
) -> None:
    if not embedders:
        log.error("no embedders requested")
        return

    bench_root = Path(args.output_root)
    bench_root.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, PipelineResult]] = []
    for name in embedders:
        log.info("[bench] running embedder=%s", name)
        cfg = cfg_factory(args, name)
        result = run_pipeline(source, cfg)
        results.append((name, result))

    ref_name, ref_result = results[0]
    ref_ts = _load_keyframe_timestamps(ref_result.run_dir)

    rows: list[BenchmarkRow] = []
    for name, result in results:
        if name == ref_name:
            jaccard = 1.0
        else:
            other_ts = _load_keyframe_timestamps(result.run_dir)
            jaccard = _jaccard_with_tolerance(ref_ts, other_ts, tolerance_sec=1.5)
        rows.append(BenchmarkRow(
            embedder=name,
            frames_analyzed=result.frames_analyzed,
            segments=result.segments,
            keyframes=result.keyframes,
            embed_latency_ms_avg=result.embed_latency_ms_avg,
            embed_latency_ms_p95=result.embed_latency_ms_p95,
            wall_time_sec=result.wall_time_sec,
            timestamps_jaccard_vs_ref=round(jaccard, 3),
            run_dir=str(result.run_dir),
        ))

    md: list[str] = [
        "# Embedder benchmark",
        "",
        f"- Source: `{source}`",
        f"- Reference embedder: `{ref_name}`",
        f"- Jaccard tolerance: 1.5s",
        "",
        "| Embedder | Frames | Segments | Keyframes | Embed mean (ms) | Embed p95 (ms) | Wall (s) | Jaccard vs ref |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| `{row.embedder}` | {row.frames_analyzed} | {row.segments} | "
            f"{row.keyframes} | {row.embed_latency_ms_avg:.2f} | "
            f"{row.embed_latency_ms_p95:.2f} | {row.wall_time_sec:.2f} | "
            f"{row.timestamps_jaccard_vs_ref:.3f} |"
        )

    md += [
        "",
        "## What the columns mean",
        "",
        "- **Embed mean / p95 (ms)** measure pure feature extraction latency per frame.",
        "- **Wall (s)** is the full pipeline cost (decode + sample + embed + segment + select + viz).",
        "- **Jaccard vs ref** is the agreement of keyframe timestamps with the first embedder, "
        "with a +/- 1.5s matching tolerance. 1.0 = identical, 0.0 = disjoint.",
        "- The first embedder is the reference, so it always reports 1.0.",
        "",
        "## Per-run output directories",
        "",
    ]
    for row in rows:
        md.append(f"- `{row.embedder}` -> `{row.run_dir}`")

    report_path = bench_root / "benchmark.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    log.info("[bench] report written: %s", report_path)

    json_path = bench_root / "benchmark.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== Benchmark summary ===")
    for row in rows:
        print(
            f"  {row.embedder:10s}  segs={row.segments:3d}  kfs={row.keyframes:3d}  "
            f"embed_mean={row.embed_latency_ms_avg:6.2f}ms  wall={row.wall_time_sec:6.2f}s  "
            f"jaccard={row.timestamps_jaccard_vs_ref:.3f}"
        )
    print(f"\nReport: {report_path}")
