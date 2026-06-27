# keyframe

Stream-friendly keyframe extraction and LLM video understanding.

`keyframe` turns a video, a webcam, or an RTSP stream into a short, ordered set
of representative frames and a natural-language description of what happens
across the clip. It uses one algorithm and one CLI for both offline files and
live streams: the only thing that changes between the two is the iterator that
yields frames.

```
   video in  →  sample  →  embed  →  segment  →  select  →  caption out
                  │          │          │          │           │
              1 frame/s    YOLO /     online      sharpness   chat.completions
              by default   phash /    EMA shot    + repr      multi-image
                           hsv /      detection   + spacing   request
                           SAM        + anti-     + diversity
                                      flicker     filter
```

## Why it works this way

A single video can have tens of thousands of frames. Sending all of them to a
multimodal LLM is wasteful and slow. Sending one frame per second is what the
v1 prototype did, and the output suffered from motion blur and near-duplicate
frames.

The fix is a four-stage funnel that runs identically on a file or a live
stream:

1.  **Sample** the input at a fixed wall-clock interval (default 1 frame/sec).
    File sources seek directly to each sample point through ffmpeg or
    `cv2.CAP_PROP_POS_FRAMES`, so the decoder never wastes work on frames the
    pipeline is going to throw away.
2.  **Embed** each sampled frame into a vector. Cheap baselines (`phash`,
    `hsv`) cost a millisecond on CPU. Neural backbones (YOLOv8 / YOLO11 /
    YOLO26 / MobileSAM) run on MPS or CUDA when available.
3.  **Segment** the embedding stream online. The shot mean is an
    exponential moving average; a new shot opens when the next frame's cosine
    similarity falls below `--sim-threshold`. A `--min-shot-sec` guard
    prevents sub-second flicker (auto-focus, hand shake) from producing
    throwaway segments.
4.  **Select** the best frames inside each segment. Each candidate is scored
    by a weighted z-score of (Laplacian sharpness, similarity to the segment
    centroid). The top frames are picked greedily under a minimum temporal
    spacing. Long segments get more keyframes (one per
    `--seconds-per-keyframe`, capped at `--max-per-segment`). A
    cross-segment **diversity filter** drops any keyframe that is nearly
    identical to one already kept, with a `min-keyframes` floor for short
    videos.
5.  **Caption** every selected keyframe in a single multi-image chat
    completion request. The system prompt forbids guessing locations, brands,
    or motivations that are not visible.

Because the only piece that distinguishes offline and online cases is the
`VideoSource` iterator, the pipeline only ever calls `for frame in source:`
and decides one frame at a time. The same code path that captions a recorded
clip also tails a live RTSP stream and writes keyframes to disk as they
happen.

## Install

Requires Python 3.10+, optionally `ffmpeg` on the `$PATH` (for the fast
decode path).

```bash
# core install
python3 -m pip install -e .

# add the optional Gradio web UI
python3 -m pip install -e '.[web]'

# add the dev tooling (pytest, ruff)
python3 -m pip install -e '.[dev]'
```

Copy `.env.example` to `.env` and fill in your `OPENAI_API_KEY` (and
`OPENAI_BASE_URL` if you are using a proxy such as OpenRouter). The caption
stage is optional — pass `--no-caption` to skip the LLM call entirely.

## Quick start

```bash
# offline: a video file in, an explanation out
python3 -m keyframe run video/video.MOV

# online: webcam stream, keyframes emitted live to logs/stream_events.jsonl
python3 -m keyframe run 0

# RTSP camera, lower threshold so subtle scene changes register
python3 -m keyframe run rtsp://192.168.1.10/stream1 --sim-threshold 0.92

# replay a file at native fps to prove the pipeline keeps up in real time
python3 -m keyframe run video/video.MOV --realtime

# skip captioning when you only want keyframes + visualisations
python3 -m keyframe run video/video.MOV --no-caption

# launch the Gradio web UI
python3 -m keyframe web --port 7860

# benchmark several embedders on the same clip and write a comparison report
python3 -m keyframe benchmark video/video.MOV --embedders yolov8n phash hsv
```

`run` writes everything for one source into
`outputs/<source_stem>_<YYYYMMDD_HHMMSS>/`. Each run is self-contained, never
overwrites a previous run, and ships a `README.md` plus `summary.json` you can
read without exploring the rest of the tree.

## Python API

```python
from keyframe import PipelineConfig, run_pipeline

result = run_pipeline("video/video.MOV", PipelineConfig())
print(result.run_dir, result.keyframes, result.caption_text)
```

For UIs and notebooks that want live updates, iterate the pipeline directly:

```python
from keyframe import iter_pipeline, PipelineConfig

for event in iter_pipeline("video/video.MOV", PipelineConfig()):
    if event.stage == "frame":
        print(event.message)
    if event.final_result is not None:
        print("done:", event.final_result.run_dir)
```

`iter_pipeline` is a generator of `ProgressEvent` objects. It is what the CLI
and the Gradio app consume internally.

## Repository layout

```
.
├── pyproject.toml                 installable package + entry point
├── requirements.txt               runtime deps (plain pip)
├── README.md                      this file
├── .env.example                   template for OPENAI_API_KEY / OPENAI_BASE_URL
├── video/                         input videos (gitignored)
├── outputs/                       one folder per run (gitignored)
├── tests/                         pytest suite (unit + smoke tests)
└── src/keyframe/
    ├── __main__.py                python3 -m keyframe
    ├── cli.py                     argparse, sub-commands
    ├── pipeline.py                streaming orchestrator + ProgressEvent
    ├── source.py                  VideoSource: file, webcam, or RTSP via cv2
    ├── ffmpeg_source.py           hardware-accelerated decode for files
    ├── embedders.py               YOLO, MobileSAM, phash, hsv (plugin registry)
    ├── segmenter.py               online EMA shot detector with anti-flicker
    ├── selector.py                per-segment + cross-segment keyframe picker
    ├── captioner.py               multi-image chat.completions caller
    ├── visualize.py               timeline, similarity curve, grid, film strip
    ├── benchmark.py               cross-embedder report
    ├── webapp.py                  Gradio UI with live progress
    ├── config.py                  dataclass configs, dotenv loader
    ├── io_utils.py                unicode-safe image IO + math helpers
    └── logging_setup.py           rich logger
```

## Output of a single run

`outputs/<source_stem>_<YYYYMMDD_HHMMSS>/`

| Path | Purpose |
| --- | --- |
| `README.md` | Human-readable run summary (segments + keyframes table + caption). |
| `summary.json` | Machine-readable counts and latencies. |
| `frames/` | Every analyzed frame, cached as JPEG at `cache_thumb_width`. |
| `keyframes/kf_NNN_segMM_tNNNN.NNs.jpg` | Selected keyframes. |
| `keyframes/keyframes.csv` | id, segment, time, sharpness, repr, composite score. |
| `viz/timeline.jpg` | Segment-coloured strip with keyframe markers. |
| `viz/similarity_curve.jpg` | Per-frame cos-sim vs the running shot mean. |
| `viz/keyframes_grid.jpg` | Contact sheet of every keyframe with metadata banners. |
| `viz/film_strip.jpg` | Hero film strip: every analyzed frame in order. |
| `caption/caption.md` | LLM narration of the entire video. |
| `caption/raw_response.json` | Full LLM response (for debugging). |
| `logs/stream_events.jsonl` | Per-frame decisions, written live. |
| `logs/analyzed_frames.csv` | Analyzed frames as CSV. |
| `logs/segments.csv` | Segment boundaries as CSV. |
| `logs/config.json` | Exact pipeline config used. |

## Embedders

| Name | Backbone | Dim | Notes |
| --- | --- | ---: | --- |
| `yolov8n` | YOLOv8 nano detector | 256 | Default. Fast on MPS / CUDA. |
| `yolov8s` | YOLOv8 small detector | 512 | Better separation, ~2× slower than nano. |
| `yolov8m` | YOLOv8 medium detector | 576 | Strongest YOLOv8 backbone shipped here. |
| `yolov8n-seg` | YOLOv8 nano segmentation backbone | 256 | Drop-in if you also need masks downstream. |
| `yolo11n` / `yolo11s` | YOLO11 detector | 256 / 512 | C3K2 + C2PSA blocks; same calling convention. |
| `yolo26n` | YOLO26 NMS-free detector | 256 | DFL-free regression, MuSGD optimiser. |
| `mobile_sam` | MobileSAM TinyViT image encoder | 256 | Dense feature map, global-pooled to 256-d. |
| `phash` | 8x8 DCT perceptual hash | 64 | <1 ms on CPU. Zero-dependency baseline. |
| `hsv` | 3-channel HSV histogram | 96 | <1 ms on CPU. Colour-only signature. |

Neural weights are downloaded on first use by `ultralytics` and cached in the
project root. The two CPU baselines (`phash`, `hsv`) exist so the benchmark
mode can answer "is the heavy model worth it?".

```bash
python3 -m keyframe list-embedders
```

## Benchmark mode

```bash
python3 -m keyframe benchmark video/video.MOV \
    --embedders yolov8n yolov8s phash hsv \
    --sample-interval 1.0 --sim-threshold 0.94
```

Writes `outputs/benchmarks/benchmark.md` plus per-embedder run folders. The
report reports embed latency, segment count, keyframe count, and the Jaccard
agreement (with ±1.5 s tolerance) between each embedder's keyframe times and
the reference embedder's. The first embedder in the list is the reference, so
it always reports 1.0.

## Tuning cheat sheet

| Symptom | Knob | Direction |
| --- | --- | --- |
| Too few keyframes, missing scene changes | `--sim-threshold` | up (e.g. 0.97) |
| Too many keyframes, lots of near-duplicates | `--sim-threshold` | down (e.g. 0.92) |
| Keyframes appear during transient flicker | `--min-shot-sec` | up (3–4 s) |
| Keyframes look motion-blurred | `--sharpness-weight` | up (0.6+) |
| Keyframes look like outliers in their segment | `--repr-weight` | up (0.7+) |
| Long segments only emit one frame | `--seconds-per-keyframe` | down (4–6 s) |
| LLM cost too high | `--max-caption-frames` | down (8–12) |

## Live / online mode

```bash
# webcam, save keyframes as they happen, no LLM
python3 -m keyframe run 0 --no-caption --sim-threshold 0.92

# replay a file at native fps so you can prove the pipeline keeps up
python3 -m keyframe run video/video.MOV --realtime
```

The pipeline emits one line of `logs/stream_events.jsonl` per analyzed frame,
in real time. Each `"event": "frame"` line carries the timestamp, similarity,
and `is_shot_start` flag. Downstream systems (queue producer, webhook,
secondary model) can tail that file or be wired directly into
`StreamingSegmenter.ingest`.

## Web UI

```bash
python3 -m keyframe web --port 7860
```

The Gradio app has two tabs:

- **Run** — upload a video, watch the film strip build as each frame is
  analyzed, see the keyframes gallery fill in, and read the final LLM caption.
  Streams `ProgressEvent`s straight from `iter_pipeline`.
- **Compare models** — run the same clip through several embedders and stack
  their film strips. A good embedder shows clear segment colour blocks
  aligned with real shot cuts; a bad one shows random colour noise.

## Development

```bash
# run the unit tests (selector, segmenter, config, embedders, CLI, IO)
python3 -m pytest tests -q

# lint (optional, requires the [dev] extra)
ruff check src tests
```

The pytest suite covers the deterministic core (segmenter, selector, config,
CPU embedders, IO helpers, CLI parser). It does not download neural weights
or hit the network, so it runs in well under a second on a laptop.

## Limitations

- The captioner sends a single chat completion request. For very long videos
  (hundreds of keyframes) you should split into a windowed map-reduce. Not
  implemented yet.
- The frame cache writes every sampled frame to disk at `cache_thumb_width`.
  For multi-hour streams this will grow without bound. Add a ring-buffer
  policy for true 24/7 operation.
- YOLO backbone embeddings are a side-effect of a detector, not a
  general-purpose visual encoder. They cluster scenes by gross layout, not
  by semantic content. For finer-grained understanding, register a new
  `Embedder` subclass that wraps CLIP or SigLIP.
