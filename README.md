# keyframe

Stream-friendly keyframe extraction and LLM video understanding.

One algorithm. One CLI. Same code path for offline files (`video.MOV`) and
online streams (webcam, RTSP). Frame-by-frame from the very first line, so
nothing about the offline run depends on the file being seekable.

```
video in  ->  embed  ->  segment  ->  select  ->  caption out
                |          |           |           |
                YOLOv8 /   shot       sharpness   gpt-5.4
                phash /    detection  + repr      via OpenAI
                hsv        with EMA   + spacing   compatible API
```

## Quick start

```bash
# install editable
python3 -m pip install -e .

# put OPENAI_API_KEY + OPENAI_BASE_URL into .env (already present)

# offline: a video file in, an explanation out
python3 -m keyframe run video/video.MOV

# online: webcam stream, keyframes emitted live to stream_events.jsonl
python3 -m keyframe run 0

# RTSP camera
python3 -m keyframe run rtsp://192.168.1.10/stream1 --sim-threshold 0.92

# benchmark embedders against the same source
python3 -m keyframe benchmark video/video.MOV --embedders yolov8n phash hsv
```

The `run` command writes everything for one source into
`outputs/<stem>_<timestamp>/`. Each run is self-contained, never overwrites a
previous run, and ships a `README.md` plus `summary.json` you can read
without poking around the rest of the tree.

## Why this design

A single video can have tens of thousands of frames. Sending all of them to a
multimodal LLM is wasteful and slow. Sending the first or middle frame from
each second is what we did in v1 and the output suffered from motion blur and
near-duplicate frames.

The fix is a four-stage funnel:

1. **Sample** the stream at a fixed wall-clock interval (default 1 frame/sec).
   This drops ~98% of frames before any expensive model runs.
2. **Embed** each sampled frame with a small visual model. The output is a
   1-D vector. Cheap embedders (`phash`, `hsv`) run on CPU; YOLOv8 variants
   run on MPS/CUDA when available.
3. **Segment** the vectors online with shot detection: maintain an EMA of the
   current shot's embedding, fork a new shot when cosine similarity drops
   below a threshold. EMA prevents drift on long slow shots; a
   `min-shot-sec` guard suppresses sub-second flicker (camera shake, focus
   hunt).
4. **Select** the best frames inside each segment. Each frame is scored as a
   weighted z-score of (Laplacian sharpness, cosine similarity to the
   segment centroid). The top frames are picked greedily under a minimum
   temporal spacing constraint, so we never return three near-identical
   frames from the same instant. Long segments get more keyframes
   (1 per `--seconds-per-keyframe` seconds, capped at `--max-per-segment`).
5. **Caption** all chosen keyframes in a single multi-image chat completion
   request. The system prompt forbids guessing locations or other unstated
   facts.

The same algorithm runs offline and online because the only piece that
distinguishes the two cases is the `VideoSource` iterator. The pipeline only
ever calls `for frame in source:` and decides one frame at a time.

## Repository layout

```
test/
├── pyproject.toml                  installable package + entrypoint
├── requirements.txt                pinned-ish runtime deps
├── README.md                       this file
├── .env                            OPENAI_API_KEY / OPENAI_BASE_URL
├── video/                          input videos
├── outputs/                        one folder per run (gitignored)
├── archive/v1/                     old experimental scripts (kept for reference)
└── src/keyframe/
    ├── __main__.py                 `python3 -m keyframe`
    ├── cli.py                      argparse, sub-commands
    ├── pipeline.py                 orchestrator
    ├── source.py                   VideoSource: file or webcam or rtsp
    ├── embedders.py                yolov8n/s/m, phash, hsv (plugin registry)
    ├── segmenter.py                online EMA shot detector with anti-flicker
    ├── selector.py                 sharpness + representativeness + spacing
    ├── captioner.py                multi-image chat.completions caller
    ├── visualize.py                timeline, similarity curve, grid (cv2)
    ├── benchmark.py                cross-embedder report
    ├── config.py                   dataclass configs, env loading
    ├── io_utils.py                 unicode-safe I/O, sharpness, cosine
    └── logging_setup.py            rich logger
```

## Output of a single run

`outputs/<source_stem>_<YYYYMMDD_HHMMSS>/`

| Path | Purpose |
| --- | --- |
| `README.md` | Human-readable run summary (segments + keyframes table + caption) |
| `summary.json` | Machine-readable counts and latencies |
| `frames/` | Every analyzed frame, cached as JPEG |
| `keyframes/kf_NNN_segMM_tNNNN.NNs.jpg` | Selected keyframes |
| `keyframes/keyframes.csv` | id, segment, time, sharpness, repr, score |
| `viz/timeline.jpg` | Segment-coloured strip with keyframe markers |
| `viz/similarity_curve.jpg` | Sim vs running shot mean over time |
| `viz/keyframes_grid.jpg` | Contact sheet of every keyframe |
| `caption/caption.md` | LLM narration of the entire video |
| `caption/raw_response.json` | Full LLM response (for debugging) |
| `logs/stream_events.jsonl` | Per-frame decisions, written live |
| `logs/analyzed_frames.csv` | Analyzed frames as CSV |
| `logs/segments.csv` | Segment boundaries as CSV |
| `logs/config.json` | Exact pipeline config used |

## Embedders

| Name | Backend | Dim | Latency (MPS, 2160x3840 frame) |
| --- | --- | --- | --- |
| `yolov8n` | YOLOv8 nano backbone | 256 | ~24 ms |
| `yolov8s` | YOLOv8 small backbone | 512 | ~40 ms |
| `yolov8m` | YOLOv8 medium backbone | 576 | ~70 ms |
| `phash` | 8x8 DCT perceptual hash | 64 | <1 ms (CPU) |
| `hsv` | 3-channel HSV histogram | 96 | <1 ms (CPU) |

YOLOv8 weights are downloaded on first use by `ultralytics` and live in the
project root. `phash` and `hsv` are zero-dependency baselines that exist
specifically so the benchmark mode can answer "is the heavy model worth it?"

Run `python3 -m keyframe list-embedders` to see what is currently registered.

## Benchmark mode

```bash
python3 -m keyframe benchmark video/video.MOV \
    --embedders yolov8n yolov8s phash hsv \
    --sample-interval 1.0 --sim-threshold 0.94
```

Output: `outputs/benchmarks/benchmark.md` plus per-embedder run dirs. Each row
reports embed latency, segment count, keyframe count, and the Jaccard
agreement (with +/-1.5s tolerance) between the embedder's keyframe times and
the reference embedder's.

## Tuning cheat sheet

| Symptom | Knob | Direction |
| --- | --- | --- |
| Too few keyframes, missing scene changes | `--sim-threshold` | up (e.g. 0.96) |
| Too many keyframes, lots of near-duplicates | `--sim-threshold` | down (e.g. 0.92) |
| Keyframes appear during transient flicker | `--min-shot-sec` | up (3-4s) |
| Keyframes too blurry | `--sharpness-weight` | up (0.6+) |
| Keyframes look like outliers in their segment | `--repr-weight` | up (0.7+) |
| Long segments only emit one frame | `--seconds-per-keyframe` | down (4-6s) |
| LLM cost too high | `--max-caption-frames` | down (8-12) |

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
secondary model) can tail that file or be wired directly into the
`StreamingSegmenter.ingest` call.

## Limitations

- The captioner sends a single chat completion request. For very long videos
  (hundreds of keyframes) you should split into a windowed map-reduce. Not
  implemented yet.
- The frame cache writes every sampled frame to disk at full resolution. For
  multi-hour streams this will grow without bound. Add a ring-buffer policy
  for true 24/7 operation.
- YOLOv8 backbone embeddings are a side-effect of a detector, not a
  general-purpose visual encoder. They cluster scenes by gross layout, not
  by semantic content. For finer-grained understanding swap in CLIP via a
  new `Embedder` subclass.
