# LLM Robot Companion

## Hardware
- Jetson Nano (Tegra X1 / Maxwell GPU, SM_53)
- JetPack 4.6.6 (R32.7.6), Ubuntu 18.04, Kernel 4.9.337-tegra
- 4GB RAM, 6GB swap, ~851GB disk
- Camera: CSI IMX219 (Pi Camera v2) at `csi://0` (`/dev/video0` via Tegra VI driver)

## Software Constraints
- Python 3.6.9 (aliased as `python`)
- CUDA 10.2, TensorRT 8.2.1, OpenCV 4.1.1
- NumPy 1.13.3
- No sudo access for the claude session
- No VPI 2.0 (only VPI 1.2.3 available, not compatible)
- No f-string `=`, no dataclasses, no walrus operator (Python 3.6)

## Build (jetson-inference)
- Built from source with Python 3.6 bindings enabled
- Fixed `npymath` linker issue in `utils/python/bindings/CMakeLists.txt` (used full path to `libnpymath.a` instead of bare `-lnpymath` which didn't propagate library search paths to dependent targets)
- Python scripts must run from `build/aarch64/bin/` so `networks/` symlink resolves for model loading
- Environment variables set in `~/.bashrc`:
  - `PATH` includes `/usr/local/cuda-10.2/bin`
  - `LD_LIBRARY_PATH` includes `build/aarch64/lib`
  - `PYTHONPATH` includes Python binding and package paths

## Models Downloaded
- **SSD-Mobilenet-v2**: Object detection, COCO 91 classes, FP16 engine cached
- **GoogleNet**: Image classification, ImageNet 1000 classes

## COCO Detection Limitations
SSD-Mobilenet-v2 only knows 91 COCO classes (person, car, dog, bottle, etc). No game controllers, cables, electronics components, tools, etc. Will confidently mislabel unfamiliar objects as the nearest COCO class.

## Architecture

```
Controller (Nano)  --->  Vision (local, jetson-inference)
    |
    +--->  LLM Backend (swappable: Claude API or OpenAI-compatible)
    |
    +--->  Context Manager (tiered memory + async summarization)
    |
    +--->  (Future) Arduino serial for hardware control
```

Four threads: main (controller loop), vision-capture, context-summarizer, (future) serial.

## Phase 1: Vision System (complete)
- `vision/` package: camera capture, SSD-Mobilenet-v2 detection, IOU tracking
- Event bus: person_entered, person_left, object_appeared/disappeared/moved, scene_changed
- World state: tracked objects with bbox, confidence, duration
- Scene describer: human-readable text for LLM consumption
- CLI: `python3 -m vision`

## Phase 2: Controller + LLM + Context (current)
- `controller/` package: orchestrator, swappable LLM backend, tiered context memory
- LLM backends: Claude API (raw HTTP) and OpenAI-compatible (OpenRouter, Aphrodite, vLLM, etc.)
- Context tiers: immediate (30s), short-term (5min, summarized), long-term (compressed)
- Reactive reasoning on person enter/leave + periodic reasoning every 30s
- CLI: `python3 -m controller --backend claude`

## Environment Variables
- `ANTHROPIC_API_KEY`: Claude API key (required for claude backend)
- `OPENAI_API_URL`: OpenAI-compatible endpoint (required for openai backend)
- `OPENAI_API_KEY`: API key for OpenAI-compatible endpoint
- `OPENAI_MODEL`: Model name for OpenAI-compatible endpoint
- `LLM_BACKEND`: Default backend selection (`claude` or `openai`)
