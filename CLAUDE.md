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
- No `gh` CLI — use GitHub API via curl with token from `~/.git-credentials`
- No VPI 2.0 (only VPI 1.2.3 available, not compatible)
- No f-string `=`, no dataclasses, no walrus operator (Python 3.6)

## Git Workflow
- Default branch: `Production`
- All feature branches are based off `Development`
- PRs target `Development`, not `Production`
- Merge strategy: squash merge, delete feature branch after merge

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

Five threads: main (controller loop), vision-capture, context-summarizer, reasoning-worker, (future) serial.

## Phase 1: Vision System (complete)
- `vision/` package: camera capture, SSD-Mobilenet-v2 detection, IOU tracking
- Event bus: person_entered, person_left, object_appeared/disappeared/moved, scene_changed
- World state: tracked objects with bbox, confidence, duration
- Scene describer: human-readable text for LLM consumption
- CLI: `python3 -m vision`

## Phase 2: Controller + LLM + Context (complete)
- `controller/` package: orchestrator, swappable LLM backend, tiered context memory
- LLM backends: Claude API (raw HTTP) and OpenAI-compatible (OpenRouter, Aphrodite, vLLM, etc.)
- Context tiers: immediate (30s), short-term (5min, summarized), long-term (compressed)
- Reactive reasoning on person enter/leave + periodic reasoning every 30s
- CLI: `python3 -m controller --backend claude`

## Phase 3: Vision Enhancement, Terminal UI, Interactive Input, Memory (current)

### 3A: Vision Enhancement
- GoogleNet ROI classification on detected objects (1000 ImageNet classes)
- `cudaCrop()` each detection bbox, classify with `imageNet("googlenet")`
- `display_name` field on tracked objects: richer labels like "wine bottle" instead of "bottle"
- Configurable via `vision/config.py`: `CLASSIFICATION_ENABLED`, threshold, min bbox size

### 3B: Terminal UI (urwid)
- `controller/terminal_ui.py`: urwid-based live display
- Layout: scene panel (top), conversation log (middle, scrollable), input line (bottom), status bar
- Thread-safe updates via `urwid.watch_pipe()` — background threads write to pipe, urwid drains on main
- Controller runs in background thread (`start_background(ui=)`), urwid owns main thread
- `--headless` flag for log-only mode (backward compat)
- UI mode logs to `companion.log` instead of stderr

### 3C: Interactive Input
- Type messages in terminal UI, model responds using visual context
- `USER_REASONING_PROMPT` for conversational responses
- User messages and robot responses stored in context with importance scoring
- `controller.on_user_input()` → queue → controller thread → reasoning-worker thread (async)

### 3D: Memory Improvements
- Importance-weighted budget fitting: user messages (3) > person events (2) > object events (1) > scenes (0)
- Facts tier: persistent strings (max 20) that survive all summarization
- `queue_fact_extraction()`: non-blocking enqueue, LLM extraction runs on summarizer thread
- High-importance entries (>=2) skip summarization, preserved with original text in short-term
- `build_context()` includes `[Known facts]` section before `[Right now]`
- Token budget: facts 10%, immediate 45%, short-term 25%, long-term 20%

### 3E: Async Reasoning Worker
- All LLM reasoning calls (periodic, reactive, user) run on a dedicated `reasoning-worker` thread
- Main loop stays responsive at 2Hz — scene updates, input, status bar unblocked during LLM calls
- Request/result queues between main loop and worker (`queue.Queue`)
- Concurrency policy: user input always queued, periodic/reactive dropped when busy, stale requests (>10s) discarded
- "Thinking..." indicator in status bar while LLM call is in-flight
- Fact extraction offloaded to summarizer thread via `queue_fact_extraction()`

### 3F: Streaming LLM Responses
- `stream_complete()` generator on both Claude and OpenAI backends, parses SSE (server-sent events)
- Tokens stream to terminal UI in real-time via `stream_start`/`stream_chunk`/`stream_end` pipe messages
- `_streaming_widget`: urwid Text widget updated in-place as chunks arrive
- Reasoning worker selects streaming path when `STREAMING_ENABLED` and UI is attached
- Headless mode falls back to non-streaming `complete()`
- `_retry_loop_stream()`: retry variant that returns response with body unconsumed for streaming
- Configurable via `controller/config.py`: `STREAMING_ENABLED` (default True)

### CLI
- `python3 -m controller --backend openai` — UI mode (default)
- `python3 -m controller --headless --backend openai` — headless mode
- `python3 -m controller --backend openai --log-level DEBUG` — verbose logging

## Environment Variables
- `ANTHROPIC_API_KEY`: Claude API key (required for claude backend)
- `OPENAI_API_URL`: OpenAI-compatible endpoint (required for openai backend)
- `OPENAI_API_KEY`: API key for OpenAI-compatible endpoint
- `OPENAI_MODEL`: Model name for OpenAI-compatible endpoint
- `LLM_BACKEND`: Default backend selection (`claude` or `openai`)
- `SUMMARIZER_BACKEND`: Summarizer backend override (falls back to `LLM_BACKEND`)
- `SUMMARIZER_API_URL`: Summarizer endpoint override (falls back to `OPENAI_API_URL`)
- `SUMMARIZER_API_KEY`: Summarizer API key override (falls back to `OPENAI_API_KEY`)
- `SUMMARIZER_MODEL`: Summarizer model override (falls back to main model)

## Testing
- Unit tests: `python3 test_streaming.py` — mocked HTTP layer, no live API calls needed
- Tests cover: SSE parsing (Claude + OpenAI), retry logic, controller streaming flow, UI message dispatch
- Python 3.6 compatible — uses `unittest` + `unittest.mock`
