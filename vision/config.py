# Camera
CAMERA_SOURCE = "csi://0"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection
DETECTION_NETWORK = "ssd-mobilenet-v2"
DETECTION_THRESHOLD = 0.5
DETECTION_OVERLAY = "none"

# Tracking (jetson-inference built-in IOU tracker)
TRACKING_ENABLED = True
TRACKER_TYPE = "IOU"
TRACKER_MIN_FRAMES = 3
TRACKER_DROP_FRAMES = 15
TRACKER_OVERLAP_THRESHOLD = 0.5

# Classification (GoogleNet ROI refinement on detected objects)
CLASSIFICATION_ENABLED = True
CLASSIFICATION_NETWORK = "googlenet"
CLASSIFICATION_THRESHOLD = 0.15
CLASSIFICATION_MIN_BBOX_SIZE = 30  # pixels; skip tiny detections

# Frame loop
TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS

# World state
STALE_OBJECT_TIMEOUT = 5.0
MOVEMENT_THRESHOLD = 50.0
PERSON_CLASS_ID = 1

# Events
MAX_EVENT_HISTORY = 200
SCENE_CHANGE_COOLDOWN = 2.0

# Model path resolution — detectNet resolves models via a relative
# "networks/" symlink that only exists in the build output directory.
# Set JETSON_INFERENCE_DIR env var to override, or defaults to ~/jetson-inference.
import os
import sys

JETSON_INFERENCE_DIR = os.environ.get(
    "JETSON_INFERENCE_DIR",
    os.path.expanduser("~/jetson-inference")
)
MODEL_RESOLVE_DIR = os.path.join(JETSON_INFERENCE_DIR, "build", "aarch64", "bin")

# Add jetson-inference Python paths to sys.path so the deferred
# `import jetson_inference` works without special PYTHONPATH setup.
_JETSON_PYTHON_PATHS = [
    os.path.join(JETSON_INFERENCE_DIR, "build", "aarch64", "lib", "python", "3.6"),
    os.path.join(JETSON_INFERENCE_DIR, "python", "python"),
    os.path.join(JETSON_INFERENCE_DIR, "utils", "python", "python"),
]
for _p in _JETSON_PYTHON_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Ensure native shared libraries are findable at runtime.
_JETSON_LIB_DIR = os.path.join(JETSON_INFERENCE_DIR, "build", "aarch64", "lib")
if os.path.isdir(_JETSON_LIB_DIR):
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _JETSON_LIB_DIR not in _ld:
        os.environ["LD_LIBRARY_PATH"] = _JETSON_LIB_DIR + (":" + _ld if _ld else "")
