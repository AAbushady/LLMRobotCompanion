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
JETSON_INFERENCE_DIR = "/home/alexander/jetson-inference"
MODEL_RESOLVE_DIR = JETSON_INFERENCE_DIR + "/build/aarch64/bin"
