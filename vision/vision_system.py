import threading
import time
import logging
import os

from . import config

logger = logging.getLogger("vision")


class VisionSystem(object):
    """
    Captures camera frames, runs object detection with tracking,
    maintains world state, and produces text scene descriptions.

    Hardware initialization is deferred to start(). Safe to construct
    without jetson-inference installed.
    """

    def __init__(self, camera_source=None, detection_threshold=None,
                 target_fps=None):
        self._camera_source = camera_source or config.CAMERA_SOURCE
        self._threshold = detection_threshold or config.DETECTION_THRESHOLD
        self._target_fps = target_fps or config.TARGET_FPS
        self._frame_interval = 1.0 / self._target_fps

        self._net = None
        self._classifier = None
        self._cudaCrop = None
        self._camera = None
        self._world_state = None
        self._event_bus = None

        self._thread = None
        self._stop_event = threading.Event()
        self._running = False
        self._error = None
        self._lock = threading.Lock()

    def start(self):
        """
        Initialize hardware and start the capture loop thread.

        Blocks during TensorRT engine loading (~2s cached, 10-30s first run).
        Raises RuntimeError if already running.
        """
        with self._lock:
            if self._running:
                raise RuntimeError("VisionSystem is already running")

        logger.info("Starting vision system...")

        from .events import EventBus, VISION_STARTED, VISION_ERROR
        from .world_state import WorldState

        self._event_bus = EventBus(max_history=config.MAX_EVENT_HISTORY)
        self._world_state = WorldState()

        # detectNet resolves models via a relative "networks/" symlink
        # that only exists in the build output directory.
        original_cwd = os.getcwd()
        try:
            os.chdir(config.MODEL_RESOLVE_DIR)
            logger.info("Changed to model dir: %s", config.MODEL_RESOLVE_DIR)
        except OSError as e:
            logger.warning("Could not chdir to %s: %s (model load may fail)",
                           config.MODEL_RESOLVE_DIR, e)

        try:
            from jetson_inference import detectNet, imageNet
            from jetson_utils import videoSource
            from jetson_utils import cudaCrop as _cudaCrop
            self._cudaCrop = _cudaCrop

            logger.info("Loading detection network '%s' (threshold=%.2f)...",
                        config.DETECTION_NETWORK, self._threshold)
            self._net = detectNet(config.DETECTION_NETWORK,
                                  threshold=self._threshold)
            logger.info("Network loaded")

            if config.TRACKING_ENABLED:
                self._net.SetTrackingEnabled(True)
                self._net.SetTrackerType(config.TRACKER_TYPE)
                self._net.SetTrackingParams(
                    minFrames=config.TRACKER_MIN_FRAMES,
                    dropFrames=config.TRACKER_DROP_FRAMES,
                    overlapThreshold=config.TRACKER_OVERLAP_THRESHOLD
                )
                logger.info("Tracking enabled: %s (minFrames=%d, dropFrames=%d, overlap=%.2f)",
                            config.TRACKER_TYPE,
                            config.TRACKER_MIN_FRAMES,
                            config.TRACKER_DROP_FRAMES,
                            config.TRACKER_OVERLAP_THRESHOLD)

            if config.CLASSIFICATION_ENABLED:
                logger.info("Loading classification network '%s'...",
                            config.CLASSIFICATION_NETWORK)
                self._classifier = imageNet(config.CLASSIFICATION_NETWORK)
                logger.info(
                    "Classification network loaded (%d classes)",
                    self._classifier.GetNumClasses()
                )

            logger.info("Opening camera: %s", self._camera_source)
            self._camera = videoSource(self._camera_source)

        except Exception as e:
            self._error = str(e)
            logger.error("Failed to initialize: %s", e)
            self._event_bus.emit(VISION_ERROR, {"error": str(e)})
            raise
        finally:
            try:
                os.chdir(original_cwd)
            except OSError:
                pass

        self._stop_event.clear()
        with self._lock:
            self._running = True
            self._error = None

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="vision-capture",
            daemon=True
        )
        self._thread.start()
        self._event_bus.emit(VISION_STARTED, {})
        logger.info("Vision system started (target FPS: %d)", self._target_fps)

    def stop(self):
        """Stop the capture loop and release resources."""
        from .events import VISION_STOPPED

        logger.info("Stopping vision system...")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Capture thread did not stop within timeout")
            self._thread = None

        with self._lock:
            self._running = False

        self._camera = None
        self._net = None
        self._classifier = None

        if self._event_bus is not None:
            self._event_bus.emit(VISION_STOPPED, {})
        logger.info("Vision system stopped")

    def _capture_loop(self):
        """Main capture/detect loop. Runs in daemon thread."""
        from .events import VISION_ERROR

        logger.debug("Capture loop started")
        consecutive_failures = 0
        max_failures = 30

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            try:
                img = self._camera.Capture()

                if img is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        msg = "Camera failed {} times consecutively".format(
                            consecutive_failures)
                        logger.error(msg)
                        with self._lock:
                            self._error = msg
                        self._event_bus.emit(VISION_ERROR, {"error": msg})
                        break
                    continue

                consecutive_failures = 0

                detections = self._net.Detect(img, overlay=config.DETECTION_OVERLAY)

                # Classify ROIs with imageNet for richer labels
                classifications = {}
                if self._classifier is not None:
                    for det in detections:
                        if det.TrackID < 0 or det.TrackStatus < 0:
                            continue
                        w = det.Right - det.Left
                        h = det.Bottom - det.Top
                        if w < config.CLASSIFICATION_MIN_BBOX_SIZE or \
                           h < config.CLASSIFICATION_MIN_BBOX_SIZE:
                            continue
                        try:
                            roi = self._cudaCrop(img, (det.Left, det.Top,
                                                       det.Right, det.Bottom))
                            class_id, confidence = self._classifier.Classify(roi)
                            if confidence >= config.CLASSIFICATION_THRESHOLD:
                                label = self._classifier.GetClassDesc(class_id)
                                classifications[det.TrackID] = {
                                    "label": label,
                                    "confidence": confidence,
                                }
                        except Exception:
                            pass  # skip failed crops

                self._world_state.update(
                    detections,
                    self._net,
                    self._event_bus,
                    frame_width=img.width,
                    frame_height=img.height,
                    classifications=classifications,
                )

            except Exception as e:
                logger.error("Capture loop error: %s", e, exc_info=True)
                with self._lock:
                    self._error = str(e)
                self._event_bus.emit(VISION_ERROR, {"error": str(e)})
                time.sleep(0.1)
                continue

            # Throttle — stop_event.wait() allows responsive shutdown
            elapsed = time.monotonic() - loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

        logger.debug("Capture loop exited")

    def get_state(self):
        """Return a snapshot of current world state. Thread-safe."""
        if self._world_state is None:
            return {}
        return self._world_state.snapshot()

    def describe_scene(self):
        """Generate a text description of the current scene. Thread-safe."""
        from .scene_describer import describe_scene
        if self._world_state is None:
            return "Vision system not started."
        return describe_scene(self._world_state, self._event_bus)

    def is_running(self):
        """Check if the vision system is running."""
        with self._lock:
            return self._running

    def get_error(self):
        """Return the last error message, or None."""
        with self._lock:
            return self._error

    @property
    def event_bus(self):
        """Access the event bus for subscribing to events."""
        return self._event_bus

    @property
    def fps(self):
        """Current network inference FPS, or 0 if not running."""
        if self._net is not None:
            return self._net.GetNetworkFPS()
        return 0.0
