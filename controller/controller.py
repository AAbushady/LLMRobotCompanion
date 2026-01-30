"""Main controller: orchestrates vision, LLM reasoning, and context memory."""

import logging
import threading
import time

from . import config
from .llm_backend import create_backend, create_summarizer_backend, LLMError
from .context_manager import ContextManager

logger = logging.getLogger(__name__)

# Vision event types that trigger reactive reasoning
REACTIVE_EVENTS = {"person_entered", "person_left", "scene_changed"}


class Controller(object):
    """Orchestrates vision, LLM backend, and context manager.

    Usage:
        ctrl = Controller()
        ctrl.start()  # blocks until stop() is called
    """

    def __init__(self, llm_backend=None, camera_source=None,
                 detection_threshold=None, target_fps=None):
        """
        Args:
            llm_backend: LLMBackend instance, or None to create from config.
            camera_source: Camera URI for VisionSystem.
            detection_threshold: Detection confidence threshold.
            target_fps: Vision capture FPS.
        """
        self._llm_backend = llm_backend
        self._camera_source = camera_source
        self._detection_threshold = detection_threshold
        self._target_fps = target_fps

        self._vision = None
        self._context_mgr = None
        self._stop_event = threading.Event()

        # Reactive reasoning state
        self._reactive_flag = threading.Event()
        self._reactive_trigger = None
        self._reactive_lock = threading.Lock()
        self._last_reactive_time = 0.0

    def start(self):
        """Initialize all components and block in the main loop.

        Creates the LLM backend, context manager, and vision system,
        then enters the main loop polling for scene descriptions and
        triggering LLM reasoning.
        """
        logger.info("Controller starting...")

        # LLM backend
        if self._llm_backend is None:
            self._llm_backend = create_backend()
        logger.info("LLM backend: %s", type(self._llm_backend).__name__)

        # Summarizer backend (separate cheaper model, or same as main)
        summarizer = create_summarizer_backend()
        if summarizer:
            logger.info("Summarizer backend: %s", type(summarizer).__name__)
        else:
            logger.info("Summarizer using main backend")
            summarizer = self._llm_backend

        # Context manager
        self._context_mgr = ContextManager(summarizer)
        self._context_mgr.start()

        # Vision system (deferred import — jetson-inference may not be available)
        try:
            from vision import VisionSystem
            self._vision = VisionSystem(
                camera_source=self._camera_source,
                detection_threshold=self._detection_threshold,
                target_fps=self._target_fps,
            )
            self._vision.start()
            self._vision.event_bus.subscribe_all(self._on_vision_event)
            logger.info("Vision system started and subscribed to events")
        except Exception:
            logger.error("Failed to start vision system", exc_info=True)
            self._context_mgr.stop()
            raise

        self._stop_event.clear()
        logger.info("Controller started. Entering main loop.")

        try:
            self._main_loop()
        finally:
            self._shutdown()

    def stop(self):
        """Signal the main loop to exit."""
        logger.info("Controller stop requested")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self):
        """Poll scene, check for reasoning triggers, loop until stopped."""
        last_scene_poll = 0.0
        last_periodic = time.time()
        last_scene_text = ""

        while not self._stop_event.is_set():
            now = time.time()

            # Poll scene description
            if now - last_scene_poll >= config.SCENE_POLL_INTERVAL:
                last_scene_poll = now
                try:
                    scene = self._vision.describe_scene()
                    if scene and scene != last_scene_text:
                        self._context_mgr.add_scene(scene)
                        last_scene_text = scene
                except Exception:
                    logger.debug("Scene poll failed", exc_info=True)

            # Check for reactive trigger
            if self._reactive_flag.is_set():
                self._reactive_flag.clear()
                with self._reactive_lock:
                    trigger = self._reactive_trigger
                    self._reactive_trigger = None

                if trigger and (now - self._last_reactive_time >= config.REACTIVE_COOLDOWN):
                    self._last_reactive_time = now
                    self._do_reasoning("reactive: {}".format(trigger))

            # Periodic reasoning
            if now - last_periodic >= config.PERIODIC_REASONING_INTERVAL:
                last_periodic = now
                if last_scene_text and "empty" not in last_scene_text.lower():
                    self._do_reasoning("periodic")

            # Check vision health
            if self._vision and not self._vision.is_running():
                error = self._vision.get_error()
                if error:
                    logger.error("Vision system error: %s", error)
                    break

            # Sleep with responsive shutdown
            self._stop_event.wait(0.5)

        logger.info("Main loop exited")

    def _on_vision_event(self, event):
        """Vision event callback. Runs on the vision thread — must be fast.

        Adds event to context manager and sets reactive flag for
        qualifying events.
        """
        self._context_mgr.add_event(event)

        etype = event.get("type", "")
        if etype in REACTIVE_EVENTS:
            # For scene_changed, only trigger if delta is significant
            if etype == "scene_changed":
                data = event.get("data", {})
                prev = data.get("previous_count", 0)
                curr = data.get("current_count", 0)
                if abs(curr - prev) < 2:
                    return

            with self._reactive_lock:
                self._reactive_trigger = etype
            self._reactive_flag.set()

    def _do_reasoning(self, trigger):
        """Build context, send to LLM, log the response."""
        logger.info("Reasoning triggered: %s", trigger)

        try:
            context = self._context_mgr.build_context()
            if not context.strip():
                logger.debug("Empty context, skipping reasoning")
                return

            prompt = "{}\n\nTrigger: {}\n\n{}".format(
                config.REASONING_PROMPT, trigger, context
            )

            response = self._llm_backend.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.LLM_MAX_TOKENS,
            )

            logger.info("LLM response [%s]: %s", trigger, response.strip())

            stats = self._context_mgr.get_stats()
            logger.debug(
                "Context stats: immediate=%d, short_term=%d, long_term=%d",
                stats["immediate"], stats["short_term"], stats["long_term"]
            )

        except LLMError as exc:
            logger.error("Reasoning failed [%s]: %s", trigger, exc)
        except Exception:
            logger.error("Unexpected reasoning error [%s]", trigger,
                         exc_info=True)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        """Stop components in order: vision first, then context manager."""
        logger.info("Shutting down...")

        if self._vision is not None:
            try:
                self._vision.stop()
                logger.info("Vision system stopped")
            except Exception:
                logger.error("Error stopping vision", exc_info=True)

        if self._context_mgr is not None:
            try:
                self._context_mgr.stop()
                logger.info("Context manager stopped")
            except Exception:
                logger.error("Error stopping context manager", exc_info=True)

        logger.info("Controller shutdown complete")
