"""Main controller: orchestrates vision, LLM reasoning, and context memory."""

import logging
import threading
import time

try:
    import queue
except ImportError:
    import Queue as queue

from . import config
from .llm_backend import create_backend, create_summarizer_backend, LLMError
from .context_manager import ContextManager

logger = logging.getLogger(__name__)

# Vision event types that trigger reactive reasoning
REACTIVE_EVENTS = {"person_entered", "person_left", "scene_changed"}


class Controller(object):
    """Orchestrates vision, LLM backend, and context manager.

    Usage (headless):
        ctrl = Controller()
        ctrl.start()  # blocks until stop() is called

    Usage (with UI):
        ctrl = Controller()
        ctrl.start_background(ui=terminal_ui)
        # ... ui.run() blocks on main thread ...
        ctrl.stop()
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
        self._ui = None

        # User input queue (from UI thread to controller thread)
        self._input_queue = queue.Queue()

        # Reactive reasoning state
        self._reactive_flag = threading.Event()
        self._reactive_trigger = None
        self._reactive_lock = threading.Lock()
        self._last_reactive_time = 0.0

        # Controller thread (for background mode)
        self._thread = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_components(self):
        """Initialize LLM backend, context manager, and vision system."""
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

        # Vision system (deferred import -- jetson-inference may not be available)
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

    # ------------------------------------------------------------------
    # Start modes
    # ------------------------------------------------------------------

    def start(self):
        """Initialize all components and block in the main loop (headless)."""
        logger.info("Controller starting (headless)...")
        self._init_components()
        self._stop_event.clear()
        logger.info("Controller started. Entering main loop.")

        try:
            self._main_loop()
        finally:
            self._shutdown()

    def init(self):
        """Initialize all components (LLM, vision, context).

        Call this before start_background() to ensure all native library
        output (TensorRT, gstreamer) finishes before urwid takes the terminal.
        """
        logger.info("Controller initializing components...")
        self._init_components()

        # Wait for the first camera frame so gstreamer/ARGUS native output
        # finishes before the caller starts the UI.
        if self._vision is not None:
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if self._vision.get_state():
                    break
                time.sleep(0.2)
        logger.info("Controller initialization complete")

    def start_background(self, ui=None):
        """Run the main loop in a daemon thread.

        Call init() first if you need native output to finish before the UI.

        Args:
            ui: TerminalUI instance (or None).  If provided, scene/response
                messages are forwarded to it.
        """
        logger.info("Controller starting (background)...")
        self._ui = ui
        if self._context_mgr is None:
            self._init_components()
        self._stop_event.clear()
        logger.info("Controller started. Launching background loop.")

        self._thread = threading.Thread(
            target=self._main_loop,
            name="controller-loop",
            daemon=True,
        )
        self._thread.start()

    def on_user_input(self, text):
        """Called from the UI thread when the user submits a message.

        Non-blocking -- just enqueues for the controller thread.
        """
        self._input_queue.put(text)

    def stop(self):
        """Signal the main loop to exit and wait for shutdown."""
        logger.info("Controller stop requested")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("Controller thread did not stop within timeout")
            self._thread = None

        self._shutdown()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _send_ui(self, msg_type, text):
        """Forward a message to the UI if attached."""
        if self._ui is not None:
            self._ui.send_message(msg_type, text)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self):
        """Poll scene, check for reasoning triggers, loop until stopped."""
        last_scene_poll = 0.0
        last_periodic = time.time()
        last_scene_text = ""
        last_status_update = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            # Drain user input queue
            while True:
                try:
                    text = self._input_queue.get_nowait()
                    self._handle_user_input(text)
                except queue.Empty:
                    break

            # Poll scene description
            if now - last_scene_poll >= config.SCENE_POLL_INTERVAL:
                last_scene_poll = now
                try:
                    scene = self._vision.describe_scene()
                    if scene and scene != last_scene_text:
                        self._context_mgr.add_scene(scene)
                        last_scene_text = scene
                        logger.info("Scene: %s", scene)
                        self._send_ui("scene", scene)
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

            # Update status bar
            if self._ui is not None and now - last_status_update >= 2.0:
                last_status_update = now
                self._update_status()

            # Check vision health
            if self._vision and not self._vision.is_running():
                error = self._vision.get_error()
                if error:
                    logger.error("Vision system error: %s", error)
                    self._send_ui("error", "Vision: " + error)
                    break

            # Sleep with responsive shutdown
            self._stop_event.wait(0.5)

        logger.info("Main loop exited")

    def _handle_user_input(self, text):
        """Process a user message: add to context and trigger reasoning."""
        logger.info("User input: %s", text)
        self._context_mgr.add_user_message(text)
        self._do_user_reasoning(text)

    def _on_vision_event(self, event):
        """Vision event callback. Runs on the vision thread -- must be fast.

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
        """Build context, send to LLM, log and display the response."""
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

            response_text = response.strip()
            logger.info("LLM response [%s]: %s", trigger, response_text)
            self._send_ui("response", response_text)

            # Store response in context for continuity
            self._context_mgr.add_response(response_text)

            stats = self._context_mgr.get_stats()
            logger.debug(
                "Context stats: immediate=%d, short_term=%d, long_term=%d",
                stats["immediate"], stats["short_term"], stats["long_term"]
            )

        except LLMError as exc:
            logger.error("Reasoning failed [%s]: %s", trigger, exc)
            self._send_ui("error", "LLM error: {}".format(exc))
        except Exception:
            logger.error("Unexpected reasoning error [%s]", trigger,
                         exc_info=True)

    def _do_user_reasoning(self, text):
        """Respond to a user message using visual context."""
        logger.info("User reasoning for: %s", text)

        try:
            context = self._context_mgr.build_context()

            prompt = "{}\n\nThe person said: \"{}\"\n\n{}".format(
                config.USER_REASONING_PROMPT, text, context
            )

            response = self._llm_backend.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.LLM_MAX_TOKENS,
            )

            response_text = response.strip()
            logger.info("LLM user response: %s", response_text)
            self._send_ui("response", response_text)

            # Store response in context for continuity
            self._context_mgr.add_response(response_text)

            # Extract facts from the exchange
            exchange = "User said: \"{}\"\nYou replied: \"{}\"".format(
                text, response_text
            )
            self._context_mgr.extract_facts(exchange)

        except LLMError as exc:
            logger.error("User reasoning failed: %s", exc)
            self._send_ui("error", "LLM error: {}".format(exc))
        except Exception:
            logger.error("Unexpected user reasoning error", exc_info=True)

    def _update_status(self):
        """Send current status to the UI."""
        parts = ["Phase 3"]
        if self._vision is not None:
            parts.append("FPS: {:.0f}".format(self._vision.fps))
        stats = self._context_mgr.get_stats()
        parts.append("Ctx: {}/{}/{}/{}".format(
            stats["immediate"], stats["short_term"],
            stats["long_term"], stats.get("facts", 0),
        ))
        self._send_ui("status", " | ".join(parts))

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        """Stop components in order: vision first, then context manager."""
        if self._vision is None and self._context_mgr is None:
            return  # already shut down
        logger.info("Shutting down...")

        if self._vision is not None:
            try:
                self._vision.stop()
                logger.info("Vision system stopped")
            except Exception:
                logger.error("Error stopping vision", exc_info=True)
            self._vision = None

        if self._context_mgr is not None:
            try:
                self._context_mgr.stop()
                logger.info("Context manager stopped")
            except Exception:
                logger.error("Error stopping context manager", exc_info=True)
            self._context_mgr = None

        logger.info("Controller shutdown complete")
