"""
Vision system entry point.

Usage:
    python3 -m vision [--camera csi://0] [--fps 10] [--threshold 0.5]
                      [--log-level INFO] [--describe-interval 5]
"""

import argparse
import signal
import logging
import time
import sys


def main():
    parser = argparse.ArgumentParser(
        description="LLM Robot Companion - Vision System")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera source (default: /dev/video0)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Target FPS (default: 10)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--describe-interval", type=float, default=5.0,
                        help="Seconds between scene description logs (default: 5)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("vision.main")

    from .vision_system import VisionSystem

    vs = VisionSystem(
        camera_source=args.camera,
        detection_threshold=args.threshold,
        target_fps=args.fps
    )

    shutdown_requested = [False]

    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        shutdown_requested[0] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def log_event(event):
        logger.info("EVENT [%s]: %s", event["type"], event["data"])

    try:
        vs.start()
        vs.event_bus.subscribe_all(log_event)

        logger.info("Vision system running. Press Ctrl+C to stop.")

        last_describe = 0
        while not shutdown_requested[0]:
            time.sleep(0.5)

            now = time.time()
            if now - last_describe >= args.describe_interval:
                scene = vs.describe_scene()
                logger.info("SCENE: %s", scene)
                last_describe = now
                logger.debug("Network FPS: %.1f", vs.fps)

            err = vs.get_error()
            if err and not vs.is_running():
                logger.error("Vision system stopped with error: %s", err)
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
    finally:
        vs.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
