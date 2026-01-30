"""Entry point: python -m controller"""

import argparse
import logging
import signal
import sys

from .controller import Controller


def main():
    parser = argparse.ArgumentParser(
        description="LLM Robot Companion controller"
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        help="LLM backend: claude or aphrodite (default: from LLM_BACKEND env)"
    )
    parser.add_argument(
        "--camera", type=str, default=None,
        help="Camera source (default: csi://0)"
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Vision target FPS (default: 10)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Override backend config if specified
    if args.backend:
        from . import config
        config.LLM_BACKEND = args.backend

    ctrl = Controller(
        camera_source=args.camera,
        detection_threshold=args.threshold,
        target_fps=args.fps,
    )

    # Signal handlers for clean shutdown
    def handle_signal(signum, frame):
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logging.getLogger(__name__).info("Received %s, stopping...", sig_name)
        ctrl.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        ctrl.start()
    except KeyboardInterrupt:
        ctrl.stop()
    except Exception:
        logging.getLogger(__name__).error(
            "Controller failed", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
