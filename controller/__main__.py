"""Entry point: python -m controller"""

import argparse
import logging
import os
import signal
import sys

from .controller import Controller


def main():
    parser = argparse.ArgumentParser(
        description="LLM Robot Companion controller"
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        help="LLM backend: claude or openai (default: from LLM_BACKEND env)"
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
    parser.add_argument(
        "--headless", action="store_true", default=False,
        help="Run without terminal UI (log to stderr)"
    )

    args = parser.parse_args()

    # Override backend config if specified
    if args.backend:
        from . import config
        config.LLM_BACKEND = args.backend

    # Logging setup: file in UI mode, stderr in headless mode
    log_level = getattr(logging, args.log_level)
    if args.headless:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(project_root, "companion.log")
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            filename=log_path,
            filemode="a",
        )
        # Save real terminal fd for later restoration, then redirect
        # stdout/stderr to the log file so native C library output
        # (TensorRT, gstreamer, ARGUS) doesn't corrupt the urwid display.
        _saved_tty_fd = os.dup(1)
        log_fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        os.dup2(log_fd, 1)  # stdout
        os.dup2(log_fd, 2)  # stderr
        os.close(log_fd)

    ctrl = Controller(
        camera_source=args.camera,
        detection_threshold=args.threshold,
        target_fps=args.fps,
    )

    if args.headless:
        # Headless mode: controller blocks on main thread
        def handle_signal(signum, frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            logging.getLogger(__name__).info(
                "Received %s, stopping...", sig_name
            )
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
    else:
        # UI mode: controller runs in background, urwid owns main thread
        from .terminal_ui import TerminalUI

        # Print loading message to the real terminal before init.
        os.write(_saved_tty_fd,
                 b"Loading models and opening camera... "
                 b"(see companion.log for details)\n")

        # Initialize hardware while stdout/stderr are redirected to the log
        # file, so TensorRT/gstreamer/ARGUS output doesn't hit the terminal.
        try:
            ctrl.init()
        except Exception:
            os.write(_saved_tty_fd, b"Init failed! Check companion.log\n")
            logging.getLogger(__name__).error(
                "Controller init failed", exc_info=True
            )
            sys.exit(1)

        os.write(_saved_tty_fd, b"Ready. Starting UI...\n")

        # DON'T restore fd 1 to the terminal -- keep it pointed at the
        # log file so the tracker's native C stdout output goes there
        # instead of corrupting the urwid display.  Give urwid a direct
        # handle to the terminal via the saved fd.

        ui = TerminalUI(
            on_user_input=ctrl.on_user_input,
            on_quit=ctrl.stop,
            tty_fd=_saved_tty_fd,
        )

        try:
            ctrl.start_background(ui=ui)
            ui.run()
        except KeyboardInterrupt:
            pass
        except Exception:
            logging.getLogger(__name__).error(
                "Fatal error", exc_info=True
            )
            sys.exit(1)
        finally:
            ctrl.stop()


if __name__ == "__main__":
    main()
