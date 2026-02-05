"""urwid-based terminal UI for the robot companion.

Layout:
    Scene panel (top, fixed height)
    Conversation log (middle, scrollable)
    Input line (bottom, fixed height)
    Status bar (footer)

Thread safety:
    All widget updates go through urwid.watch_pipe (the only thread-safe bridge).
    User input is sent to the controller via queue.Queue.
"""

import os
import threading
import time

import urwid

# Message types for send_message()
MSG_SCENE = "scene"
MSG_RESPONSE = "response"
MSG_USER = "user"
MSG_STATUS = "status"
MSG_ERROR = "error"
MSG_STREAM_START = "stream_start"
MSG_STREAM_CHUNK = "stream_chunk"
MSG_STREAM_END = "stream_end"

MAX_CONVERSATION_LINES = 200


class TerminalUI(object):
    """urwid terminal interface.  run() blocks on the main thread."""

    def __init__(self, on_user_input=None, on_quit=None, tty_fd=None):
        """
        Args:
            on_user_input: callback(text) called when user submits a message.
                           Called from the urwid main thread -- must be fast.
            on_quit: callback() called when user requests quit.
            tty_fd: file descriptor for the real terminal.  When set, urwid
                    writes to this fd instead of stdout so native C library
                    output on fd 1 (tracker, gstreamer) goes to the log.
        """
        self._on_user_input = on_user_input
        self._on_quit = on_quit
        self._tty_fd = tty_fd

        # Pending messages from background threads
        self._pending_lock = threading.Lock()
        self._pending = []
        self._streaming_widget = None

        # -- Widgets --------------------------------------------------------

        # Scene panel (top)
        self._scene_text = urwid.Text("Scene: waiting for vision...")
        scene_box = urwid.LineBox(
            urwid.Filler(self._scene_text, valign="top"),
            title="Scene",
        )
        scene_fixed = urwid.BoxAdapter(scene_box, height=4)

        # Conversation log (middle, scrollable)
        self._conv_walker = urwid.SimpleFocusListWalker([])
        conv_listbox = urwid.ListBox(self._conv_walker)
        self._conv_box = urwid.LineBox(conv_listbox, title="Conversation")

        # Input line (bottom)
        self._input_edit = urwid.Edit("You: ")
        input_box = urwid.LineBox(
            urwid.Filler(self._input_edit, valign="top")
        )
        input_fixed = urwid.BoxAdapter(input_box, height=3)

        # Status bar
        self._status_text = urwid.Text(" Phase 3 | starting...")
        status_attr = urwid.AttrMap(self._status_text, "status")

        # Layout: scene (fixed) / conversation (weight) / input (fixed) / status
        body = urwid.Pile([
            ("pack", scene_fixed),
            ("weight", 1, self._conv_box),
            ("pack", input_fixed),
            ("pack", status_attr),
        ])

        # Focus the input edit by default
        body.focus_position = 2

        self._palette = [
            ("status", "white", "dark blue"),
        ]

        self._loop = None
        self._pipe_fd = None

    # -- Public API (thread-safe) -------------------------------------------

    def send_message(self, msg_type, text):
        """Enqueue a message from any thread.  Safe to call from background."""
        with self._pending_lock:
            self._pending.append((msg_type, text))
        # Wake urwid main loop by writing a byte to the pipe
        if self._pipe_fd is not None:
            try:
                os.write(self._pipe_fd, b"x")
            except OSError:
                pass

    def run(self, body=None):
        """Start the urwid main loop.  Blocks until quit."""
        if body is None:
            # Rebuild body widget -- urwid needs fresh reference
            scene_fixed = urwid.BoxAdapter(
                urwid.LineBox(
                    urwid.Filler(self._scene_text, valign="top"),
                    title="Scene",
                ),
                height=4,
            )
            input_fixed = urwid.BoxAdapter(
                urwid.LineBox(urwid.Filler(self._input_edit, valign="top")),
                height=3,
            )
            status_attr = urwid.AttrMap(self._status_text, "status")
            body = urwid.Pile([
                ("pack", scene_fixed),
                ("weight", 1, self._conv_box),
                ("pack", input_fixed),
                ("pack", status_attr),
            ])
            body.focus_position = 2

        screen = None
        if self._tty_fd is not None:
            self._tty_output = os.fdopen(self._tty_fd, 'w')
            screen = urwid.raw_display.Screen(output=self._tty_output)

        self._loop = urwid.MainLoop(
            body,
            palette=self._palette,
            unhandled_input=self._on_key,
            screen=screen,
        )
        self._pipe_fd = self._loop.watch_pipe(self._on_pipe_data)
        self._loop.run()

    def stop(self):
        """Request the urwid loop to exit."""
        if self._loop is not None:
            try:
                raise urwid.ExitMainLoop()
            except urwid.ExitMainLoop:
                pass
            # The clean way: set an alarm that raises ExitMainLoop
            def _exit(loop, data):
                raise urwid.ExitMainLoop()
            try:
                self._loop.set_alarm_in(0, _exit)
            except Exception:
                pass

    # -- Private ------------------------------------------------------------

    def _on_pipe_data(self, data):
        """Called by urwid on the main thread when pipe is written to.

        Must return True to keep the pipe registered.
        """
        with self._pending_lock:
            messages = list(self._pending)
            self._pending = []

        for msg_type, text in messages:
            if msg_type == MSG_SCENE:
                self._scene_text.set_text(text)
            elif msg_type == MSG_STATUS:
                self._status_text.set_text(" " + text)
            elif msg_type == MSG_ERROR:
                self._append_conv("[ERROR] " + text)
            elif msg_type == MSG_RESPONSE:
                ts = time.strftime("%H:%M:%S")
                self._append_conv("[{} Robot] {}".format(ts, text))
            elif msg_type == MSG_USER:
                ts = time.strftime("%H:%M:%S")
                self._append_conv("[{} You] {}".format(ts, text))
            elif msg_type == MSG_STREAM_START:
                ts = time.strftime("%H:%M:%S")
                widget = urwid.Text("[{} Robot] ".format(ts))
                self._conv_walker.append(widget)
                # Cap history
                while len(self._conv_walker) > MAX_CONVERSATION_LINES:
                    self._conv_walker.pop(0)
                self._conv_walker.set_focus(len(self._conv_walker) - 1)
                self._streaming_widget = widget
            elif msg_type == MSG_STREAM_CHUNK:
                if self._streaming_widget is not None:
                    current = self._streaming_widget.get_text()[0]
                    self._streaming_widget.set_text(current + text)
                    self._conv_walker.set_focus(len(self._conv_walker) - 1)
            elif msg_type == MSG_STREAM_END:
                self._streaming_widget = None
        return True

    def _append_conv(self, text):
        """Add a line to the conversation walker and scroll to bottom."""
        self._conv_walker.append(urwid.Text(text))
        # Cap history
        while len(self._conv_walker) > MAX_CONVERSATION_LINES:
            self._conv_walker.pop(0)
        # Scroll to bottom
        self._conv_walker.set_focus(len(self._conv_walker) - 1)

    def _on_key(self, key):
        """Handle unhandled key presses."""
        if key == "enter":
            text = self._input_edit.get_edit_text().strip()
            if text:
                self._input_edit.set_edit_text("")
                # Show user message immediately
                ts = time.strftime("%H:%M:%S")
                self._append_conv("[{} You] {}".format(ts, text))
                if self._on_user_input is not None:
                    self._on_user_input(text)
            return

        if key in ("q", "esc"):
            # Only quit when input is empty
            if not self._input_edit.get_edit_text().strip():
                if self._on_quit is not None:
                    self._on_quit()
                raise urwid.ExitMainLoop()
