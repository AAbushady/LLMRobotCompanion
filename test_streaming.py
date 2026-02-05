"""Unit tests for streaming LLM responses.

Tests mock the HTTP layer to verify SSE parsing, retry logic,
controller streaming flow, and UI message dispatch.
Python 3.6 compatible -- no f-strings, dataclasses, or walrus.
"""

import json
import sys
import time
import unittest

try:
    from unittest import mock
except ImportError:
    import mock

try:
    import queue
except ImportError:
    import Queue as queue

# Ensure controller package is importable
sys.path.insert(0, ".")


class MockSSEResponse(object):
    """Fake requests.Response that yields SSE lines."""

    def __init__(self, status_code, lines=None, content_chunk=b""):
        self.status_code = status_code
        self._lines = lines or []
        self._content_chunk = content_chunk
        self._closed = False

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line

    def iter_content(self, chunk_size=512):
        if self._content_chunk:
            yield self._content_chunk

    def close(self):
        self._closed = True

    def json(self):
        return {}


# ---------------------------------------------------------------------------
# 1. Base class stream_complete fallback
# ---------------------------------------------------------------------------
class TestBaseStreamComplete(unittest.TestCase):
    """Base class stream_complete() calls complete() and yields result."""

    def test_fallback_yields_complete_result(self):
        from controller.llm_backend import LLMBackend

        class StubBackend(LLMBackend):
            def complete(self, messages, system_prompt=None, max_tokens=None):
                return "full response"

        backend = StubBackend()
        chunks = list(backend.stream_complete(
            messages=[{"role": "user", "content": "test"}]
        ))
        self.assertEqual(chunks, ["full response"])


# ---------------------------------------------------------------------------
# 2. Claude SSE parsing
# ---------------------------------------------------------------------------
class TestClaudeStreamComplete(unittest.TestCase):

    @mock.patch("controller.llm_backend.requests.post")
    def test_parses_content_block_delta(self, mock_post):
        from controller.llm_backend import ClaudeBackend
        from controller import config
        # Ensure API key is set for the test
        orig_key = config.CLAUDE_API_KEY
        config.CLAUDE_API_KEY = "test-key"

        sse_lines = [
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
        ]
        mock_post.return_value = MockSSEResponse(200, sse_lines)

        try:
            backend = ClaudeBackend()
            chunks = list(backend.stream_complete(
                messages=[{"role": "user", "content": "test"}]
            ))
            self.assertEqual(chunks, ["Hello", " world"])
        finally:
            config.CLAUDE_API_KEY = orig_key

    @mock.patch("controller.llm_backend.requests.post")
    def test_raises_on_stream_error_event(self, mock_post):
        from controller.llm_backend import ClaudeBackend, LLMError
        from controller import config
        orig_key = config.CLAUDE_API_KEY
        config.CLAUDE_API_KEY = "test-key"

        sse_lines = [
            "event: error",
            'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
        ]
        mock_post.return_value = MockSSEResponse(200, sse_lines)

        try:
            backend = ClaudeBackend()
            with self.assertRaises(LLMError) as ctx:
                list(backend.stream_complete(
                    messages=[{"role": "user", "content": "test"}]
                ))
            self.assertIn("Overloaded", str(ctx.exception))
        finally:
            config.CLAUDE_API_KEY = orig_key

    @mock.patch("controller.llm_backend.requests.post")
    def test_raises_on_empty_stream(self, mock_post):
        from controller.llm_backend import ClaudeBackend, LLMError
        from controller import config
        orig_key = config.CLAUDE_API_KEY
        config.CLAUDE_API_KEY = "test-key"

        mock_post.return_value = MockSSEResponse(200, [])

        try:
            backend = ClaudeBackend()
            with self.assertRaises(LLMError):
                list(backend.stream_complete(
                    messages=[{"role": "user", "content": "test"}]
                ))
        finally:
            config.CLAUDE_API_KEY = orig_key


# ---------------------------------------------------------------------------
# 3. OpenAI SSE parsing
# ---------------------------------------------------------------------------
class TestOpenAIStreamComplete(unittest.TestCase):

    @mock.patch("controller.llm_backend.requests.post")
    def test_parses_delta_content(self, mock_post):
        from controller.llm_backend import OpenAIBackend

        sse_lines = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: {"choices":[{"delta":{"content":" there"}}]}',
            "data: [DONE]",
        ]
        mock_post.return_value = MockSSEResponse(200, sse_lines)

        backend = OpenAIBackend(api_url="http://localhost:8080/v1/chat/completions")
        chunks = list(backend.stream_complete(
            messages=[{"role": "user", "content": "test"}]
        ))
        self.assertEqual(chunks, ["Hi", " there"])

    @mock.patch("controller.llm_backend.requests.post")
    def test_handles_done_sentinel(self, mock_post):
        from controller.llm_backend import OpenAIBackend

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"tok"}}]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"SHOULD NOT APPEAR"}}]}',
        ]
        mock_post.return_value = MockSSEResponse(200, sse_lines)

        backend = OpenAIBackend(api_url="http://localhost:8080/v1/chat/completions")
        chunks = list(backend.stream_complete(
            messages=[{"role": "user", "content": "test"}]
        ))
        self.assertEqual(chunks, ["tok"])

    @mock.patch("controller.llm_backend.requests.post")
    def test_raises_on_empty_stream(self, mock_post):
        from controller.llm_backend import OpenAIBackend, LLMError

        mock_post.return_value = MockSSEResponse(200, ["data: [DONE]"])

        backend = OpenAIBackend(api_url="http://localhost:8080/v1/chat/completions")
        with self.assertRaises(LLMError):
            list(backend.stream_complete(
                messages=[{"role": "user", "content": "test"}]
            ))


# ---------------------------------------------------------------------------
# 4. _retry_loop_stream
# ---------------------------------------------------------------------------
class TestRetryLoopStream(unittest.TestCase):

    def _make_backend(self):
        from controller.llm_backend import LLMBackend
        class StubBackend(LLMBackend):
            def complete(self, messages, system_prompt=None, max_tokens=None):
                return "ok"
        return StubBackend()

    def test_returns_response_on_200(self):
        backend = self._make_backend()
        resp = MockSSEResponse(200)
        result = backend._retry_loop_stream(lambda: resp)
        self.assertIs(result, resp)

    def test_retries_on_429(self):
        from controller import config
        orig = config.LLM_RETRIES
        config.LLM_RETRIES = 1

        backend = self._make_backend()
        calls = []

        def make_response():
            calls.append(1)
            if len(calls) == 1:
                return MockSSEResponse(429, content_chunk=b"rate limited")
            return MockSSEResponse(200)

        with mock.patch("controller.llm_backend.time.sleep"):
            result = backend._retry_loop_stream(make_response)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(calls), 2)
        config.LLM_RETRIES = orig

    def test_raises_on_400(self):
        from controller.llm_backend import LLMError
        backend = self._make_backend()
        resp = MockSSEResponse(400, content_chunk=b"bad request")

        with self.assertRaises(LLMError) as ctx:
            backend._retry_loop_stream(lambda: resp)
        self.assertIn("400", str(ctx.exception))

    def test_closes_response_on_error(self):
        from controller.llm_backend import LLMError
        backend = self._make_backend()
        resp = MockSSEResponse(400, content_chunk=b"bad request")

        with self.assertRaises(LLMError):
            backend._retry_loop_stream(lambda: resp)
        self.assertTrue(resp._closed)


# ---------------------------------------------------------------------------
# 5. Controller streaming methods
# ---------------------------------------------------------------------------
class TestControllerStreaming(unittest.TestCase):

    def _make_controller(self):
        from controller.controller import Controller

        ctrl = Controller.__new__(Controller)
        ctrl._stop_event = __import__("threading").Event()
        ctrl._ui = mock.MagicMock()
        ctrl._llm_backend = mock.MagicMock()
        ctrl._context_mgr = mock.MagicMock()
        ctrl._context_mgr.build_context.return_value = "some context"
        ctrl._reasoning_busy = __import__("threading").Event()
        ctrl._reasoning_queue = queue.Queue(maxsize=2)
        ctrl._reasoning_results = queue.Queue()
        ctrl._input_queue = queue.Queue()
        ctrl._reactive_flag = __import__("threading").Event()
        ctrl._reactive_trigger = None
        ctrl._reactive_lock = __import__("threading").Lock()
        ctrl._last_reactive_time = 0.0
        return ctrl

    def test_execute_reasoning_streaming_sends_messages(self):
        ctrl = self._make_controller()
        ctrl._llm_backend.stream_complete.return_value = iter(["Hello", " world"])

        request = {
            "kind": "periodic",
            "trigger": "periodic",
            "context": "test context",
            "user_text": "",
            "enqueued_at": time.time(),
        }
        result = ctrl._execute_reasoning_streaming(request)

        # Check stream_start, stream_chunk x2, stream_end were sent
        calls = ctrl._ui.send_message.call_args_list
        self.assertEqual(calls[0], mock.call("stream_start", ""))
        self.assertEqual(calls[1], mock.call("stream_chunk", "Hello"))
        self.assertEqual(calls[2], mock.call("stream_chunk", " world"))
        self.assertEqual(calls[3], mock.call("stream_end", ""))

        self.assertEqual(result["response"], "Hello world")
        self.assertTrue(result["streamed"])
        self.assertIsNone(result["error"])

    def test_execute_user_reasoning_streaming_extracts_facts(self):
        ctrl = self._make_controller()
        ctrl._llm_backend.stream_complete.return_value = iter(["Response"])

        request = {
            "kind": "user",
            "trigger": "user_input",
            "context": "test context",
            "user_text": "Hello robot",
            "enqueued_at": time.time(),
        }
        result = ctrl._execute_user_reasoning_streaming(request)

        self.assertEqual(result["response"], "Response")
        self.assertTrue(result["streamed"])
        ctrl._context_mgr.queue_fact_extraction.assert_called_once()

    def test_execute_reasoning_streaming_handles_error(self):
        from controller.llm_backend import LLMError

        ctrl = self._make_controller()
        ctrl._llm_backend.stream_complete.side_effect = LLMError("boom")

        request = {
            "kind": "periodic",
            "trigger": "periodic",
            "context": "test context",
            "user_text": "",
            "enqueued_at": time.time(),
        }
        result = ctrl._execute_reasoning_streaming(request)

        # Should still send stream_end on error
        calls = ctrl._ui.send_message.call_args_list
        self.assertEqual(calls[0], mock.call("stream_start", ""))
        self.assertEqual(calls[1], mock.call("stream_end", ""))

        self.assertEqual(result["error"], "boom")
        self.assertTrue(result["streamed"])


# ---------------------------------------------------------------------------
# 6. _deliver_reasoning_result skips UI when streamed
# ---------------------------------------------------------------------------
class TestDeliverResult(unittest.TestCase):

    def _make_controller(self):
        from controller.controller import Controller

        ctrl = Controller.__new__(Controller)
        ctrl._ui = mock.MagicMock()
        ctrl._context_mgr = mock.MagicMock()
        ctrl._context_mgr.get_stats.return_value = {
            "immediate": 0, "short_term": 0, "long_term": 0, "facts": 0
        }
        return ctrl

    def test_skips_ui_send_when_streamed(self):
        ctrl = self._make_controller()
        result = {
            "kind": "user",
            "response": "Hello",
            "user_text": "Hi",
            "error": None,
            "streamed": True,
        }
        ctrl._deliver_reasoning_result(result)

        # Should NOT call send_message with "response"
        for call in ctrl._ui.send_message.call_args_list:
            self.assertNotEqual(call[0][0], "response")

        # But should still add to context
        ctrl._context_mgr.add_response.assert_called_once_with("Hello")

    def test_sends_ui_when_not_streamed(self):
        ctrl = self._make_controller()
        result = {
            "kind": "user",
            "response": "Hello",
            "user_text": "Hi",
            "error": None,
        }
        ctrl._deliver_reasoning_result(result)

        ctrl._ui.send_message.assert_any_call("response", "Hello")
        ctrl._context_mgr.add_response.assert_called_once_with("Hello")


# ---------------------------------------------------------------------------
# 7. Terminal UI streaming message handling
# ---------------------------------------------------------------------------
class TestTerminalUIStreaming(unittest.TestCase):
    """Test the streaming message dispatch without running urwid."""

    def test_stream_messages_build_widget(self):
        # We test the logic by calling _on_pipe_data directly
        # after setting up pending messages
        from controller.terminal_ui import (
            TerminalUI, MSG_STREAM_START, MSG_STREAM_CHUNK, MSG_STREAM_END
        )

        ui = TerminalUI.__new__(TerminalUI)
        ui._pending_lock = __import__("threading").Lock()
        ui._pending = []
        ui._streaming_widget = None

        # We need a real walker for the test
        import urwid
        ui._conv_walker = urwid.SimpleFocusListWalker([])

        # Simulate stream_start
        ui._pending = [(MSG_STREAM_START, "")]
        ui._on_pipe_data(b"x")

        self.assertIsNotNone(ui._streaming_widget)
        self.assertEqual(len(ui._conv_walker), 1)
        text = ui._streaming_widget.get_text()[0]
        self.assertIn("Robot", text)

        # Simulate stream_chunk
        ui._pending = [(MSG_STREAM_CHUNK, "Hello")]
        ui._on_pipe_data(b"x")

        text = ui._streaming_widget.get_text()[0]
        self.assertIn("Hello", text)

        # Simulate another chunk
        ui._pending = [(MSG_STREAM_CHUNK, " world")]
        ui._on_pipe_data(b"x")

        text = ui._streaming_widget.get_text()[0]
        self.assertIn("Hello world", text)

        # Simulate stream_end
        ui._pending = [(MSG_STREAM_END, "")]
        ui._on_pipe_data(b"x")

        self.assertIsNone(ui._streaming_widget)

    def test_stream_chunk_ignored_without_start(self):
        from controller.terminal_ui import TerminalUI, MSG_STREAM_CHUNK

        ui = TerminalUI.__new__(TerminalUI)
        ui._pending_lock = __import__("threading").Lock()
        ui._pending = [(MSG_STREAM_CHUNK, "orphan")]
        ui._streaming_widget = None

        import urwid
        ui._conv_walker = urwid.SimpleFocusListWalker([])

        # Should not raise
        ui._on_pipe_data(b"x")
        self.assertEqual(len(ui._conv_walker), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
