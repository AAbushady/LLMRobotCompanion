"""Tiered context memory with async summarization.

Three tiers:
- Immediate: raw events (deque), max 30s age
- Short-term: summarized batches (list), max 5min age
- Long-term: compressed summaries (list)

Background thread handles summarization and compression without blocking
the controller main loop.
"""

import collections
import logging
import threading
import time

from . import config

logger = logging.getLogger(__name__)


class ContextManager(object):
    """Tiered context memory with async LLM-based summarization."""

    def __init__(self, llm_backend):
        """
        Args:
            llm_backend: LLMBackend instance for summarization calls.
        """
        self._llm = llm_backend
        self._lock = threading.Lock()

        # Immediate tier: deque of {"timestamp", "source", "text"}
        self._immediate = collections.deque(
            maxlen=config.IMMEDIATE_MAX_ENTRIES
        )

        # Short-term tier: list of
        # {"timestamp", "covers_from", "covers_to", "text", "source_count"}
        self._short_term = []

        # Long-term tier: list of {"timestamp", "text"}
        self._long_term = []

        # Summarization thread
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the background summarization thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._summarization_loop,
            name="context-summarizer",
            daemon=True,
        )
        self._thread.start()
        logger.info("Context manager started")

    def stop(self):
        """Stop the summarization thread and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("Context manager stopped")

    def add_event(self, event):
        """Add a vision event to the immediate tier.

        Args:
            event: dict with "type", "timestamp", "data" from EventBus.
        """
        text = self._event_to_text(event)
        entry = {
            "timestamp": event.get("timestamp", time.time()),
            "source": "event",
            "text": text,
        }
        with self._lock:
            self._immediate.append(entry)

    def add_scene(self, description):
        """Add a scene description to the immediate tier.

        Args:
            description: str from VisionSystem.describe_scene().
        """
        entry = {
            "timestamp": time.time(),
            "source": "scene",
            "text": description,
        }
        with self._lock:
            self._immediate.append(entry)

    def build_context(self, max_tokens=None):
        """Assemble all three tiers into a context string.

        Returns:
            str with sections [Right now], [Recent history], [Background].
        """
        if max_tokens is None:
            max_tokens = config.TOKEN_BUDGET_TOTAL

        chars_immediate = int(
            max_tokens * config.TOKEN_BUDGET_IMMEDIATE_RATIO
            * config.CHARS_PER_TOKEN
        )
        chars_short = int(
            max_tokens * config.TOKEN_BUDGET_SHORT_TERM_RATIO
            * config.CHARS_PER_TOKEN
        )
        chars_long = int(
            max_tokens * config.TOKEN_BUDGET_LONG_TERM_RATIO
            * config.CHARS_PER_TOKEN
        )

        with self._lock:
            imm_texts = [e["text"] for e in self._immediate]
            short_texts = [e["text"] for e in self._short_term]
            long_texts = [e["text"] for e in self._long_term]

        sections = []

        # Immediate tier: newest first priority
        imm_fitted = self._fit_to_budget(imm_texts, chars_immediate, "newest")
        if imm_fitted:
            sections.append("[Right now]\n" + "\n".join(imm_fitted))

        # Short-term tier: newest first priority
        short_fitted = self._fit_to_budget(short_texts, chars_short, "newest")
        if short_fitted:
            sections.append("[Recent history]\n" + "\n".join(short_fitted))

        # Long-term tier: newest first priority
        long_fitted = self._fit_to_budget(long_texts, chars_long, "newest")
        if long_fitted:
            sections.append("[Background]\n" + "\n".join(long_fitted))

        return "\n\n".join(sections)

    def get_stats(self):
        """Return tier sizes for monitoring.

        Returns:
            dict with "immediate", "short_term", "long_term" counts.
        """
        with self._lock:
            return {
                "immediate": len(self._immediate),
                "short_term": len(self._short_term),
                "long_term": len(self._long_term),
            }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _event_to_text(self, event):
        """Convert a vision event dict to a human-readable string."""
        etype = event.get("type", "unknown")
        data = event.get("data", {})

        if etype == "person_entered":
            return "Person entered the scene."
        elif etype == "person_left":
            duration = data.get("duration", 0)
            return "Person left the scene (was present {:.0f}s).".format(
                duration
            )
        elif etype == "object_appeared":
            name = data.get("class_name", "object")
            return "{} appeared.".format(name.capitalize())
        elif etype == "object_disappeared":
            name = data.get("class_name", "object")
            return "{} disappeared.".format(name.capitalize())
        elif etype == "object_moved":
            name = data.get("class_name", "object")
            return "{} moved significantly.".format(name.capitalize())
        elif etype == "scene_changed":
            prev = data.get("previous_count", "?")
            curr = data.get("current_count", "?")
            return "Scene changed: {} -> {} objects.".format(prev, curr)
        else:
            return "Event: {}".format(etype)

    @staticmethod
    def _fit_to_budget(texts, max_chars, keep="newest"):
        """Select texts that fit within a character budget.

        Args:
            texts: list of strings (oldest first).
            max_chars: maximum total characters.
            keep: "newest" to prioritize recent entries, "oldest" for the reverse.

        Returns:
            list of strings in chronological order that fit the budget.
        """
        if not texts:
            return []

        if keep == "newest":
            # Walk backwards, accumulate from the end
            selected = []
            remaining = max_chars
            for text in reversed(texts):
                cost = len(text) + 1  # +1 for newline separator
                if cost <= remaining:
                    selected.append(text)
                    remaining -= cost
                else:
                    break
            selected.reverse()  # restore chronological order
        else:
            # Walk forwards
            selected = []
            remaining = max_chars
            for text in texts:
                cost = len(text) + 1
                if cost <= remaining:
                    selected.append(text)
                    remaining -= cost
                else:
                    break

        return selected

    # ------------------------------------------------------------------
    # Summarization thread
    # ------------------------------------------------------------------

    def _summarization_loop(self):
        """Background loop: summarize immediate->short-term, compress short-term->long-term."""
        last_summarize = time.time()
        last_compress = time.time()

        while not self._stop_event.is_set():
            now = time.time()

            # Summarize: drain old immediate entries -> short-term
            if now - last_summarize >= config.SUMMARIZE_INTERVAL:
                self._do_summarize(now)
                last_summarize = now

            # Compress: drain old short-term entries -> long-term
            if now - last_compress >= config.COMPRESS_INTERVAL:
                self._do_compress(now)
                last_compress = now

            # Sleep with responsive shutdown
            self._stop_event.wait(5.0)

    def _do_summarize(self, now):
        """Drain immediate entries older than IMMEDIATE_MAX_AGE, summarize into short-term."""
        cutoff = now - config.IMMEDIATE_MAX_AGE

        # Pop old entries under lock
        batch = []
        with self._lock:
            while self._immediate and self._immediate[0]["timestamp"] < cutoff:
                batch.append(self._immediate.popleft())

        if not batch:
            return

        # Build text for summarization (outside lock)
        batch_texts = [e["text"] for e in batch]
        combined = "\n".join(batch_texts)

        logger.debug(
            "Summarizing %d immediate entries (%.0f chars)",
            len(batch), len(combined)
        )

        try:
            summary = self._llm.complete(
                messages=[{"role": "user", "content": combined}],
                system_prompt=config.SUMMARIZE_PROMPT,
                max_tokens=150,
            )
            entry = {
                "timestamp": now,
                "covers_from": batch[0]["timestamp"],
                "covers_to": batch[-1]["timestamp"],
                "text": summary.strip(),
                "source_count": len(batch),
            }
            with self._lock:
                self._short_term.append(entry)
                # Enforce max entries
                while len(self._short_term) > config.SHORT_TERM_MAX_ENTRIES:
                    self._short_term.pop(0)

            logger.debug("Summarized %d entries -> short-term", len(batch))

        except Exception:
            # Re-insert batch on failure (no data loss)
            logger.warning(
                "Summarization failed, re-inserting %d entries", len(batch),
                exc_info=True,
            )
            with self._lock:
                for entry in reversed(batch):
                    self._immediate.appendleft(entry)

    def _do_compress(self, now):
        """Drain short-term entries older than SHORT_TERM_MAX_AGE, compress into long-term."""
        cutoff = now - config.SHORT_TERM_MAX_AGE

        # Pop old entries under lock
        batch = []
        with self._lock:
            new_short = []
            for entry in self._short_term:
                if entry["timestamp"] < cutoff:
                    batch.append(entry)
                else:
                    new_short.append(entry)
            if batch:
                self._short_term = new_short

        if not batch:
            return

        # Build text for compression (outside lock)
        batch_texts = [e["text"] for e in batch]
        combined = "\n".join(batch_texts)

        logger.debug(
            "Compressing %d short-term entries (%.0f chars)",
            len(batch), len(combined)
        )

        try:
            compressed = self._llm.complete(
                messages=[{"role": "user", "content": combined}],
                system_prompt=config.COMPRESS_PROMPT,
                max_tokens=100,
            )
            entry = {
                "timestamp": now,
                "text": compressed.strip(),
            }
            with self._lock:
                self._long_term.append(entry)
                # Enforce max entries
                while len(self._long_term) > config.LONG_TERM_MAX_ENTRIES:
                    self._long_term.pop(0)

            logger.debug("Compressed %d entries -> long-term", len(batch))

        except Exception:
            # Re-insert batch on failure (no data loss)
            logger.warning(
                "Compression failed, re-inserting %d entries", len(batch),
                exc_info=True,
            )
            with self._lock:
                self._short_term = batch + self._short_term
