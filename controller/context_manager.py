"""Tiered context memory with async summarization.

Four tiers:
- Facts: persistent strings that survive all summarization (max 20)
- Immediate: raw events (deque), max 30s age, importance-scored
- Short-term: summarized batches (list), max 5min age
- Long-term: compressed summaries (list)

Background thread handles summarization and compression without blocking
the controller main loop.  High-importance entries (>=2) skip summarization
and move directly to short-term with original text.
"""

import collections
import logging
import threading
import time

from . import config

logger = logging.getLogger(__name__)

# Importance levels
IMPORTANCE_DEFAULT = 0
IMPORTANCE_EVENT = 1
IMPORTANCE_HIGH = 2
IMPORTANCE_USER = 3


class ContextManager(object):
    """Tiered context memory with async LLM-based summarization."""

    def __init__(self, llm_backend):
        """
        Args:
            llm_backend: LLMBackend instance for summarization calls.
        """
        self._llm = llm_backend
        self._lock = threading.Lock()

        # Facts tier: persistent strings that survive all summarization
        self._facts = []

        # Immediate tier: deque of {"timestamp", "source", "text", "importance"}
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
        importance = self._event_importance(event)
        entry = {
            "timestamp": event.get("timestamp", time.time()),
            "source": "event",
            "text": text,
            "importance": importance,
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
            "importance": IMPORTANCE_DEFAULT,
        }
        with self._lock:
            self._immediate.append(entry)

    def add_user_message(self, text):
        """Add a user message to the immediate tier (highest importance)."""
        entry = {
            "timestamp": time.time(),
            "source": "user",
            "text": "User said: \"{}\"".format(text),
            "importance": IMPORTANCE_USER,
        }
        with self._lock:
            self._immediate.append(entry)

    def add_response(self, text):
        """Add a robot response to the immediate tier (high importance)."""
        entry = {
            "timestamp": time.time(),
            "source": "response",
            "text": "You replied: \"{}\"".format(text),
            "importance": IMPORTANCE_HIGH,
        }
        with self._lock:
            self._immediate.append(entry)

    def add_fact(self, fact):
        """Add a persistent fact (deduplicated)."""
        fact = fact.strip()
        if not fact:
            return
        with self._lock:
            # Simple dedup: skip if already present
            for existing in self._facts:
                if existing.lower() == fact.lower():
                    return
            self._facts.append(fact)
            # Enforce max
            while len(self._facts) > config.FACTS_MAX_ENTRIES:
                self._facts.pop(0)
        logger.debug("Fact added: %s", fact)

    def extract_facts(self, exchange_text):
        """Ask the LLM to extract persistent facts from a conversation exchange.

        Runs synchronously on the calling thread (controller thread).
        """
        try:
            result = self._llm.complete(
                messages=[{"role": "user", "content": exchange_text}],
                system_prompt=config.FACT_EXTRACTION_PROMPT,
                max_tokens=150,
            )
            result = result.strip()
            if result.lower() == "none" or not result:
                return
            for line in result.split("\n"):
                line = line.strip().lstrip("- ").strip()
                if line and line.lower() != "none":
                    self.add_fact(line)
        except Exception:
            logger.debug("Fact extraction failed", exc_info=True)

    def build_context(self, max_tokens=None):
        """Assemble all tiers into a context string.

        Returns:
            str with sections [Known facts], [Right now], [Recent history],
            [Background].
        """
        if max_tokens is None:
            max_tokens = config.TOKEN_BUDGET_TOTAL

        chars_facts = int(
            max_tokens * config.TOKEN_BUDGET_FACTS_RATIO
            * config.CHARS_PER_TOKEN
        )
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
            facts_list = list(self._facts)
            imm_entries = list(self._immediate)
            short_texts = [e["text"] for e in self._short_term]
            long_texts = [e["text"] for e in self._long_term]

        sections = []

        # Facts tier
        facts_fitted = self._fit_to_budget(facts_list, chars_facts, "oldest")
        if facts_fitted:
            sections.append("[Known facts]\n" + "\n".join(facts_fitted))

        # Immediate tier: importance-weighted fitting
        imm_texts = [e["text"] for e in imm_entries]
        imm_importances = [e["importance"] for e in imm_entries]
        imm_fitted = self._fit_to_budget_weighted(
            imm_texts, imm_importances, chars_immediate
        )
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
            dict with "immediate", "short_term", "long_term", "facts" counts.
        """
        with self._lock:
            return {
                "immediate": len(self._immediate),
                "short_term": len(self._short_term),
                "long_term": len(self._long_term),
                "facts": len(self._facts),
            }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _event_importance(event):
        """Score the importance of a vision event."""
        etype = event.get("type", "")
        if etype in ("person_entered", "person_left"):
            return IMPORTANCE_HIGH
        elif etype in ("object_appeared", "object_disappeared",
                       "object_moved", "scene_changed"):
            return IMPORTANCE_EVENT
        return IMPORTANCE_DEFAULT

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

    @staticmethod
    def _fit_to_budget_weighted(texts, importances, max_chars):
        """Select texts by importance (desc) then recency (desc), within budget.

        User messages (importance >= IMPORTANCE_USER) are always included.
        Returns texts in their original chronological order.

        Args:
            texts: list of strings (oldest first, matching importances).
            importances: list of int importance scores.
            max_chars: maximum total characters.

        Returns:
            list of strings in chronological order that fit the budget.
        """
        if not texts:
            return []

        # Build indexed entries: (index, text, importance)
        entries = []
        for i, (text, imp) in enumerate(zip(texts, importances)):
            entries.append((i, text, imp))

        # Sort by importance desc, then by index desc (newest first within tier)
        entries.sort(key=lambda e: (e[2], e[0]), reverse=True)

        selected_indices = set()
        remaining = max_chars

        for idx, text, imp in entries:
            cost = len(text) + 1
            # Always include user messages
            if imp >= IMPORTANCE_USER:
                selected_indices.add(idx)
                remaining -= cost
            elif cost <= remaining:
                selected_indices.add(idx)
                remaining -= cost

        # Return in original chronological order
        result = []
        for i, text in enumerate(texts):
            if i in selected_indices:
                result.append(text)
        return result

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
        """Drain immediate entries older than IMMEDIATE_MAX_AGE.

        High-importance entries (>=IMPORTANCE_HIGH) skip summarization and
        go directly to short-term with their original text.
        Low-importance entries are batched and summarized via LLM.
        """
        cutoff = now - config.IMMEDIATE_MAX_AGE

        # Pop old entries under lock
        batch = []
        with self._lock:
            while self._immediate and self._immediate[0]["timestamp"] < cutoff:
                batch.append(self._immediate.popleft())

        if not batch:
            return

        # Separate high-importance entries from the rest
        high_entries = []
        low_entries = []
        for entry in batch:
            if entry["importance"] >= IMPORTANCE_HIGH:
                high_entries.append(entry)
            else:
                low_entries.append(entry)

        # High-importance entries go directly to short-term
        if high_entries:
            with self._lock:
                for entry in high_entries:
                    self._short_term.append({
                        "timestamp": entry["timestamp"],
                        "covers_from": entry["timestamp"],
                        "covers_to": entry["timestamp"],
                        "text": entry["text"],
                        "source_count": 1,
                    })
                while len(self._short_term) > config.SHORT_TERM_MAX_ENTRIES:
                    self._short_term.pop(0)
            logger.debug(
                "Preserved %d high-importance entries in short-term",
                len(high_entries)
            )

        # Low-importance entries get summarized
        if not low_entries:
            return

        batch_texts = [e["text"] for e in low_entries]
        combined = "\n".join(batch_texts)

        logger.debug(
            "Summarizing %d immediate entries (%.0f chars)",
            len(low_entries), len(combined)
        )

        try:
            summary = self._llm.complete(
                messages=[{"role": "user", "content": combined}],
                system_prompt=config.SUMMARIZE_PROMPT,
                max_tokens=150,
            )
            entry = {
                "timestamp": now,
                "covers_from": low_entries[0]["timestamp"],
                "covers_to": low_entries[-1]["timestamp"],
                "text": summary.strip(),
                "source_count": len(low_entries),
            }
            with self._lock:
                self._short_term.append(entry)
                # Enforce max entries
                while len(self._short_term) > config.SHORT_TERM_MAX_ENTRIES:
                    self._short_term.pop(0)

            logger.debug("Summarized %d entries -> short-term", len(low_entries))

        except Exception:
            # Re-insert batch on failure (no data loss)
            logger.warning(
                "Summarization failed, re-inserting %d entries", len(low_entries),
                exc_info=True,
            )
            with self._lock:
                for entry in reversed(low_entries):
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
