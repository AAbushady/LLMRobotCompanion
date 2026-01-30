import threading
import time
import collections

from . import config

# Event types
PERSON_ENTERED = "person_entered"
PERSON_LEFT = "person_left"
OBJECT_APPEARED = "object_appeared"
OBJECT_DISAPPEARED = "object_disappeared"
OBJECT_MOVED = "object_moved"
SCENE_CHANGED = "scene_changed"
VISION_STARTED = "vision_started"
VISION_STOPPED = "vision_stopped"
VISION_ERROR = "vision_error"


class EventBus(object):
    """Thread-safe publish/subscribe event bus with bounded history."""

    def __init__(self, max_history=None):
        if max_history is None:
            max_history = config.MAX_EVENT_HISTORY
        self._lock = threading.Lock()
        self._subscribers = {}
        self._all_subscribers = []
        self._history = collections.deque(maxlen=max_history)

    def subscribe(self, event_type, callback):
        """Register a callback for a specific event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback):
        """Register a callback for all event types."""
        with self._lock:
            self._all_subscribers.append(callback)

    def emit(self, event_type, data=None):
        """Fire an event. Callbacks are called outside the lock."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data if data is not None else {},
        }
        with self._lock:
            self._history.append(event)
            specific = list(self._subscribers.get(event_type, []))
            all_subs = list(self._all_subscribers)

        for cb in specific:
            try:
                cb(event)
            except Exception:
                pass
        for cb in all_subs:
            try:
                cb(event)
            except Exception:
                pass

    def get_recent(self, n=10):
        """Return the last n events (newest last)."""
        with self._lock:
            items = list(self._history)
        return items[-n:] if n < len(items) else list(items)
