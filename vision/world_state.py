import threading
import time
import copy
import math

from . import config
from .events import (PERSON_ENTERED, PERSON_LEFT, OBJECT_APPEARED,
                     OBJECT_DISAPPEARED, OBJECT_MOVED, SCENE_CHANGED)


class WorldState(object):
    """Maintains tracked objects from detection frames and fires events on changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._objects = {}
        self._frame_width = 0
        self._frame_height = 0
        self._last_scene_change = 0.0
        self._object_count_prev = 0

    def update(self, detections, net, event_bus, frame_width=0, frame_height=0,
               classifications=None):
        """
        Process detections from net.Detect(), update tracked objects, fire events.

        Args:
            detections: list of detectNet.Detection objects
            net: detectNet instance (for GetClassDesc)
            event_bus: EventBus to emit events on
            frame_width: frame width in pixels
            frame_height: frame height in pixels
            classifications: optional dict {track_id: {"label", "confidence"}}
                from imageNet ROI classification
        """
        if classifications is None:
            classifications = {}
        now = time.time()

        if frame_width > 0:
            self._frame_width = frame_width
        if frame_height > 0:
            self._frame_height = frame_height

        seen_ids = set()
        events_to_fire = []

        with self._lock:
            for det in detections:
                tid = det.TrackID
                if tid < 0:
                    continue
                if det.TrackStatus < 0:
                    continue

                seen_ids.add(tid)

                class_id = det.ClassID
                class_name = net.GetClassDesc(class_id)
                center = (det.Center[0], det.Center[1])
                bbox = (det.Left, det.Top, det.Right, det.Bottom)
                status = "active" if det.TrackStatus == 1 else "initializing"

                # Resolve display_name from classification or COCO label
                cls_info = classifications.get(tid)
                display_name = cls_info["label"] if cls_info else class_name

                if tid in self._objects:
                    obj = self._objects[tid]
                    old_center = obj["center"]

                    obj["confidence"] = det.Confidence
                    obj["bbox"] = bbox
                    obj["center"] = center
                    obj["last_seen"] = now
                    obj["status"] = status
                    if cls_info:
                        obj["display_name"] = display_name

                    if status == "active":
                        dx = center[0] - old_center[0]
                        dy = center[1] - old_center[1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist >= config.MOVEMENT_THRESHOLD:
                            events_to_fire.append((
                                OBJECT_MOVED,
                                {"track_id": tid,
                                 "class_name": obj.get("display_name", class_name),
                                 "distance": round(dist, 1),
                                 "from": old_center, "to": center}
                            ))
                else:
                    self._objects[tid] = {
                        "track_id": tid,
                        "class_id": class_id,
                        "class_name": class_name,
                        "display_name": display_name,
                        "confidence": det.Confidence,
                        "bbox": bbox,
                        "center": center,
                        "first_seen": now,
                        "last_seen": now,
                        "status": status,
                    }

                    if status == "active":
                        evt_type = PERSON_ENTERED if class_id == config.PERSON_CLASS_ID else OBJECT_APPEARED
                        events_to_fire.append((
                            evt_type,
                            {"track_id": tid, "class_name": display_name,
                             "confidence": round(det.Confidence, 2)}
                        ))

            # Stale object cleanup
            gone_ids = []
            for tid, obj in self._objects.items():
                if tid not in seen_ids:
                    if now - obj["last_seen"] >= config.STALE_OBJECT_TIMEOUT:
                        gone_ids.append(tid)
                        evt_type = PERSON_LEFT if obj["class_id"] == config.PERSON_CLASS_ID else OBJECT_DISAPPEARED
                        events_to_fire.append((
                            evt_type,
                            {"track_id": tid,
                             "class_name": obj.get("display_name", obj["class_name"]),
                             "duration": round(now - obj["first_seen"], 1)}
                        ))

            for tid in gone_ids:
                del self._objects[tid]

            # Scene change detection
            current_count = len(self._objects)
            if (current_count != self._object_count_prev and
                    now - self._last_scene_change >= config.SCENE_CHANGE_COOLDOWN):
                events_to_fire.append((
                    SCENE_CHANGED,
                    {"object_count": current_count,
                     "previous_count": self._object_count_prev}
                ))
                self._last_scene_change = now
            self._object_count_prev = current_count

        # Fire events outside the lock
        for event_type, data in events_to_fire:
            event_bus.emit(event_type, data)

    def snapshot(self):
        """Return a deep copy of current objects dict. Thread-safe."""
        with self._lock:
            return copy.deepcopy(self._objects)

    def get_frame_size(self):
        """Return (width, height) tuple."""
        return (self._frame_width, self._frame_height)
