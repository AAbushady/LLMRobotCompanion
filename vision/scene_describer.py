import time

from . import config

_ZONE_NAMES = [
    ["upper-left", "upper-center", "upper-right"],
    ["center-left", "center", "center-right"],
    ["lower-left", "lower-center", "lower-right"],
]


def _get_zone(center, frame_width, frame_height):
    """Map a (cx, cy) point to a spatial zone name."""
    if frame_width <= 0 or frame_height <= 0:
        return "unknown area"
    cx, cy = center
    col = min(int(cx / (frame_width / 3.0)), 2)
    row = min(int(cy / (frame_height / 3.0)), 2)
    return _ZONE_NAMES[row][col]


def _format_duration(seconds):
    """Format a duration for human reading."""
    if seconds < 1:
        return "just appeared"
    elif seconds < 60:
        return "present {}s".format(int(seconds))
    else:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        if secs == 0:
            return "present {}m".format(mins)
        return "present {}m {}s".format(mins, secs)


def describe_scene(world_state, event_bus=None):
    """
    Generate a text scene description from current world state.

    Args:
        world_state: WorldState instance
        event_bus: optional EventBus for recent events summary

    Returns:
        str: Human-readable scene description for LLM consumption
    """
    objects = world_state.snapshot()
    frame_width, frame_height = world_state.get_frame_size()
    now = time.time()

    if not objects:
        return "Scene (empty): No objects detected."

    person_count = 0
    other_count = 0
    for obj in objects.values():
        if obj["class_id"] == config.PERSON_CLASS_ID:
            person_count += 1
        else:
            other_count += 1

    total = len(objects)

    # Header
    count_parts = []
    if total == 1:
        count_parts.append("1 object")
    else:
        count_parts.append("{} objects".format(total))
    if person_count > 0:
        count_parts.append("{} {}".format(
            person_count, "person" if person_count == 1 else "people"))

    header = "Scene ({})".format(", ".join(count_parts))

    # Per-object descriptions sorted by confidence
    sorted_objects = sorted(
        objects.values(),
        key=lambda o: o["confidence"],
        reverse=True
    )

    descriptions = []
    for obj in sorted_objects:
        zone = _get_zone(obj["center"], frame_width, frame_height)
        conf_pct = int(obj["confidence"] * 100)
        duration = now - obj["first_seen"]
        dur_str = _format_duration(duration)
        desc = "A {} in the {} area ({}%, {})".format(
            obj["class_name"], zone, conf_pct, dur_str)
        descriptions.append(desc)

    result = header + ": " + ". ".join(descriptions) + "."

    # Recent events summary
    if event_bus is not None:
        recent = event_bus.get_recent(5)
        event_strs = []
        for evt in recent:
            age = now - evt["timestamp"]
            if age > 30:
                continue
            etype = evt["type"]
            edata = evt["data"]
            name = edata.get("class_name", "object")

            if etype == "person_entered":
                event_strs.append("Person entered ({:.0f}s ago)".format(age))
            elif etype == "person_left":
                event_strs.append("Person left ({:.0f}s ago)".format(age))
            elif etype == "object_appeared":
                event_strs.append("{} appeared ({:.0f}s ago)".format(
                    name.capitalize(), age))
            elif etype == "object_disappeared":
                event_strs.append("{} disappeared ({:.0f}s ago)".format(
                    name.capitalize(), age))
            elif etype == "object_moved":
                event_strs.append("{} moved ({:.0f}s ago)".format(
                    name.capitalize(), age))

        if event_strs:
            result += " Recent: " + "; ".join(event_strs) + "."

    return result
