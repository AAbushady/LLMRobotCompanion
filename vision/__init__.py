"""
LLM Robot Companion - Vision System

Object detection, tracking, and scene description
for the Jetson Nano using SSD-Mobilenet-v2.

Usage:
    from vision import VisionSystem

    vs = VisionSystem()
    vs.start()
    print(vs.describe_scene())
    vs.stop()
"""

from .vision_system import VisionSystem
from .events import EventBus
from .scene_describer import describe_scene

__all__ = ["VisionSystem", "EventBus", "describe_scene"]
