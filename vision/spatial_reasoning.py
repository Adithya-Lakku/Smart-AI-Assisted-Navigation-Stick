"""
vision/spatial_reasoning.py
===========================
Transforms 2D bounding boxes into 3D-aware spatial navigation guidance.
Divides visual field into Left, Center, and Right corridors, calculates
relative proximity, and prioritizes navigation hazards.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import enum


class Direction(str, enum.Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class Proximity(str, enum.Enum):
    NEAR = "near"       # < 1 meter / urgent collision risk
    MEDIUM = "medium"   # 1 - 2.5 meters / caution
    FAR = "far"         # > 2.5 meters / background awareness


class ThreatLevel(str, enum.Enum):
    SAFE = "safe"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SpatialObstacle:
    label: str
    direction: Direction
    proximity: Proximity
    confidence: float
    threat_level: ThreatLevel
    bbox: tuple[int, int, int, int]
    estimated_distance_cm: Optional[float] = None


class SpatialReasoner:
    """Performs spatial zoning and threat assessment on visual detections."""

    def __init__(
        self,
        left_ratio: float = 0.33,
        right_ratio: float = 0.66,
        near_ratio: float = 0.40,
        medium_ratio: float = 0.20,
    ):
        self.left_ratio = left_ratio
        self.right_ratio = right_ratio
        self.near_ratio = near_ratio
        self.medium_ratio = medium_ratio

    def classify_direction(self, center_x: float, frame_width: int) -> Direction:
        """Categorize horizontal position into LEFT, CENTER, or RIGHT."""
        if center_x < frame_width * self.left_ratio:
            return Direction.LEFT
        elif center_x < frame_width * self.right_ratio:
            return Direction.CENTER
        else:
            return Direction.RIGHT

    def classify_proximity(self, box_height: float, frame_height: int) -> Proximity:
        """Estimate proximity based on bounding box vertical coverage."""
        ratio = box_height / max(frame_height, 1)
        if ratio >= self.near_ratio:
            return Proximity.NEAR
        elif ratio >= self.medium_ratio:
            return Proximity.MEDIUM
        else:
            return Proximity.FAR

    def assess_threat(self, direction: Direction, proximity: Proximity) -> ThreatLevel:
        """Evaluate navigational urgency for a given obstacle."""
        if direction == Direction.CENTER:
            if proximity == Proximity.NEAR:
                return ThreatLevel.CRITICAL
            elif proximity == Proximity.MEDIUM:
                return ThreatLevel.WARNING
            else:
                return ThreatLevel.INFO
        else:
            # Lateral obstacles (left/right)
            if proximity == Proximity.NEAR:
                return ThreatLevel.WARNING
            return ThreatLevel.INFO

    def analyze(
        self,
        raw_detections: List[Dict[str, Any]],
        ultrasonic_distance_cm: Optional[float] = None
    ) -> List[SpatialObstacle]:
        """
        Process raw YOLO detections into prioritized spatial obstacles.
        Deduplicates by (label, direction) keeping the highest confidence.
        """
        seen: Dict[tuple[str, Direction], SpatialObstacle] = {}

        for det in raw_detections:
            direction = self.classify_direction(det["center_x"], det["frame_width"])
            proximity = self.classify_proximity(det["box_height"], det["frame_height"])
            threat = self.assess_threat(direction, proximity)

            # If ultrasonic distance is available and this object is in the center,
            # fuse the sensor reading to elevate threat if necessary
            est_cm = None
            if direction == Direction.CENTER and ultrasonic_distance_cm is not None:
                est_cm = ultrasonic_distance_cm
                if ultrasonic_distance_cm < 40.0:
                    threat = ThreatLevel.CRITICAL
                    proximity = Proximity.NEAR
                elif ultrasonic_distance_cm < 80.0:
                    threat = max(threat, ThreatLevel.WARNING)

            key = (det["label"], direction)
            if key not in seen or det["confidence"] > seen[key].confidence:
                seen[key] = SpatialObstacle(
                    label=det["label"],
                    direction=direction,
                    proximity=proximity,
                    confidence=det["confidence"],
                    threat_level=threat,
                    bbox=det["box"],
                    estimated_distance_cm=est_cm,
                )

        # Sort by urgency: CRITICAL > WARNING > INFO > SAFE
        threat_order = {
            ThreatLevel.CRITICAL: 0,
            ThreatLevel.WARNING: 1,
            ThreatLevel.INFO: 2,
            ThreatLevel.SAFE: 3,
        }
        sorted_obstacles = sorted(
            seen.values(),
            key=lambda o: (threat_order.get(o.threat_level, 4), -o.confidence)
        )
        return sorted_obstacles
