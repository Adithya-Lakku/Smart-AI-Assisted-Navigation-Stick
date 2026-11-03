"""
tests/test_spatial_reasoning.py
===============================
Unit tests for 3-zone spatial reasoning and LLM prompt generation.
"""

import unittest
from vision.spatial_reasoning import (
    SpatialReasoner,
    Direction,
    Proximity,
    ThreatLevel,
)
from vision.scene_builder import (
    build_llm_prompt,
    build_fast_alert_phrase,
    obstacles_to_dict,
)


class TestSpatialReasoning(unittest.TestCase):

    def test_spatial_direction_zones(self):
        reasoner = SpatialReasoner()
        width = 600

        self.assertEqual(reasoner.classify_direction(center_x=100, frame_width=width), Direction.LEFT)
        self.assertEqual(reasoner.classify_direction(center_x=300, frame_width=width), Direction.CENTER)
        self.assertEqual(reasoner.classify_direction(center_x=500, frame_width=width), Direction.RIGHT)

    def test_spatial_proximity_estimation(self):
        reasoner = SpatialReasoner(near_ratio=0.40, medium_ratio=0.20)
        height = 400

        # > 40% height (160px) -> NEAR
        self.assertEqual(reasoner.classify_proximity(box_height=200, frame_height=height), Proximity.NEAR)
        # 20% - 40% (80px - 160px) -> MEDIUM
        self.assertEqual(reasoner.classify_proximity(box_height=100, frame_height=height), Proximity.MEDIUM)
        # < 20% (<80px) -> FAR
        self.assertEqual(reasoner.classify_proximity(box_height=40, frame_height=height), Proximity.FAR)

    def test_spatial_threat_evaluation(self):
        reasoner = SpatialReasoner()

        # Center + Near -> CRITICAL
        self.assertEqual(reasoner.assess_threat(Direction.CENTER, Proximity.NEAR), ThreatLevel.CRITICAL)
        # Center + Medium -> WARNING
        self.assertEqual(reasoner.assess_threat(Direction.CENTER, Proximity.MEDIUM), ThreatLevel.WARNING)
        # Left + Near -> WARNING
        self.assertEqual(reasoner.assess_threat(Direction.LEFT, Proximity.NEAR), ThreatLevel.WARNING)
        # Right + Far -> INFO
        self.assertEqual(reasoner.assess_threat(Direction.RIGHT, Proximity.FAR), ThreatLevel.INFO)

    def test_scene_builder_prompt(self):
        reasoner = SpatialReasoner()
        raw_detections = [
            {
                "label": "chair",
                "confidence": 0.88,
                "box": (200, 100, 380, 350),
                "center_x": 290.0,
                "center_y": 225.0,
                "box_width": 180.0,
                "box_height": 250.0,
                "frame_width": 640,
                "frame_height": 480,
            }
        ]
        obstacles = reasoner.analyze(raw_detections, ultrasonic_distance_cm=35.0)
        self.assertEqual(len(obstacles), 1)
        self.assertEqual(obstacles[0].threat_level, ThreatLevel.CRITICAL)

        prompt = build_llm_prompt(obstacles, ultrasonic_distance_cm=35.0)
        self.assertIn("chair", prompt.lower())
        self.assertIn("35.0 cm", prompt)

        fast_phrase = build_fast_alert_phrase(obstacles, ultrasonic_distance_cm=35.0)
        self.assertIn("Warning!", fast_phrase)


if __name__ == "__main__":
    unittest.main()
