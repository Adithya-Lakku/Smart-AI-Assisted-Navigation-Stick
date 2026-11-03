"""
server/vision_pipeline.py
=========================
Server-side computer vision pipeline.
Decodes JPEG frames, executes YOLOv11n inference, and calculates 3-zone spatial reasoning.
"""

from __future__ import annotations
import time
import logging
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np

from vision.detector import get_detector
from vision.spatial_reasoning import SpatialReasoner, SpatialObstacle
from config.settings import config

logger = logging.getLogger("smart_stick.server_vision")


class VisionPipeline:
    """Manages server-side YOLOv11n and spatial reasoning."""

    def __init__(self):
        self.detector = get_detector(
            model_name=config.vision.model_name,
            confidence=config.vision.confidence_threshold,
        )
        self.reasoner = SpatialReasoner(
            left_ratio=config.vision.left_boundary_ratio,
            right_ratio=config.vision.right_boundary_ratio,
            near_ratio=config.vision.near_ratio,
            medium_ratio=config.vision.medium_ratio,
        )

    def process(
        self,
        jpeg_bytes: bytes,
        ultrasonic_distance_cm: Optional[float] = None
    ) -> Tuple[List[SpatialObstacle], Optional[np.ndarray], float]:
        """
        Decodes JPEG bytes, detects objects, and produces sorted spatial obstacles.
        Returns (obstacles: List[SpatialObstacle], frame: np.ndarray, latency_ms: float)
        """
        t_start = time.perf_counter()

        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.warning("[VisionPipeline] Failed to decode incoming JPEG bytes.")
            return [], None, 0.0

        raw_detections = self.detector.detect(frame)
        obstacles = self.reasoner.analyze(raw_detections, ultrasonic_distance_cm=ultrasonic_distance_cm)

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return obstacles, frame, round(latency_ms, 1)
