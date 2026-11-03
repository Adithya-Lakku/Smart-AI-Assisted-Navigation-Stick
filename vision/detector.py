"""
vision/detector.py
==================
YOLOv11n object detection engine tailored for assistive navigation.
Detects obstacles, navigability markers (doors, stairs, paths), and people.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger("smart_stick.vision")

# Key obstacle and navigation classes from COCO / assistive navigation
NAVIGATION_CLASSES = {
    "person", "chair", "table", "door", "stairs", "wall", "railing",
    "trash bin", "pole", "bench", "bed", "couch", "toilet", "sink",
    "refrigerator", "tv", "laptop", "bicycle", "motorcycle", "car",
    "truck", "bus", "fire hydrant", "stop sign", "traffic light",
    "backpack", "umbrella", "suitcase", "dog", "cat"
}


class ObjectDetector:
    """Wrapper around YOLOv11 for real-time obstacle detection."""

    def __init__(self, model_name: str = "yolo11n.pt", confidence_threshold: float = 0.35):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._model = None

    def load_model(self):
        """Lazy load or preload YOLO model."""
        if self._model is not None:
            return self._model

        logger.info(f"Loading YOLO model: {self.model_name}...")
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load {self.model_name}: {e}. Attempting fallback to yolov8n.pt...")
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolov8n.pt")
                logger.info("Fallback YOLOv8n model loaded.")
            except Exception as e2:
                logger.error(f"Failed to load any YOLO model: {e2}")
                self._model = None
        return self._model

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on an OpenCV BGR image.
        Returns a list of raw detected bounding boxes and metadata.
        """
        model = self.load_model()
        if model is None or image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        results = model(image, verbose=False)[0]

        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Filter relevant navigation classes
            if label not in NAVIGATION_CLASSES:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
                "box_width": x2 - x1,
                "box_height": y2 - y1,
                "frame_width": w,
                "frame_height": h,
            })

        return detections


# Global singleton instance
_default_detector: Optional[ObjectDetector] = None

def get_detector(model_name: str = "yolo11n.pt", confidence: float = 0.35) -> ObjectDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = ObjectDetector(model_name=model_name, confidence_threshold=confidence)
    return _default_detector
