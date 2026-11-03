"""
edge/hardware/camera.py
=======================
Low-latency camera driver for USB webcams and CSI camera modules on Raspberry Pi.
Features buffer flushing to prevent stale frame lag and mock frame generation for headless tests.
"""

from __future__ import annotations
import logging
import time
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("smart_stick.camera")


class CameraController:
    """Manages OpenCV video capture with buffer purging and mock fallback."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        force_mock: bool = False,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.force_mock = force_mock
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self.is_mock = force_mock

        if not self.force_mock:
            self._init_camera()

    def _init_camera(self):
        """Open V4L2 video capture."""
        try:
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._cap.isOpened():
                logger.warning(f"Could not open camera device {self.device_index}. Enabling synthetic camera mock.")
                self.is_mock = True
            else:
                logger.info(f"Camera opened successfully on /dev/video{self.device_index} ({self.width}x{self.height}).")
        except Exception as e:
            logger.warning(f"Camera initialization exception: {e}. Falling back to mock camera.")
            self.is_mock = True

    def _generate_mock_frame(self) -> np.ndarray:
        """Create a synthetic test frame for development/headless testing."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :] = (40, 40, 40)

        # Draw simulated corridor lines
        cv2.line(frame, (int(self.width * 0.33), 0), (int(self.width * 0.33), self.height), (80, 80, 80), 1)
        cv2.line(frame, (int(self.width * 0.66), 0), (int(self.width * 0.66), self.height), (80, 80, 80), 1)

        # Simulated obstacle in center corridor (e.g. Chair / Person)
        cx1, cy1 = int(self.width * 0.40), int(self.height * 0.35)
        cx2, cy2 = int(self.width * 0.60), int(self.height * 0.85)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)
        cv2.putText(
            frame,
            "Simulated Obstacle (Center/Near)",
            (cx1, cy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
        )

        cv2.putText(
            frame,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (10, self.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        return frame

    def capture_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Capture a fresh frame from camera, flushing hardware buffers to ensure zero lag.
        Returns: (success: bool, frame: np.ndarray)
        """
        with self._lock:
            if self.is_mock or self._cap is None:
                return True, self._generate_mock_frame()

            # Flush 3 stale frames from buffer
            for _ in range(3):
                self._cap.grab()

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Frame read failed from webcam. Falling back to synthetic frame.")
                return True, self._generate_mock_frame()

            return True, frame

    def capture_jpeg_bytes(self, quality: int = 85) -> bytes:
        """Capture frame and encode to JPEG bytes."""
        success, frame = self.capture_frame()
        if not success or frame is None:
            frame = self._generate_mock_frame()

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buffer = cv2.imencode(".jpg", frame, encode_param)
        if not ret:
            raise RuntimeError("Failed to encode frame to JPEG format.")
        return buffer.tobytes()

    def release(self):
        """Release video capture device."""
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                self._cap.release()
                logger.info("Camera released.")
