"""
edge/obstacle_monitor.py
========================
Continuous real-time obstacle monitoring loop.
Fuses ultrasonic distance telemetry with YOLOv11n spatial zoning.
Drives buzzer alerts and vibration motor haptics, with full interruptibility
for spoken navigation instructions.
"""

from __future__ import annotations
import time
import threading
import logging
from typing import Optional

from edge.hardware.gpio_controller import GPIOController
from edge.hardware.camera import CameraController
from edge.hardware.audio_output import AudioOutputController
from vision.detector import ObjectDetector
from vision.spatial_reasoning import SpatialReasoner, ThreatLevel

logger = logging.getLogger("smart_stick.obstacle_monitor")


class ObstacleMonitor(threading.Thread):
    """
    Background worker maintaining the continuous obstacle monitoring loop.
    Supports real-time pause/resume when on-demand guidance or speech is active.
    """

    def __init__(
        self,
        gpio: GPIOController,
        camera: CameraController,
        audio: AudioOutputController,
        detector: Optional[ObjectDetector] = None,
        reasoner: Optional[SpatialReasoner] = None,
        loop_interval_sec: float = 0.15,
        vision_interval_sec: float = 0.5,
    ):
        super().__init__(name="ObstacleMonitorThread", daemon=True)
        self.gpio = gpio
        self.camera = camera
        self.audio = audio
        self.detector = detector
        self.reasoner = reasoner or SpatialReasoner()
        self.loop_interval = loop_interval_sec
        self.vision_interval = vision_interval_sec

        self._running = threading.Event()
        self._paused = threading.Event()
        self._last_vision_check = 0.0
        self._cached_threat = ThreatLevel.SAFE

    def start_monitoring(self):
        """Start the background monitoring loop."""
        self._running.set()
        self.start()
        logger.info("[Monitor] Continuous obstacle monitoring loop started.")

    def stop_monitoring(self):
        """Stop monitoring loop and silence actuators."""
        self._running.clear()
        self.gpio.silence_all()

    def pause(self):
        """Temporarily suspend obstacle alerts (e.g., during on-demand guidance)."""
        self._paused.set()
        self.gpio.silence_all()
        logger.debug("[Monitor] Monitoring loop paused.")

    def resume(self):
        """Resume obstacle alerts."""
        self._paused.clear()
        logger.debug("[Monitor] Monitoring loop resumed.")

    def run(self):
        """Continuous execution loop."""
        while self._running.is_set():
            # If paused or audio is actively speaking, silence alerts and sleep
            if self._paused.is_set() or self.audio.is_speaking:
                self.gpio.silence_all()
                time.sleep(0.1)
                continue

            try:
                # 1. Ultrasonic Distance Measurement
                dist_cm = self.gpio.measure_distance()

                # 2. Periodic Edge Vision Check (YOLOv11n)
                now = time.time()
                if self.detector and (now - self._last_vision_check >= self.vision_interval):
                    self._last_vision_check = now
                    _, frame = self.camera.capture_frame()
                    if frame is not None:
                        raw_dets = self.detector.detect(frame)
                        obstacles = self.reasoner.analyze(raw_dets, ultrasonic_distance_cm=dist_cm)
                        if obstacles:
                            self._cached_threat = obstacles[0].threat_level
                        else:
                            self._cached_threat = ThreatLevel.SAFE

                # 3. Decision Matrix & Alert Actuation
                self._dispatch_alerts(dist_cm, self._cached_threat)

            except Exception as e:
                logger.error(f"[Monitor] Error in monitoring cycle: {e}")
                self.gpio.silence_all()

            time.sleep(self.loop_interval)

    def _dispatch_alerts(self, dist_cm: Optional[float], visual_threat: ThreatLevel):
        """
        Actuates buzzer and vibration motor based on threat levels:
        - Critical: Distance < 40cm OR Visual Threat is CRITICAL (Center corridor + Near)
        - Warning: Distance 40cm - 80cm OR Visual Threat is WARNING
        - Safe: Distance > 80cm and no visual hazard
        """
        is_critical = (dist_cm is not None and dist_cm < 40.0) or (visual_threat == ThreatLevel.CRITICAL)
        is_warning = (dist_cm is not None and 40.0 <= dist_cm < 80.0) or (visual_threat == ThreatLevel.WARNING)

        if is_critical:
            # Continuous vibration and buzzer beep for urgent proximity
            self.gpio.trigger_urgent_alert()
        elif is_warning:
            # Gentle pulsing vibration
            self.gpio.trigger_warning_alert()
        else:
            self.gpio.silence_all()
