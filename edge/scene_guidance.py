"""
edge/scene_guidance.py
======================
Handles on-demand scene captioning and contextual navigation guidance.
Triggered by physical pushbutton on the walking stick.
Pauses continuous monitoring alerts, contacts the local inference server,
and streams PiperTTS speech feedback through earphones.
"""

from __future__ import annotations
import time
import logging
import requests
from typing import Optional

from edge.hardware.gpio_controller import GPIOController
from edge.hardware.camera import CameraController
from edge.hardware.audio_output import AudioOutputController
from edge.obstacle_monitor import ObstacleMonitor

logger = logging.getLogger("smart_stick.scene_guidance")


class SceneGuidanceService:
    """Manages on-demand scene description and conversational navigation guidance."""

    def __init__(
        self,
        server_url: str,
        gpio: GPIOController,
        camera: CameraController,
        audio: AudioOutputController,
        monitor: ObstacleMonitor,
        timeout_sec: float = 12.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.gpio = gpio
        self.camera = camera
        self.audio = audio
        self.monitor = monitor
        self.timeout = timeout_sec
        self._is_processing = False

    def trigger_guidance(self, prompt_query: Optional[str] = None):
        """
        Executes complete on-demand guidance workflow:
        1. Acknowledge with haptic feedback
        2. Interrupt & pause obstacle buzzer alerts
        3. Capture fresh frame & read ultrasonic range
        4. Transmit to local server for multimodal inference + PiperTTS
        5. Play verbal navigation instructions through earphones
        6. Resume obstacle monitoring loop
        """
        if self._is_processing:
            logger.info("[Guidance] Request already in flight. Ignoring duplicate trigger.")
            return

        self._is_processing = True
        logger.info("[Guidance] Pushbutton pressed - starting on-demand scene analysis...")

        # 1. Haptic pulse confirmation
        self.gpio.pulse_feedback(duration_sec=0.1)

        # 2. Pause obstacle alerts so user can hear clearly
        self.monitor.pause()

        try:
            # 3. Read current telemetry
            dist_cm = self.gpio.measure_distance()
            image_bytes = self.camera.capture_jpeg_bytes()

            # 4. Transmit to server
            headers = {
                "Content-Type": "image/jpeg",
                "X-Ultrasonic-Distance": str(dist_cm) if dist_cm is not None else "",
                "X-User-Query": prompt_query or "",
            }

            upload_url = f"{self.server_url}/upload"
            logger.info(f"[Guidance] Uploading frame ({len(image_bytes)} bytes) to {upload_url}...")

            t_start = time.time()
            response = requests.post(
                upload_url,
                data=image_bytes,
                headers=headers,
                timeout=self.timeout,
            )
            elapsed = time.time() - t_start
            logger.info(f"[Guidance] Server responded in {elapsed:.2f}s (HTTP {response.status_code})")

            if response.status_code == 200:
                payload = response.json()
                audio_url = payload.get("audio_url")
                description = payload.get("description", "")
                logger.info(f"[Guidance] Navigation instruction: \"{description}\"")

                if audio_url:
                    full_audio_url = (
                        audio_url if audio_url.startswith("http") else f"{self.server_url}{audio_url}"
                    )
                    self._download_and_play(full_audio_url)
            else:
                logger.error(f"[Guidance] Server error: {response.status_code} - {response.text}")
                # Play brief error chime or notification

        except requests.exceptions.RequestException as e:
            logger.error(f"[Guidance] Network error reaching inference server: {e}")
        except Exception as ex:
            logger.error(f"[Guidance] Unexpected exception in guidance pipeline: {ex}")
        finally:
            self._is_processing = False
            # 6. Resume continuous obstacle monitoring loop
            self.monitor.resume()

    def _download_and_play(self, full_url: str):
        """Fetch synthesized audio from server and stream to earphones."""
        try:
            res = requests.get(full_url, timeout=self.timeout)
            if res.status_code == 200:
                audio_data = res.content
                # Infer format from headers or fallback to wav
                fmt = "mp3" if "audio/mpeg" in res.headers.get("Content-Type", "") else "wav"
                self.audio.play_bytes(audio_data, format_hint=fmt, block=True)
            else:
                logger.warning(f"[Guidance] Failed to download audio from {full_url}: {res.status_code}")
        except Exception as e:
            logger.error(f"[Guidance] Error during audio download/playback: {e}")
