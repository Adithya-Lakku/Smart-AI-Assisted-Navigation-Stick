"""
edge/hardware/audio_output.py
=============================
Audio playback controller for spoken earphone feedback.
Supports aplay (low-latency WAV), mpg123 (MP3), ffplay, and mock fallback.
Exposes playback state so acoustic obstacle warnings can be silenced while speech is playing.
"""

from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import threading
import logging
from typing import Optional

logger = logging.getLogger("smart_stick.audio")


class AudioOutputController:
    """Manages audio playback via system utilities and tracks active speech state."""

    def __init__(self, preferred_player: Optional[str] = None):
        self.player = preferred_player or self._detect_player()
        self.is_speaking_event = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        logger.info(f"[Audio] Selected audio player: {self.player}")

    def _detect_player(self) -> str:
        """Autodetect available Linux audio player."""
        for candidate in ["aplay", "pw-play", "paplay", "mpg123", "ffplay"]:
            if shutil.which(candidate):
                return candidate
        return "mock"

    @property
    def is_speaking(self) -> bool:
        """True while speech feedback is currently playing through earphones."""
        return self.is_speaking_event.is_set()

    def play_bytes(self, audio_data: bytes, format_hint: str = "wav", block: bool = False) -> bool:
        """
        Write audio bytes to a temporary file and play them through the earphone output.
        Format hint: 'wav' or 'mp3'.
        """
        suffix = f".{format_hint.lstrip('.')}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        return self.play_file(tmp_path, auto_delete=True, block=block)

    def play_file(self, file_path: str, auto_delete: bool = False, block: bool = False) -> bool:
        """Play an existing audio file asynchronously or synchronously."""
        def _play_worker():
            with self._lock:
                self.is_speaking_event.set()

            try:
                if self.player == "mock":
                    logger.info(f"[Audio Mock] Playing audio file: {file_path}")
                    # Simulate speech duration based on file size or fixed pause
                    import time
                    time.sleep(1.0)
                elif self.player in ["aplay", "pw-play", "paplay"]:
                    self._current_process = subprocess.Popen(
                        [self.player, file_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_process.wait()
                elif self.player == "mpg123":
                    self._current_process = subprocess.Popen(
                        [self.player, "-q", file_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_process.wait()
                elif self.player == "ffplay":
                    self._current_process = subprocess.Popen(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._current_process.wait()
            except Exception as e:
                logger.error(f"[Audio] Error during playback: {e}")
            finally:
                with self._lock:
                    self._current_process = None
                    self.is_speaking_event.clear()

                if auto_delete and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass

        thread = threading.Thread(target=_play_worker, daemon=True)
        thread.start()
        if block:
            thread.join()
        return True

    def stop(self):
        """Interrupt and stop current playback immediately."""
        with self._lock:
            if self._current_process and self._current_process.poll() is None:
                self._current_process.terminate()
                self._current_process = None
            self.is_speaking_event.clear()
