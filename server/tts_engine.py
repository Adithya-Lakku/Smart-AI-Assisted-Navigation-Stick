"""
server/tts_engine.py
====================
Text-to-Speech (TTS) Engine supporting:
1. PiperTTS: Fast offline neural TTS (<150ms latency, ONNX runtime)
2. gTTS: Cloud Google TTS fallback (1500-2500ms latency, MP3)
3. MockTTS: Generates lightweight synthetic WAV for headless unit testing
"""

from __future__ import annotations
import os
import time
import uuid
import shutil
import wave
import struct
import subprocess
import threading
import logging
from typing import Optional, Dict, Any, Tuple

from config.settings import config

logger = logging.getLogger("smart_stick.tts")


class BaseTTSEngine:
    def synthesize(self, text: str, output_dir: str) -> Tuple[str, str, float]:
        """
        Synthesizes text and saves to output_dir.
        Returns (audio_id: str, file_path: str, synthesis_latency_ms: float)
        """
        raise NotImplementedError


class PiperTTSEngine(BaseTTSEngine):
    """
    Sub-150ms local neural speech synthesis using Piper (ONNX-based).
    Outputs raw or RIFF WAV (22.05 kHz, 16-bit mono).
    """

    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        self.model_path = model_path or config.tts.piper.model_path
        self.config_path = config_path or config.tts.piper.config_path
        self._piper_bin = shutil.which("piper")
        logger.info(f"[PiperTTS] Initialized (binary={self._piper_bin}, model={self.model_path})")

    def synthesize(self, text: str, output_dir: str) -> Tuple[str, str, float]:
        audio_id = uuid.uuid4().hex
        output_path = os.path.join(output_dir, f"{audio_id}.wav")

        t_start = time.perf_counter()

        # Check if piper binary and model exist
        if self._piper_bin and os.path.exists(self.model_path):
            try:
                cmd = [
                    self._piper_bin,
                    "--model", self.model_path,
                    "--output_file", output_path,
                ]
                if self.config_path and os.path.exists(self.config_path):
                    cmd.extend(["--config", self.config_path])

                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.communicate(input=text.encode("utf-8"), timeout=5.0)
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                return audio_id, output_path, round(latency_ms, 1)

            except Exception as e:
                logger.warning(f"[PiperTTS] CLI invocation failed ({e}). Falling back to synthetic WAV.")

        # Fallback to generating a valid lightweight WAV
        _generate_synthetic_speech_wav(output_path, text)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return audio_id, output_path, round(latency_ms, 1)


class GTTSEngine(BaseTTSEngine):
    """
    Cloud fallback using Google Text-to-Speech (gTTS).
    Generates MP3 files with higher network latency (~1.5-2.5s).
    """

    def __init__(self, language: str = "en", slow: bool = False):
        self.language = language
        self.slow = slow

    def synthesize(self, text: str, output_dir: str) -> Tuple[str, str, float]:
        from gtts import gTTS

        audio_id = uuid.uuid4().hex
        output_path = os.path.join(output_dir, f"{audio_id}.mp3")

        t_start = time.perf_counter()
        tts = gTTS(text=text, lang=self.language, slow=self.slow)
        tts.save(output_path)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return audio_id, output_path, round(latency_ms, 1)


class MockTTSEngine(BaseTTSEngine):
    """Instant dummy WAV generator for development and unit testing."""

    def synthesize(self, text: str, output_dir: str) -> Tuple[str, str, float]:
        audio_id = uuid.uuid4().hex
        output_path = os.path.join(output_dir, f"{audio_id}.wav")
        t_start = time.perf_counter()
        _generate_synthetic_speech_wav(output_path, text)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return audio_id, output_path, round(latency_ms, 1)


def _generate_synthetic_speech_wav(file_path: str, text: str, sample_rate: int = 16000):
    """Generate a short clean tone WAV file mimicking spoken audio."""
    duration = min(max(len(text) * 0.05, 0.4), 2.5)  # approximate speech length
    num_samples = int(sample_rate * duration)
    with wave.open(file_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        # Generate 440Hz sine wave tone
        import math
        samples = []
        for i in range(num_samples):
            val = int(32767.0 * 0.2 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
            samples.append(struct.pack("<h", val))
        wav.writeframes(b"".join(samples))


def get_tts_engine(engine_name: Optional[str] = None) -> BaseTTSEngine:
    selected = (engine_name or config.tts.engine).lower()
    if selected == "piper":
        return PiperTTSEngine()
    elif selected == "gtts":
        try:
            import gtts
            return GTTSEngine()
        except ImportError:
            logger.warning("[TTS] gtts not installed. Using Piper/Mock fallback.")
            return PiperTTSEngine()
    else:
        return MockTTSEngine()
