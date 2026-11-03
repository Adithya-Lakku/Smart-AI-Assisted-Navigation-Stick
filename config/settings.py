"""
config/settings.py
==================
Centralized configuration manager for Smart AI Navigation Stick.
Loads default YAML configuration and applies environment variable overrides.
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

try:
    import yaml
except ImportError:
    yaml = None


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config" / "default_config.yaml"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    cooldown_seconds: float = 3.0
    audio_ttl_seconds: int = 60
    save_captures: bool = True
    captures_dir: str = str(BASE_DIR / "captures")
    audio_dir: str = str(BASE_DIR / "audio_files")


@dataclass
class EdgeConfig:
    server_url: str = "http://192.168.1.100:5000"
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    simulate_hardware: bool = False
    continuous_monitoring_interval_sec: float = 0.15


@dataclass
class HardwarePins:
    trigger: int = 23
    echo: int = 24
    buzzer: int = 18
    vibration: int = 25
    button: int = 17


@dataclass
class HardwareThresholds:
    critical_distance_cm: float = 40.0
    warning_distance_cm: float = 80.0
    safe_distance_cm: float = 150.0


@dataclass
class HardwareConfig:
    pins: HardwarePins = field(default_factory=HardwarePins)
    thresholds: HardwareThresholds = field(default_factory=HardwareThresholds)
    pulse_fast_ms: int = 80
    pulse_slow_ms: int = 250


@dataclass
class VisionConfig:
    model_name: str = "yolo11n.pt"
    confidence_threshold: float = 0.35
    left_boundary_ratio: float = 0.33
    right_boundary_ratio: float = 0.66
    near_ratio: float = 0.40
    medium_ratio: float = 0.20


@dataclass
class LlamaCppConfig:
    model_path: str = str(BASE_DIR / "models" / "gemma-4b-it-q4_k_m.gguf")
    mmproj_path: str = str(BASE_DIR / "models" / "gemma-mmproj-f16.gguf")
    server_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.2
    max_tokens: int = 150
    threads: int = 4
    n_gpu_layers: int = 33


@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_tokens: int = 200
    api_key: str = ""


@dataclass
class InferenceConfig:
    mode: str = "local"  # "local" | "cloud" | "mock"
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


@dataclass
class PiperConfig:
    model_path: str = str(BASE_DIR / "models" / "en_US-lessac-medium.onnx")
    config_path: str = str(BASE_DIR / "models" / "en_US-lessac-medium.onnx.json")
    sample_rate: int = 22050


@dataclass
class GttsConfig:
    language: str = "en"
    slow: bool = False


@dataclass
class TTSConfig:
    engine: str = "piper"  # "piper" | "gtts" | "mock"
    piper: PiperConfig = field(default_factory=PiperConfig)
    gtts: GttsConfig = field(default_factory=GttsConfig)


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


def load_config() -> AppConfig:
    """Load configuration from YAML and apply environment variable overrides."""
    raw_data: Dict[str, Any] = {}
    if CONFIG_FILE.exists() and yaml is not None:
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[CONFIG] Warning: Could not parse {CONFIG_FILE}: {e}")

    # Build typed config objects
    server_raw = raw_data.get("server", {})
    edge_raw = raw_data.get("edge", {})
    hw_raw = raw_data.get("hardware", {})
    pins_raw = hw_raw.get("pins", {})
    thresh_raw = hw_raw.get("thresholds", {})
    vision_raw = raw_data.get("vision", {})
    inf_raw = raw_data.get("inference", {})
    llama_raw = inf_raw.get("llama_cpp", {})
    gemini_raw = inf_raw.get("gemini", {})
    tts_raw = raw_data.get("tts", {})
    piper_raw = tts_raw.get("piper", {})
    gtts_raw = tts_raw.get("gtts", {})

    cfg = AppConfig()

    # Server overrides
    cfg.server.host = os.getenv("STICK_SERVER_HOST", server_raw.get("host", "0.0.0.0"))
    cfg.server.port = int(os.getenv("STICK_SERVER_PORT", server_raw.get("port", 5000)))
    cfg.server.debug = os.getenv("STICK_SERVER_DEBUG", str(server_raw.get("debug", False))).lower() == "true"
    cfg.server.cooldown_seconds = float(server_raw.get("cooldown_seconds", 3.0))

    # Edge overrides
    cfg.edge.server_url = os.getenv("STICK_EDGE_SERVER_URL", edge_raw.get("server_url", "http://192.168.1.100:5000"))
    cfg.edge.camera_index = int(os.getenv("STICK_CAMERA_DEVICE_INDEX", edge_raw.get("camera_index", 0)))
    cfg.edge.simulate_hardware = os.getenv("STICK_SIMULATE_HARDWARE", str(edge_raw.get("simulate_hardware", False))).lower() == "true"

    # Hardware pins overrides
    cfg.hardware.pins.trigger = int(os.getenv("STICK_PIN_TRIG", pins_raw.get("trigger", 23)))
    cfg.hardware.pins.echo = int(os.getenv("STICK_PIN_ECHO", pins_raw.get("echo", 24)))
    cfg.hardware.pins.buzzer = int(os.getenv("STICK_PIN_BUZZER", pins_raw.get("buzzer", 18)))
    cfg.hardware.pins.vibration = int(os.getenv("STICK_PIN_VIBRATION", pins_raw.get("vibration", 25)))
    cfg.hardware.pins.button = int(os.getenv("STICK_PIN_BUTTON", pins_raw.get("button", 17)))

    # Thresholds
    cfg.hardware.thresholds.critical_distance_cm = float(os.getenv("STICK_DISTANCE_CRITICAL_CM", thresh_raw.get("critical_distance_cm", 40.0)))
    cfg.hardware.thresholds.warning_distance_cm = float(os.getenv("STICK_DISTANCE_WARNING_CM", thresh_raw.get("warning_distance_cm", 80.0)))

    # Vision overrides
    cfg.vision.model_name = os.getenv("STICK_YOLO_MODEL", vision_raw.get("model_name", "yolo11n.pt"))
    cfg.vision.confidence_threshold = float(os.getenv("STICK_YOLO_CONFIDENCE", vision_raw.get("confidence_threshold", 0.35)))

    # Inference overrides
    cfg.inference.mode = os.getenv("STICK_INFERENCE_ENGINE", inf_raw.get("mode", "local"))
    cfg.inference.llama_cpp.model_path = os.getenv("STICK_LLAMA_MODEL_PATH", llama_raw.get("model_path", cfg.inference.llama_cpp.model_path))
    cfg.inference.llama_cpp.mmproj_path = os.getenv("STICK_LLAMA_MMPROJ_PATH", llama_raw.get("mmproj_path", cfg.inference.llama_cpp.mmproj_path))
    cfg.inference.llama_cpp.server_url = os.getenv("STICK_LLAMA_SERVER_URL", llama_raw.get("server_url", "http://127.0.0.1:8080"))
    cfg.inference.gemini.api_key = os.getenv("GEMINI_API_KEY", "")

    # TTS overrides
    cfg.tts.engine = os.getenv("STICK_TTS_ENGINE", tts_raw.get("engine", "piper"))
    cfg.tts.piper.model_path = os.getenv("STICK_PIPER_MODEL_PATH", piper_raw.get("model_path", cfg.tts.piper.model_path))

    return cfg


# Module-level singleton
config = load_config()
