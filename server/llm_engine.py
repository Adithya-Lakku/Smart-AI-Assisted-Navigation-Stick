"""
server/llm_engine.py
====================
Multimodal Inference Engine supporting:
1. Local llama.cpp 4-bit quantized model (e.g. gemma4:e4b / Gemma-VLM / LLaVA GGUF)
2. Cloud Google Gemini Vision API (for comparison baseline & fallback)
3. Simulated Mock Engine (for CI/CD and zero-setup local testing)
"""

from __future__ import annotations
import os
import time
import base64
import logging
import requests
from typing import Optional, Dict, Any

from config.settings import config
from vision.spatial_reasoning import SpatialObstacle
from vision.scene_builder import build_llm_prompt, build_fast_alert_phrase

logger = logging.getLogger("smart_stick.llm_engine")


class BaseMultimodalEngine:
    """Interface for multimodal scene guidance inference."""
    def generate_guidance(
        self,
        image_bytes: bytes,
        obstacles: list[SpatialObstacle],
        ultrasonic_distance_cm: Optional[float] = None,
        user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class LlamaCppEngine(BaseMultimodalEngine):
    """
    Local multimodal inference using 4-bit quantized model via llama.cpp.
    Connects to local llama-server instance (e.g. llama-server --model gemma4-q4.gguf --mmproj ...).
    Cuts latency from 6-8s down to <2.5s.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        temperature: float = 0.2,
        max_tokens: int = 150,
    ):
        self.server_url = server_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"[LlamaCppEngine] Initialized targeting server: {self.server_url}")

    def generate_guidance(
        self,
        image_bytes: bytes,
        obstacles: list[SpatialObstacle],
        ultrasonic_distance_cm: Optional[float] = None,
        user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = build_llm_prompt(obstacles, ultrasonic_distance_cm, user_query)
        base64_img = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "prompt": prompt,
            "image_data": [{"data": base64_img, "id": 10}],
            "temperature": self.temperature,
            "n_predict": self.max_tokens,
            "stream": False,
        }

        t_start = time.perf_counter()
        try:
            # First attempt standard llama.cpp /completion endpoint
            res = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=10.0,
            )
            if res.status_code == 200:
                result = res.json()
                text = result.get("content", "").strip()
            else:
                # Try OpenAI-compatible chat completions endpoint (standard in recent llama-server)
                chat_payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                            ],
                        }
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                res2 = requests.post(f"{self.server_url}/v1/chat/completions", json=chat_payload, timeout=10.0)
                res2.raise_for_status()
                text = res2.json()["choices"][0]["message"]["content"].strip()

            inference_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "text": text,
                "engine": "llama.cpp (4-bit local)",
                "inference_ms": round(inference_ms, 1),
            }

        except Exception as e:
            logger.warning(f"[LlamaCppEngine] Local inference failed ({e}). Falling back to fast heuristic.")
            inference_ms = (time.perf_counter() - t_start) * 1000.0
            heuristic_text = build_fast_alert_phrase(obstacles, ultrasonic_distance_cm)
            return {
                "text": heuristic_text,
                "engine": "heuristic_fallback",
                "inference_ms": round(inference_ms, 1),
                "error": str(e),
            }


class CloudGeminiEngine(BaseMultimodalEngine):
    """
    Cloud fallback using Google Gemini Vision API.
    Baseline pipeline for latency and quality comparisons.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self._model = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
                logger.info(f"[GeminiEngine] Configured model: {self.model_name}")
            except Exception as e:
                logger.warning(f"[GeminiEngine] Could not configure google.generativeai: {e}")

    def generate_guidance(
        self,
        image_bytes: bytes,
        obstacles: list[SpatialObstacle],
        ultrasonic_distance_cm: Optional[float] = None,
        user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._model:
            raise RuntimeError("Gemini API key is not configured.")

        prompt = build_llm_prompt(obstacles, ultrasonic_distance_cm, user_query)
        t_start = time.perf_counter()

        response = self._model.generate_content(
            contents=[
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64.b64encode(image_bytes).decode("utf-8"),
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ]
        )
        inference_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "text": response.text.strip(),
            "engine": f"Google Gemini ({self.model_name})",
            "inference_ms": round(inference_ms, 1),
        }


class MockEngine(BaseMultimodalEngine):
    """Zero-dependency mock engine providing realistic responses for testing."""

    def generate_guidance(
        self,
        image_bytes: bytes,
        obstacles: list[SpatialObstacle],
        ultrasonic_distance_cm: Optional[float] = None,
        user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        text = build_fast_alert_phrase(obstacles, ultrasonic_distance_cm)
        return {
            "text": text,
            "engine": "mock_engine",
            "inference_ms": 15.0,
        }


def get_inference_engine(mode: Optional[str] = None) -> BaseMultimodalEngine:
    selected = (mode or config.inference.mode).lower()
    if selected == "local":
        return LlamaCppEngine(
            server_url=config.inference.llama_cpp.server_url,
            temperature=config.inference.llama_cpp.temperature,
            max_tokens=config.inference.llama_cpp.max_tokens,
        )
    elif selected == "cloud":
        return CloudGeminiEngine(
            api_key=config.inference.gemini.api_key,
            model_name=config.inference.gemini.model,
        )
    else:
        return MockEngine()
