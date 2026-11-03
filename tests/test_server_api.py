"""
tests/test_server_api.py
========================
Integration tests for Flask Server endpoints.
"""

import unittest
import cv2
import numpy as np

try:
    import flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if HAS_FLASK:
    from config.settings import config
    config.inference.mode = "mock"
    config.tts.engine = "mock"
    config.server.cooldown_seconds = 0.0
    from server.app import app


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in current environment")
class TestServerAPI(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")

    def test_benchmark_endpoint(self):
        response = self.client.get("/benchmark")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("comparison", data)
        self.assertIn("local_pipeline", data["comparison"])

    def test_say_endpoint(self):
        response = self.client.post("/say", json={"text": "Navigation test audio"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("audio_url", data)
        self.assertEqual(data["text"], "Navigation test audio")

    def test_upload_endpoint(self):
        # Create synthetic JPEG image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", frame)
        image_bytes = encoded.tobytes()

        headers = {
            "Content-Type": "image/jpeg",
            "X-Ultrasonic-Distance": "65.5",
        }
        response = self.client.post("/upload", data=image_bytes, headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("audio_url", data)
        self.assertIn("description", data)
        self.assertIn("latency", data)
        self.assertIn("total_ms", data["latency"])


if __name__ == "__main__":
    unittest.main()
