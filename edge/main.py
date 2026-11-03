"""
edge/main.py
============
Main entry point for Raspberry Pi 3B+ Edge Navigation Client.
Initializes hardware peripherals, starts continuous obstacle monitoring,
and listens for on-demand guidance button triggers.
"""

from __future__ import annotations
import argparse
import logging
import signal
import sys
import time

from config.settings import config
from edge.hardware.gpio_controller import GPIOController
from edge.hardware.camera import CameraController
from edge.hardware.audio_output import AudioOutputController
from edge.obstacle_monitor import ObstacleMonitor
from edge.scene_guidance import SceneGuidanceService
from vision.detector import get_detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("smart_stick.edge")


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Blind Stick - Raspberry Pi Edge Client")
    parser.add_argument(
        "--server",
        default=config.edge.server_url,
        help=f"Local AI inference server URL (default: {config.edge.server_url})",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=config.edge.camera_index,
        help="OpenCV V4L2 camera device index (default: 0)",
    )
    parser.add_argument(
        "--simulate-hardware",
        action="store_true",
        default=config.edge.simulate_hardware,
        help="Force MockGPIO and synthetic camera for non-Pi desktop testing",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=config.edge.continuous_monitoring_interval_sec,
        help="Obstacle monitoring loop interval in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--enable-edge-yolo",
        action="store_true",
        help="Enable YOLOv11n inference directly on the Pi in addition to server VLM",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 65)
    logger.info("  Smart AI-Assisted Navigation Stick — Raspberry Pi Edge Client")
    logger.info("=" * 65)
    logger.info(f"  Target Server URL  : {args.server}")
    logger.info(f"  Camera Index       : {args.camera_index}")
    logger.info(f"  Simulate Hardware  : {args.simulate_hardware}")
    logger.info(f"  Monitoring Interval: {args.interval}s")
    logger.info(f"  Edge YOLO Active   : {args.enable_edge_yolo}")
    logger.info("=" * 65)

    # 1. Initialize Hardware Abstractions
    gpio = GPIOController(
        pin_trig=config.hardware.pins.trigger,
        pin_echo=config.hardware.pins.echo,
        pin_buzzer=config.hardware.pins.buzzer,
        pin_vibration=config.hardware.pins.vibration,
        pin_button=config.hardware.pins.button,
        force_mock=args.simulate_hardware,
    )

    camera = CameraController(
        device_index=args.camera_index,
        width=config.edge.frame_width,
        height=config.edge.frame_height,
        force_mock=args.simulate_hardware,
    )

    audio = AudioOutputController()

    # Optional local detector on Pi
    detector = get_detector(config.vision.model_name) if args.enable_edge_yolo else None

    # 2. Continuous Obstacle Monitor Thread
    monitor = ObstacleMonitor(
        gpio=gpio,
        camera=camera,
        audio=audio,
        detector=detector,
        loop_interval_sec=args.interval,
    )

    # 3. On-Demand Scene Guidance Service
    guidance = SceneGuidanceService(
        server_url=args.server,
        gpio=gpio,
        camera=camera,
        audio=audio,
        monitor=monitor,
    )

    # Handle graceful exit
    def handle_shutdown(signum, frame):
        logger.info("\n[Shutdown] Stopping Smart Navigation Stick Edge Client...")
        monitor.stop_monitoring()
        audio.stop()
        gpio.cleanup()
        camera.release()
        logger.info("[Shutdown] Complete. Safe travels!")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # 4. Start Obstacle Monitoring Loop
    monitor.start_monitoring()

    logger.info("[Ready] Stick is operational. Press the tactile button for scene guidance.")

    # 5. Main Event Loop - Button Monitoring
    button_was_pressed = False
    try:
        while True:
            is_pressed = gpio.is_button_pressed()
            if is_pressed and not button_was_pressed:
                # Button pressed edge transition
                button_was_pressed = True
                guidance.trigger_guidance()
            elif not is_pressed:
                button_was_pressed = False

            time.sleep(0.05)
    except KeyboardInterrupt:
        handle_shutdown(None, None)


if __name__ == "__main__":
    main()
