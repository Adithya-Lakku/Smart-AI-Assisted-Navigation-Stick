"""
edge/hardware/gpio_controller.py
================================
Hardware Abstraction Layer for Raspberry Pi 3B+ GPIO peripherals.
Controls HC-SR04 ultrasonic sensor, active buzzer, vibration motor,
and tactile pushbutton with automatic fallback to mock hardware on non-Pi platforms.
"""

from __future__ import annotations
import time
import logging
from typing import Optional, Callable
import threading

logger = logging.getLogger("smart_stick.gpio")

# Attempt importing RPi.GPIO
try:
    import RPi.GPIO as GPIO
    REAL_HARDWARE = True
except (ImportError, RuntimeError):
    REAL_HARDWARE = False
    logger.info("[GPIO] RPi.GPIO not detected. Initializing MockGPIO mode for development.")


class MockGPIO:
    """Mock implementation of RPi.GPIO for desktop simulation and automated testing."""
    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    PUD_UP = "PUD_UP"
    HIGH = 1
    LOW = 0

    _pin_states = {}
    _button_pressed = False

    @classmethod
    def setmode(cls, mode):
        logger.debug(f"[MockGPIO] setmode({mode})")

    @classmethod
    def setwarnings(cls, flag):
        pass

    @classmethod
    def setup(cls, pin, mode, pull_up_down=None):
        cls._pin_states[pin] = 0

    @classmethod
    def output(cls, pin, value):
        cls._pin_states[pin] = value

    @classmethod
    def input(cls, pin):
        # By default PUD_UP button is unpressed (HIGH / 1). When pressed, it pulls to LOW (0).
        if pin == 17:  # default button pin
            return 0 if cls._button_pressed else 1
        return cls._pin_states.get(pin, 0)

    @classmethod
    def cleanup(cls):
        cls._pin_states.clear()

    @classmethod
    def simulate_button_press(cls, duration_sec: float = 0.2):
        """Helper to simulate physical button press in tests."""
        cls._button_pressed = True
        time.sleep(duration_sec)
        cls._button_pressed = False


class GPIOController:
    """High-level hardware controller for Smart Navigation Stick."""

    def __init__(
        self,
        pin_trig: int = 23,
        pin_echo: int = 24,
        pin_buzzer: int = 18,
        pin_vibration: int = 25,
        pin_button: int = 17,
        force_mock: bool = False,
    ):
        self.trig = pin_trig
        self.echo = pin_echo
        self.buzzer = pin_buzzer
        self.vibration = pin_vibration
        self.button = pin_button
        self.is_mock = force_mock or not REAL_HARDWARE
        self.gpio = MockGPIO if self.is_mock else GPIO

        self._active_haptic_thread: Optional[threading.Thread] = None
        self._stop_haptic_event = threading.Event()
        self._lock = threading.Lock()

        self._initialize_pins()

    def _initialize_pins(self):
        """Set up pin modes and default states."""
        self.gpio.setwarnings(False)
        self.gpio.setmode(self.gpio.BCM)

        # Ultrasonic
        self.gpio.setup(self.trig, self.gpio.OUT)
        self.gpio.setup(self.echo, self.gpio.IN)
        self.gpio.output(self.trig, False)

        # Alert Actuators
        self.gpio.setup(self.buzzer, self.gpio.OUT)
        self.gpio.setup(self.vibration, self.gpio.OUT)
        self.gpio.output(self.buzzer, False)
        self.gpio.output(self.vibration, False)

        # Pushbutton with internal pull-up resistor
        self.gpio.setup(self.button, self.gpio.IN, pull_up_down=self.gpio.PUD_UP)

        time.sleep(0.05)
        logger.info(
            f"[GPIO] Pins initialized: Trig={self.trig}, Echo={self.echo}, "
            f"Buzzer={self.buzzer}, Vibration={self.vibration}, Button={self.button} (Mock={self.is_mock})"
        )

    # ---------------- Ultrasonic Sensor ----------------
    def measure_distance(self) -> Optional[float]:
        """
        Measure distance via HC-SR04 ultrasonic echo.
        Returns distance in centimeters, or None if timed out.
        """
        if self.is_mock:
            # Return realistic simulated distance for testing
            return 75.0

        # Send 10µs pulse
        self.gpio.output(self.trig, True)
        time.sleep(0.00001)
        self.gpio.output(self.trig, False)

        timeout = time.time() + 0.04  # 40ms timeout (~6m range limit)
        pulse_start = time.time()

        while self.gpio.input(self.echo) == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                return None

        pulse_end = time.time()
        while self.gpio.input(self.echo) == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                return None

        duration = pulse_end - pulse_start
        distance_cm = duration * 17150.0  # Speed of sound: 343m/s -> 17150 cm/s roundtrip
        return round(distance_cm, 1)

    # ---------------- Button Sensing ----------------
    def is_button_pressed(self) -> bool:
        """Active-low reading: button connected between GPIO and GND."""
        return self.gpio.input(self.button) == 0

    # ---------------- Actuators: Buzzer & Vibration ----------------
    def set_buzzer(self, state: bool):
        """Turn active buzzer ON or OFF directly."""
        self.gpio.output(self.buzzer, state)

    def set_vibration(self, state: bool):
        """Turn vibration motor ON or OFF directly."""
        self.gpio.output(self.vibration, state)

    def trigger_urgent_alert(self):
        """
        High-urgency collision alert:
        Continuous vibration with rapid buzzer pulses.
        """
        self.set_vibration(True)
        self.set_buzzer(True)

    def trigger_warning_alert(self):
        """
        Medium-urgency proximity alert:
        Pulsing vibration and intermittent beep.
        """
        self.set_vibration(True)
        self.set_buzzer(False)

    def silence_all(self):
        """Immediately stop all acoustic and haptic alert signals."""
        self.set_buzzer(False)
        self.set_vibration(False)

    def pulse_feedback(self, duration_sec: float = 0.15):
        """Single tactile haptic pulse acknowledging user interaction (e.g. button click)."""
        def _pulse():
            self.set_vibration(True)
            time.sleep(duration_sec)
            self.set_vibration(False)
        threading.Thread(target=_pulse, daemon=True).start()

    def cleanup(self):
        """Release GPIO resources."""
        self.silence_all()
        try:
            self.gpio.cleanup()
        except Exception:
            pass
        logger.info("[GPIO] Cleaned up pins.")
