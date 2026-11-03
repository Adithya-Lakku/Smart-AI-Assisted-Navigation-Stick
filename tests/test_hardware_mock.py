"""
tests/test_hardware_mock.py
===========================
Unit tests for MockGPIO and GPIOController.
Validates ultrasonic distance calculation, buzzer, vibration motor, and button states.
"""

import unittest
from edge.hardware.gpio_controller import GPIOController, MockGPIO


class TestHardwareMock(unittest.TestCase):

    def test_gpio_mock_initialization(self):
        controller = GPIOController(force_mock=True)
        self.assertTrue(controller.is_mock)
        self.assertEqual(controller.trig, 23)
        self.assertEqual(controller.echo, 24)
        self.assertEqual(controller.buzzer, 18)
        self.assertEqual(controller.vibration, 25)
        self.assertEqual(controller.button, 17)

    def test_gpio_mock_distance(self):
        controller = GPIOController(force_mock=True)
        dist = controller.measure_distance()
        self.assertIsNotNone(dist)
        self.assertGreater(dist, 0)

    def test_gpio_mock_actuators(self):
        controller = GPIOController(force_mock=True)
        controller.trigger_urgent_alert()
        self.assertEqual(MockGPIO._pin_states[18], 1)  # Buzzer HIGH
        self.assertEqual(MockGPIO._pin_states[25], 1)  # Vibration HIGH

        controller.silence_all()
        self.assertEqual(MockGPIO._pin_states[18], 0)
        self.assertEqual(MockGPIO._pin_states[25], 0)

    def test_gpio_mock_button_press(self):
        controller = GPIOController(force_mock=True)
        self.assertFalse(controller.is_button_pressed())

        MockGPIO.simulate_button_press(duration_sec=0.01)
        self.assertFalse(controller.is_button_pressed())


if __name__ == "__main__":
    unittest.main()
