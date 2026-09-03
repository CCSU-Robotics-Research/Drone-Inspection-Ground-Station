"""Unit tests for GimbalController's mapping math.

Some private functions are tested directly since they are
crucial for mapping rpy poses to gimbal movements.
"""

import pytest

from gimbal_controller import (
    GimbalController,
)
from singleton import SingletonMeta
from udp_receiver import HeadPose


def make_config() -> dict:
    """Dummy config for testing."""
    return {
        "udp": {"listen_ip": "127.0.0.1", "listen_port": 5005},
        "serial": {"port": "COM_FAKE", "baud": 115200},
        "control": {"update_hz": 60},
        "limits": {
            "roll": [-45.0, 45.0],
            "pitch": [-135.0, 45.0],
            "yaw": [-135.0, 135.0],
        },
        "mapping": {
            "invert_roll": False,
            "invert_pitch": False,
            "invert_yaw": False,
            "roll_offset": 0.0,
            "pitch_offset": 0.0,
            "yaw_offset": 0.0,
        },
        "smoothing": {
            "alpha": 0.85,
            "max_step_deg": 12.0,
            "deadband_deg": 0.25,
        },
        "failsafe": {
            "signal_lost_after_s": 0.35,
        },
    }


def make_controller(config: dict) -> GimbalController:
    return GimbalController(config, receiver=None)
class TestSingletonBehavior:

    def test_second_init_returns_first_instance(self):
        first = make_controller(make_config())
        other_config = maek_config()
        other_config["serial"]["port"] = "COM_OTHER"
        second = make_controller(other_config)
        assert second is first
        assert second._serial_port == "COM_FAKE" # Should match the first instance's arg

    def test_reset_fixture_gives_each_test_a_fresh_instance(self):
        # If the autouse fixture in conftest.py failed to clear
        # cahce then this would still be instance from previous tests
        config = make_config()
        config["serial"]["port"] = "COM_NEW"
        ctrl = make_controller(config)
        assert ctrl._serial_port == "COM_NEW"
    
    def test_distinct_classes_get_distinct_instances(self):
        class A(metaclass=SingletonMeta):
            pass
        
        class B(metaclass=SingletonMeta):
            pass
        
        assert A() is A()
        assert B() is B()
        assert A() is not B()
    
