"""Unit tests for GimbalController's mapping math.

Some private functions are tested directly since they are
crucial for mapping rpy poses to gimbal movements.
"""

import pytest

from gimbal_controller import (
    GimbalController,
)

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
