"""Unit tests for GimbalController's mapping math.

Some private functions are tested directly since they are
crucial for mapping rpy poses to gimbal movements.
"""

import pytest

from gimbal_controller import (
    GimbalController,
    _apply_deadband,
    _clamp,
    _limit_step,
    _smooth,
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


class TestMapping:

    def test_passthrough_no_inversion_or_offset(self):
        ctrl = make_controller(make_config())
        pose = HeadPose(roll=10.0, pitch=-20.0, yaw=50.0)
        assert ctrl._map_head_to_gimbal(pose) == (10.0, -20.0, 50.0)
    
    def test_inversion_flips_only_flagged_axes(self):
        config = make_config()
        config["mapping"]["invert_roll"] = True
        config["mapping"]["invert_yaw"] = True
        ctrl = make_controller(config)
        pose = HeadPose(roll=10.0, pitch=-20.0, yaw=50.0)
        assert ctrl._map_head_to_gimbal(pose) == (-10.0, -20.0, -50.0)
    
    def test_offset_applied_after_inversion(self):
        config = make_config()
        config["mapping"]["invert_roll"] = True
        config["mapping"]["roll_offset"] = 5.0
        ctrl = make_controller(config)
        pose = HeadPose(roll=10.0, pitch=0.0, yaw=0.0)
        roll, _, _ = ctrl._map_head_to_gimbal(pose)
        assert roll == pytest.approx(-5.0) # (-10) + 5
    
    def test_ddaeadband_zeroes_small_values(self):
        ctrl = make_controller(make_config())
        pose = HeadPose(roll=0.2, pitch=-0.24, yaw=0.249)
        assert ctrl._map_head_to_gimbal(pose) == (0.0, 0.0, 0.0)

    def test_deadband_applies_after_offset(self):
        # Nonzero head angle that lands inside deadband after offset
        # should map to zero.
        config = make_config()
        config["mapping"]["yaw_offset"] = -0.9
        ctrl = make_controller(config)
        pose = HeadPose(roll=0.0, pitch=0.0, yaw=1.0)
        _, _, yaw = ctrl._map_head_to_gimbal(pose)
        assert yaw == 0.0
    
    def test_limits_clamp_both_directions(self):
        ctrl = make_controller(make_config())
        pose = HeadPose(roll=999.0, pitch=-999.0, yaw=-999.0)
        assert ctrl._map_head_to_gimbal(pose) == (45.0, -135.0, -135.0)


class TestHelpers:

    @pytest.mark.parametrize(
        "x,lo,hi,expected",
        [(5, 0, 10, 5), (-1, 0, 10, 0), (11, 0, 10, 10)],
    )
    def test_clamp(self, x, lo, hi, expected):
        assert _clamp(x, lo, hi) == expected

    @pytest.mark.parametrize(
        "x,expected", [(0.1, 0.0), (-0.24, 0.0), (0.25, 0.25), (1.0, 1.0)]
    )
    def test_apply_deadband(self, x, expected):
        assert _apply_deadband(x, 0.25) == expected
    
    def test_smooth_moves_fraction_toward_target(self):
        assert _smooth(0.0, 10.0, 0.85) == pytest.approx(8.5)
        assert _smooth(10.0, 10.0, 0.85) == pytest.approx(10.0)
    
    def test_limit_step_caps_movement(self):
        assert _limit_step(0.0, 100.0, 12.0) == 12.0
        assert _limit_step(0.0, -100.0, 12.0) == -12.0
        assert _limit_step(0.0, 5.0, 12.0) == 5.0
    

class TestSingletonBehavior:

    def test_second_init_returns_first_instance(self):
        first = make_controller(make_config())
        other_config = make_config()
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
    

class FakeSerial:
    """Simulates fake port for shutdown testing."""

    def __init__(self):
        self.written = []
    
    def write(self, data):
        self.written.append(bytes(data))

    def flush(self):
        pass
    
    def close(self):
        pass


class TestShutdown:

    def test_stop_send_repeated_center_packets(self, monkeypatch):
        """stop() must emit 3 identical mode-3 packets and close
        the port. time.sleep is patched out so drain/spacing delays
        do not slow down testing.
        """
        import gimbal_controller as gc
        from heq_protocol import (
            CMD_GIMBAL_CONTROL,
            MODE_RETURN_TO_CENTER,
            build_control_payload,
            build_packet,
        )

        monkeypatch.setattr(gc.time, "sleep", lambda seconds: None)

        ctrl = make_controller(make_config())
        fake = FakeSerial()
        ctrl._ser = fake

        ctrl.stop()

        expected = build_packet(
            CMD_GIMBAL_CONTROL,
            build_control_payload(mode=MODE_RETURN_TO_CENTER),
        )
        assert fake.written == [expected] * 3
        assert ctrl._ser is None
    
    def test_stop_without_open_port_safe(self):
        ctrl = make_controller(make_config())
        assert ctrl._ser is None
        ctrl.stop() # no error should be raised
