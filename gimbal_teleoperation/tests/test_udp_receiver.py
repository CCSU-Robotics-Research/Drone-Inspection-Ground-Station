"""Unit tests for UDPReceiver. A real spare localhost UDP socket
is used.
"""

import socket
import time

import pytest

from udp_receiver import HeadPose, UDPReceiver


@pytest.fixture
def running_receiver():
    receiver = UDPReceiver({"listen_ip": "127.0.0.1", "listen_port": 0})
    receiver.start()
    port = receiver._sock.getsockname()[1]
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    yield receiver, sender, port
    sender.close()
    receiver.stop()


def wait_for_pose(receiver, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pose, stamp = receiver.get_latest_pose()
        if pose is not None:
            return pose, stamp
        time.sleep(0.01)
    return None, 0.0


class TestParsePose:
    
    def test_valid_message(self):
        pose = UDPReceiver._parse_pose("1.5,-2.25,30")
        assert pose == HeadPose(roll=1.5, pitch=-2.25, yaw=30.0)
    
    def test_extra_fields_ignored(self):
        pose = UDPReceiver._parse_pose("1,2,3,42,1234.5")
        assert pose == HeadPose(roll=1.0, pitch=2.0, yaw=3.0)
    
    def test_too_few_fields_raises(self):
        with pytest.raises(ValueError):
            UDPReceiver._parse_pose("1,2")
    
    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            UDPReceiver._parse_pose("a,b,c")

