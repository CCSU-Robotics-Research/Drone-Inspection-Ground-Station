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


class TestReceiverSocket:

    def test_no_pose_before_first_datagram(self, running_receiver):
        receiver, _, _ = running_receiver
        pose, stamp = receiver.get_latest_pose()
        assert pose is None
        assert stamp == 0.0

    def test_receives_and_parses_pose(self, running_receiver):
        receiver, sender, port = running_receiver
        sender.sendto(b"10.5,-20.25,30.0", ("127.0.0.1", port))
        pose, stamp = wait_for_pose(receiver)
        assert pose == HeadPose(roll=10.5, pitch=-20.25, yaw=30.0)
        assert time.monotonic() - stamp < 2.0

    def test_keeps_only_newest_pose(self, running_receiver):
        receiver, sender, port = running_receiver
        sender.sendto(b"1,1,1", ("127.0.0.1", port))
        wait_for_pose(receiver)
        sender.sendto(b"2,2,2", ("127.0.0.1", port))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            pose, _ = receiver.get_latest_pose()
            if pose is not None and pose.roll == 2.0:
                break
            time.sleep(0.01)
        assert receiver.get_latest_pose()[0] == HeadPose(2.0, 2.0, 2.0)

    def test_bad_datagram_is_ignored(self, running_receiver):
        receiver, sender, port = running_receiver
        sender.sendto(b"2,2,2", ("127.0.0.1", port))
        good, good_stamp = wait_for_pose(receiver)
        assert good == HeadPose(2.0, 2.0, 2.0)

        sender.sendto(b"not,a", ("127.0.0.1", port))
        sender.sendto(b"\xff\xfe\x00garbage", ("127.0.0.1", port))
        time.sleep(0.3)

        pose, stamp = receiver.get_latest_pose()
        assert pose == good
        assert stamp == good_stamp

    def test_stop_ok_repeat_safe(self, running_receiver):
        receiver, _, _ = running_receiver
        receiver.stop()
        receiver.stop()  # shouldn't raise errors
