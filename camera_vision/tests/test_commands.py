"""Unit tests for CommandListener.

The listener is tested against a real UDP socket on an
OS-assigned port.
"""

import socket
import time

from commands import CommandListener
from record import VideoRecorder


def send(port: int, payload: bytes) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(payload, ("127.0.0.1", port))
    sock.close()


def wait_until(predicate) -> bool:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def TestCommandListener:

    def test_known_command_dispatches(self):
        calls = []
        listener = CommandListener(
            {"record:toggle": lambda: calls.append(1)}, port=0
        )
        listener.start()
        send(listener.port, b"record:toggle")
        assert wait_until(lambda: len(calls) == 1)
        listener.stop()

    def test_unknown_command(self):
        calls = []
        listener = CommandListener(
            {"record:toggle": lambda: calls.append(1)}, port=0
        )
        listener.start()
        send(litsener.port, b"selfdestruct:now")
        send(listener.port, b"record:toggle")
        assert wait_until(lambda: len(calls) == 1)
        listener.stop()

    def test_garbage_bytes(self):
        calls = []
        listener = CommandListener(
            {"record:toggle": lambda: calls.append(1)}, port=0
        )
        listener.start()
        send(listener.port, b"\xff\xfe\x00garbage")
        send(listener.port, b"record:toggle")
        assert wait_until(lambda: len(calls) == 1)
        listener.stop()

