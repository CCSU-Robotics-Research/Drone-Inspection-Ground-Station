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


class TestCommandListener:

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
        send(listener.port, b"selfdestruct:now")
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

    def test_handler_exception(self):
        calls = []

        def bad_handler():
            raise RuntimeError("Bad exception handler!")

        listener = CommandListener(
            {
                "record:start": bad_handler,
                "record:toggle": lambda: calls.append(1),
            },
            port=0
        )
        listener.start()
        send(listener.port, b"record:start")
        send(listener.port, b"record:toggle")
        assert wait_until(lambda: len(calls) == 1)
        listener.stop()

    def test_whitespace_stripped(self):
        calls = []
        listener = CommandListener(
            {"record:toggle": lambda: calls.append(1)}, port=0
        )
        listener.start()
        send(listener.port, b"record:toggle\n")
        assert wait_until(lambda: len(calls) == 1)
        listener.stop()

    def test_repeated_stop(self):
        listener = CommandListener({}, port=0)
        listener.start()
        listener.stop()
        listener.stop()

    def test_commands_real_recorder(self, tmp_path):
        recorder = VideoRecorder(output_dir=tmp_path)
        listener = CommandListener(
            {
                "record:start": lambda: recorder.start(
                    (64, 48), fps=10.0
                ),
                "record:stop": recorder.stop
            },
            port=0,
        )
        listener.start()

        send(listener.port, b"record:start")
        assert wait_until(lambda: recorder.is_recording)
        path = recorder.path

        send(listener.port, b"record:stop")
        assert wait_until(lambda: not recorder.is_recording)
        assert path is not None and path.exists()
        listener.stop()
