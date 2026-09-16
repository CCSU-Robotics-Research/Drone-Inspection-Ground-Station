"""Unit tests for frame streaming.

Note that ``FrameStreamer`` is tested against a real localhost
socket on an OS-assigned port.
"""

import socket
import time

import cv2
import numpy as np
import pytest

from stream import (
    MAX_FRAME_BYTES,
    FrameAssembler,
    FrameStreamer,
    encode_jpeg,
    pack_frame
)


def make_frame(value: int, width: int = 64, height: int = 48):
    return np.full((height, width, 3), value, dtype=np.uint8)


class TestFraming:

    def test_pack_prefixes_little_endian(self):
        wire = pack_frame(b"abc")
        assert wire == b"\x03\x00\x00\x00abc"

    def test_pack_rejects_empty_payload(self):
        with pytest.raises(ValueError):
            pack_frame(b"")

    def test_single_frame_single_chunk(self):
        payloads = FrameAssembler().frames_to_payloads(pack_frame(b"hello"))
        assert payloads == [b"hello"]

    def test_frame_one_byte_at_a_time(self):
        wire = pack_frame(b"hello")
        assembler = FrameAssembler()
        payloads = []
        for i in range(len(wire)):
            payloads += assembler.frames_to_payloads(wire[i:i + 1])
        assert payloads == [b"hello"]

    def test_two_frames_in_one_chunk(self):
        wire = pack_frame(b"one") + pack_frame(b"two")
        assert FrameAssembler().frames_to_payloads(wire) == [b"one", b"two"]

    def test_split_across_chunks(self):
        wire = pack_frame(b"payload")
        assembler = FrameAssembler()
        assert assembler.frames_to_payloads(wire[:6]) == []
        assert assembler.frames_to_payloads(wire[6:]) == [b"payload"]

    def test_zero_length(self):
        with pytest.raises(ValueError):
            FrameAssembler().frames_to_payloads(b"\x00\x00\x00\x00")

    def test_oversize_length(self):
        bad = (MAX_FRAME_BYTES + 1).to_bytes(4, "little")
        with pytest.raises(ValueError):
            FrameAssembler().frames_to_payloads(bad)


class TestJpeg:

    def test_encode_produces_jpeg(self):
        # If a frame has the appropriate header bits then it's JPEG
        payload = encode_jpeg(make_frame(128))
        assert payload[:2] == b"\xff\xd8"

    def test_encode_decode(self):
        frame = make_frame(80, width=80, height=60)
        payload = encode_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.shape == (60, 80, 3)


class TestFrameStreamer:

    @pytest.fixture
    def streamer(self):
        s = FrameStreamer(port=0)
        s.start()
        yield s
        s.stop()

    def _connect(self, streamer) -> socket.socket:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(3.0)
        client.connect(("127.0.0.1", streamer.port))
        return client

    def _receive_frames(self, client, minimum=1, timeout_s=3.0):
        assembler = FrameAssembler()
        payloads = []
        deadline = time.monotonic() + timeout_s
        while len(payloads) < minimum and time.monotonic() < deadline:
            chunk = client.recv(65536)
            if not chunk:
                break
            payloads += assembler.frames_to_payloads(chunk)
        return payloads

    def test_send_without_client(self, streamer):
        # This should return
        streamer.send(make_frame(10))

    def test_client_receives_decodable(self, streamer):
        client = self._connect(streamer)
        time.sleep(0.2)

        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            streamer.send(make_frame(200))
            time.sleep(0.02)

        payloads = self._receive_frames(client)
        client.close()

        assert payloads, "no frames arrived"
        decoded = cv2.imdecode(
            np.frombuffer(payloads[0], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded.shape == (48, 64, 3)
        assert abs(int(decoded.mean()) - 200) < 10

    def _send_until_received(self, streamer, client, value):
        """Send frames continuously until the client sees one."""
        assembler = FrameAssembler()
        deadline = time.monotonic() + 3.0
        client.settimeout(0.2)
        while time.monotonic() < deadline:
            streamer.send(make_frame(value))
            try:
                chunk = client.recv(65536)
            except socket.timeout:
                continue
            if chunk and assembler.frames_to_payloads(chunk):
                return True
        return False

    def test_client_reconnection(self, streamer):
        first = self._connect(streamer)
        assert self._send_until_received(streamer, first, 50)
        first.close()

        second = self._connect(streamer)
        assert self._send_until_received(streamer, second, 60)
        second.close()

    def test_stop_with_connection(self, streamer):
        # Test for repeat safety.
        client = self._connect(streamer)
        time.sleep(0.2)
        streamer.stop()
        client.close()
        streamer.stop()
