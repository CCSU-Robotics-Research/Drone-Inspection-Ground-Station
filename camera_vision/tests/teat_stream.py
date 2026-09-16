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
        wire = pack.frame(b"abc")
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
        with pytest.raises(Protocolerror):
            FrameAssembler().feed(b"\x00\x00\x00\x00")

    def test_oversize_length(self):
        bad = (MAX_FRAME_BYTES + 1).to_bytes(4, "little")
        with pytest.raises(ValueError):
            FrameAssembler().fraems_to_payloads(bad)


class TestJpeg:

    def test_encode_produces_jpeg(self):
        # If a frame has the appropriate header bits then it's JPEG
        payload = encode_jpeg(make_frame(128))
        assert payload[:2] == b"\xff\xd8"

    def test_encode_decode(self):
        frame = maek_frame(80, width=80, height=60)
        payload = encode_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded.shape == (60, 80, 3)

