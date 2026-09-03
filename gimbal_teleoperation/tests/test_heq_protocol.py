"""Unit tests for heq_protocol."""

import pytest

from heq_protocol import (
    ANGLE_PUSH_V2_LEN,
    CMD_ANGLE_PUSH,
    CMD_FUNCTION_READ,
    CMD_GIMBAL_CONTROL,
    HEQParser,
    MODE_ANGLE,
    MODE_LOCK,
    MODE_RETURN_TO_CENTER,
    MODE_SPEED,
    build_control_payload,
    build_packet,
    decode_0x14,
    decode_0x87_v2,
)


def hx(b: bytes) -> str:
    return b.hex(" ").upper()

# Downlink example
EXAMPLES_0X85 = [
    (
        dict(mode=MODE_SPEED, yaw_speed_deg_s=30.0),
        "AE 01 0D 85 93 01 00 00 00 00 00 00 00 00 00 00 B8 0B "
        "CC F5 E1 63",
    ),
    (
        dict(mode=MODE_ANGLE, yaw_angle_deg=30.0),
        "AE 01 0D 85 93 02 00 00 00 00 B8 0B 00 00 00 00 00 00 "
        "76 AB AF 70",
    ),
    (
        dict(mode=MODE_RETURN_TO_CENTER),
        "AE 01 0D 85 93 03 00 00 00 00 00 00 00 00 00 00 00 00 "
        "44 06 BE 68",
    ),
    (
        dict(mode=MODE_LOCK),
        "AE 01 0D 85 93 04 00 00 00 00 00 00 00 00 00 00 00 00 "
        "9B 5B 72 2F",
    ),
]

# Uplink example
TELEMETRY_0X87 = bytes.fromhex(
    "AE011887A000000000AF0091FF03000000"
    "F8FFF1FF0100F8FFF1FF0100E0327E13"
)


@pytest.mark.parametrize("kwargs,expected_hex", EXAMPLES_0X85)
def test_control_frames(kwargs, expected_hex):
    pkt = build_packet(CMD_GIMBAL_CONTROL, build_control_payload(**kwargs))
    assert hx(pkt) == expected_hex


def test_function_read_frame_has_no_crc():
    pkt = build_packet(CMD_FUNCTION_READ)
    assert hx(pkt) == "AE 01 00 13 14"
    assert len(pkt) == 5


def test_header_checksum_is_sum_of_header_fields():
    pkt = build_packet(
        CMD_GIMBAL_CONTROL, build_control_payload(mode=MODE_ANGLE)
    )
    version, length, command, checksum = pkt[1], pkt[2], pkt[3], pkt[4]
    assert checksum == (version + length + command) & 0xFF


def test_payload_angle_rounding():
    payload = build_control_payload(mode=MODE_ANGLE, yaw_angle_deg=30.004)
    yaw_raw = int.from_bytes(payload[5:7], "little", signed=True)
    assert yaw_raw == 3000

    payload = build_control_payload(mode=MODE_ANGLE, pitch_angle_deg=-19.216)
    pitch_raw = int.from_bytes(payload[3:5], "little", signed=True)
    assert pitch_raw == -1922


def test_builder_output_survives_own_parser():
    payload = bytes(range(13))
    pkt = build_packet(CMD_GIMBAL_CONTROL, payload)
    frames = HEQParser().feed(pkt)
    assert len(frames) == 1
    frame = frames[0]
    assert frame.command == CMD_GIMBAL_CONTROL
    assert frame.data == payload
    assert frame.header_ok and frame.crc_ok


class TestParser:

    def test_single_Frame_single_chunk(self):
        frames = HEQParser().feed(TELEMETRY_0X87)
        assert len(frames) == 1
        frame = frames[0]
        assert frame.command == CMD_ANGLE_PUSH
        assert frame.length == ANGLE_PUSH_V2_LEN
        assert frame.header_ok and frame.crc_ok
        assert frame.raw == TELEMETRY_0X87

    def test_frame_per_byte(self):
        parser = HEQParser()
        frames = []
        for i in range(len(TELEMETRY_0X87)):
            frames += parser.feed(TELEMETRY_0X87[i:i + 1])
        assert len(frames) == 1
        assert frames[0].crc_ok
    
    def test_frame_split(self):
        parser = HEQParser()
        assert parser.feed(TELEMETRY_0X87[:7]) == []
        frames = parser.feed(TELEMETRY_0X87[7:])
        assert len(frames) == 1 and frames[0].crc_ok
    
    def test_junk_before_discarded_header(self):
        frames = HEQParser().feed(b"\x00\x55\x12" + TELEMETRY_0X87)
        assert len(frames) == 1 and frames[0].crc_ok
    
    def test_two_frames_in_one_chunk(self):
        frames = HEQParser().feed(TELEMETRY_0X87 * 2)
        assert len(frames) == 2
        assert all(f.crc_ok for f in frames)
    
    def test_corrupted_crc(self):
        bad = TELEMETRY_0X87[:-1] + bytes([TELEMETRY_0X87[-1] ^ 0xFF])
        frames = HEQParser().feed(bad)
        assert len(frames) == 1
        assert frames[0].header_ok
        assert not frames[0].crc_ok
    
    def test_corrupted_checksum(self):
        bad = bytearray(TELEMETRY_0X87)
        bad[4] ^= 0xFF
        frames = HEQParser().feed(bytes(bad))
        assert len(frames) == 1
        assert not frames[0].header_ok
    
    def test_junk_flood_buffer(self):
        parser = HEQParser()
        assert parser.feed(b"\x00" * 2000) == []
        assert len(parser.buffer) <= 16
        # Add a subsequent legit frame
        frames = parser.feed(TELEMETRY_0X87)
        assert len(frames) == 1 and frames[0].crc_ok
    
    def test_empty_feed(self):
        assert HEQParser().feed(b"") == []

class TestDecoders:

    def test_decode_0x87_matches_examples(self):
        frame = HEQParser().feed(TELEMETRY_0X87)[0]
        telemetry = decode_0x87_v2(frame.data)
        assert telemetry["imu_roll"] == 0.0
        assert telemetry["imu_pitch"] == 0.0
        assert telemetry["imu_yaw"] == 1.75
        assert telemetry["hall_roll"] == -1.11
        assert telemetry["hall_pitch"] == 0.03
        assert telemetry["hall_yaw"] == 0.0
        assert telemetry["hall_roll_rate"] == -0.08
        assert telemetry["hall_pitch_rate"] == -0.15
        assert telemetry["hall_yaw_rate"] == 0.01
        assert telemetry["imu_x_rate"] == -0.08
        assert telemetry["imu_y_rate"] == -0.15
        assert telemetry["imu_z_rate"] == 0.01

    def test_decode_0x87_rejects_wrong_len(self):
        with pytest.raises(ValueError):
            decode_0x87_v2(b"\x00" * 12)
    
    def test_decode_0x14_fields(self):
        data = bytearray(15)
        data[11] = 50    # dead zone
        data[12] = 10    # follow speed
        data[13] = 0xFF  # inversion: -1
        result = decode_0x14(bytes(data))
        assert result["dead_zone_range"] == 50
        assert result["follow_speed"] == 10
        assert result["inversion"] == -1
    
    def test_decode_0x14_rejects_wrong_len(self):
        with pytest.raises(ValueError):
            decode_0x14(b"\x00" * 14)
