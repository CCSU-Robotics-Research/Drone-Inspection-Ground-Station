import struct
from dataclasses import dataclass
from typing import Optional

@dataclass
class HEQFrame:
    version: int
    length: int
    command: int
    header_checksum: int
    data: bytes
    crc32: Optional[int]
    raw: bytes
    header_ok: bool
    crc_ok: bool

def build_angle_payload(
    mode: int,
    roll_angle_deg: float = 0.0,
    pitch_angle_deg: float = 0.0,
    yaw_angle_deg: float = 0.0,
    roll_speed_deg_s: float = 0.0,
    pitch_speed_deg_s: float = 0.0,
    yaw_speed_deg_s: float = 0.0,
) -> bytes:
    return struct.pack(
        "<b6h",
        mode,
        int(round(roll_angle_deg * 100)),
        int(round(pitch_angle_deg * 100)),
        int(round(yaw_angle_deg * 100)),
        int(round(roll_speed_deg_s * 100)),
        int(round(pitch_speed_deg_s * 100)),
        int(round(yaw_speed_deg_s * 100)),
    )

def calc_header_checksum(version: int, length: int, command: int) -> int:
    return (version + length + command) & 0xFF

def calc_crc32_heq(data: bytes) -> int:
    nreg = 0xFFFFFFFF
    for b in data:
        nreg ^= b
        for _ in range(4):
            ntemp = crc32_table[(nreg >> 24) & 0xFF]
            nreg = ((nreg << 8) & 0xFFFFFFFF) ^ ntemp
    return nreg & 0xFFFFFFFF

def build_packet(command: int, data: bytes = b"") -> bytes:
    frame_header = 0xAE
    version = 0x01
    length = len(data)
    header_checksum = calc_header_checksum(version, length, command)

    pkt = bytearray([frame_header, version, length, command, header_checksum])
    pkt += data

    if length > 0:
        pkt += struct.pack("<I", calc_crc32_heq(data))

    return bytes(pkt)

class HEQParser:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> list[HEQFrame]:
        self.buffer.extend(chunk)
        frames: list[HEQFrame] = []

        while True:
            # Find frame header
            start = self.buffer.find(b"\xAE")
            if start == -1:
                # Keep only a small tail in case a future AE arrives
                if len(self.buffer) > 1024:
                    self.buffer = self.buffer[-16:]
                break

            # Drop junk before AE
            if start > 0:
                del self.buffer[:start]

            # Need at least minimal header
            if len(self.buffer) < 5:
                break

            frame_header = self.buffer[0]
            if frame_header != 0xAE:
                del self.buffer[0]
                continue

            version = self.buffer[1]
            length = self.buffer[2]
            command = self.buffer[3]
            header_checksum = self.buffer[4]

            total_len = 5 + length + (4 if length > 0 else 0)
            if len(self.buffer) < total_len:
                break

            raw = bytes(self.buffer[:total_len])
            data = raw[5:5 + length]

            header_ok = header_checksum == calc_header_checksum(version, length, command)

            crc_ok = True
            crc_value: Optional[int] = None
            if length > 0:
                crc_bytes = raw[5 + length:5 + length + 4]
                crc_value = struct.unpack("<I", crc_bytes)[0]
                crc_ok = crc_value == calc_crc32_heq(data)

            frames.append(
                HEQFrame(
                    version=version,
                    length=length,
                    command=command,
                    header_checksum=header_checksum,
                    data=data,
                    crc32=crc_value,
                    raw=raw,
                    header_ok=header_ok,
                    crc_ok=crc_ok,
                )
            )

            del self.buffer[:total_len]

        return frames

def decode_0x87(data: bytes) -> dict:
    if len(data) != 24:
        raise ValueError(f"0x87 V2 payload must be 24 bytes, got {len(data)}")

    vals = struct.unpack("<12h", data)
    names = [
        "imu_roll",
        "imu_pitch",
        "imu_yaw",
        "hall_roll",
        "hall_pitch",
        "hall_yaw",
        "hall_roll_rate",
        "hall_pitch_rate",
        "hall_yaw_rate",
        "imu_x_rate",
        "imu_y_rate",
        "imu_z_rate",
    ]

    return {name: value / 100.0 for name, value in zip(names, vals)}
