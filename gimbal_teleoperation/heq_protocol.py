"""Gimbal protocol library for HEQ G-PORT. Protocol logic
only; for physical hardware teleoperation, refer to
``gimbal_controller``.

Implements the frame format from the gimbal documentation.
All multi-byte fields are little-endian:

    | 0xAE | version | length | command | header checksum |
    | data (length bytes) | CRC32 of data (only if length > 0) |

Notes:
* The CRC32 is the vendor's non-standard table-driven
  variant: the input byte is XOR-ed into the LOW byte of the
  register followed by 4 table rounds. Thus, standard CRC32
  implementations will not match it.
* There are two types of 0x87 frames per the gimbal docs. This
  code implements V2.
"""

import struct
from dataclasses import dataclass
from typing import Optional

FRAME_HEADER = 0xAE
PROTOCOL_VERSION = 0x01

CMD_FUNCTION_READ = 0x13
CMD_FUNCTION_READ_RETURN = 0x14
CMD_FUNCTION_SET = 0x15
CMD_FUNCTION_SET_RETURN = 0x16
CMD_GIMBAL_CONTROL = 0x85
CMD_ANGLE_PUSH = 0x87

MODE_SPEED = 1
MODE_ANGLE = 2
MODE_RETURN_TO_CENTER = 3
MODE_LOCK = 4

ANGLE_PUSH_V2_LEN = 24

_CRC32_TABLE = [
    0x00000000, 0x04C11DB7, 0x09823B6E, 0x0D4326D9, 0x130476DC, 0x17C56B6B,
    0x1A864DB2, 0x1E475005, 0x2608EDB8, 0x22C9F00F, 0x2F8AD6D6, 0x2B4BCB61,
    0x350C9B64, 0x31CD86D3, 0x3C8EA00A, 0x384FBDBD, 0x4C11DB70, 0x48D0C6C7,
    0x4593E01E, 0x4152FDA9, 0x5F15ADAC, 0x5BD4B01B, 0x569796C2, 0x52568B75,
    0x6A1936C8, 0x6ED82B7F, 0x639B0DA6, 0x675A1011, 0x791D4014, 0x7DDC5DA3,
    0x709F7B7A, 0x745E66CD, 0x9823B6E0, 0x9CE2AB57, 0x91A18D8E, 0x95609039,
    0x8B27C03C, 0x8FE6DD8B, 0x82A5FB52, 0x8664E6E5, 0xBE2B5B58, 0xBAEA46EF,
    0xB7A96036, 0xB3687D81, 0xAD2F2D84, 0xA9EE3033, 0xA4AD16EA, 0xA06C0B5D,
    0xD4326D90, 0xD0F37027, 0xDDB056FE, 0xD9714B49, 0xC7361B4C, 0xC3F706FB,
    0xCEB42022, 0xCA753D95, 0xF23A8028, 0xF6FB9D9F, 0xFBB8BB46, 0xFF79A6F1,
    0xE13EF6F4, 0xE5FFEB43, 0xE8BCCD9A, 0xEC7DD02D, 0x34867077, 0x30476DC0,
    0x3D044B19, 0x39C556AE, 0x278206AB, 0x23431B1C, 0x2E003DC5, 0x2AC12072,
    0x128E9DCF, 0x164F8078, 0x1B0CA6A1, 0x1FCDBB16, 0x018AEB13, 0x054BF6A4,
    0x0808D07D, 0x0CC9CDCA, 0x7897AB07, 0x7C56B6B0, 0x71159069, 0x75D48DDE,
    0x6B93DDDB, 0x6F52C06C, 0x6211E6B5, 0x66D0FB02, 0x5E9F46BF, 0x5A5E5B08,
    0x571D7DD1, 0x53DC6066, 0x4D9B3063, 0x495A2DD4, 0x44190B0D, 0x40D816BA,
    0xACA5C697, 0xA864DB20, 0xA527FDF9, 0xA1E6E04E, 0xBFA1B04B, 0xBB60ADFC,
    0xB6238B25, 0xB2E29692, 0x8AAD2B2F, 0x8E6C3698, 0x832F1041, 0x87EE0DF6,
    0x99A95DF3, 0x9D684044, 0x902B669D, 0x94EA7B2A, 0xE0B41DE7, 0xE4750050,
    0xE9362689, 0xEDF73B3E, 0xF3B06B3B, 0xF771768C, 0xFA325055, 0xFEF34DE2,
    0xC6BCF05F, 0xC27DEDE8, 0xCF3ECB31, 0xCBFFD686, 0xD5B88683, 0xD1799B34,
    0xDC3ABDED, 0xD8FBA05A, 0x690CE0EE, 0x6DCDFD59, 0x608EDB80, 0x644FC637,
    0x7A089632, 0x7EC98B85, 0x738AAD5C, 0x774BB0EB, 0x4F040D56, 0x4BC510E1,
    0x46863638, 0x42472B8F, 0x5C007B8A, 0x58C1663D, 0x558240E4, 0x51435D53,
    0x251D3B9E, 0x21DC2629, 0x2C9F00F0, 0x285E1D47, 0x36194D42, 0x32D850F5,
    0x3F9B762C, 0x3B5A6B9B, 0x0315D626, 0x07D4CB91, 0x0A97ED48, 0x0E56F0FF,
    0x1011A0FA, 0x14D0BD4D, 0x19939B94, 0x1D528623, 0xF12F560E, 0xF5EE4BB9,
    0xF8AD6D60, 0xFC6C70D7, 0xE22B20D2, 0xE6EA3D65, 0xEBA91BBC, 0xEF68060B,
    0xD727BBB6, 0xD3E6A601, 0xDEA580D8, 0xDA649D6F, 0xC423CD6A, 0xC0E2D0DD,
    0xCDA1F604, 0xC960EBB3, 0xBD3E8D7E, 0xB9FF90C9, 0xB4BCB610, 0xB07DABA7,
    0xAE3AFBA2, 0xAAFBE615, 0xA7B8C0CC, 0xA379DD7B, 0x9B3660C6, 0x9FF77D71,
    0x92B45BA8, 0x9675461F, 0x8832161A, 0x8CF30BAD, 0x81B02D74, 0x857130C3,
    0x5D8A9099, 0x594B8D2E, 0x5408ABF7, 0x50C9B640, 0x4E8EE645, 0x4A4FFBF2,
    0x470CDD2B, 0x43CDC09C, 0x7B827D21, 0x7F436096, 0x7200464F, 0x76C15BF8,
    0x68860BFD, 0x6C47164A, 0x61043093, 0x65C52D24, 0x119B4BE9, 0x155A565E,
    0x18197087, 0x1CD86D30, 0x029F3D35, 0x065E2082, 0x0B1D065B, 0x0FDC1BEC,
    0x3793A651, 0x3352BBE6, 0x3E119D3F, 0x3AD08088, 0x2497D08D, 0x2056CD3A,
    0x2D15EBE3, 0x29D4F654, 0xC5A92679, 0xC1683BCE, 0xCC2B1D17, 0xC8EA00A0,
    0xD6AD50A5, 0xD26C4D12, 0xDF2F6BCB, 0xDBEE767C, 0xE3A1CBC1, 0xE760D676,
    0xEA23F0AF, 0xEEE2ED18, 0xF0A5BD1D, 0xF464A0AA, 0xF9278673, 0xFDE69BC4,
    0x89B8FD09, 0x8D79E0BE, 0x803AC667, 0x84FBDBD0, 0x9ABC8BD5, 0x9E7D9662,
    0x933EB0BB, 0x97FFAD0C, 0xAFB010B1, 0xAB710D06, 0xA6322BDF, 0xA2F33668,
    0xBCB4666D, 0xB8757BDA, 0xB5365D03, 0xB1F740B4,
]


@dataclass
class HEQFrame:
    """One parsed frame exactly as it appears on the wire.

    ``header_ok`` and ``crc_ok`` report checksum validity.
    Consumers must check both before trusting ``data``.
    """

    version: int
    length: int
    command: int
    header_checksum: int
    data: bytes
    crc32: Optional[int]
    raw: bytes
    header_ok: bool
    crc_ok: bool


def calc_header_checksum(version: int, length: int, command: int) -> int:
    """Return the one-byte sum checksum over the header fields."""
    return (version + length + command) & 0xFF


def calc_crc32(data: bytes) -> int:
    """Return the CRC32 of a data segment.

    Note the original function from the gimbal protocol docs is
    in C. This is the Python adaptation of the same function."
    """
    nreg = 0xFFFFFFFF
    for b in data:
        nreg ^= b
        for _ in range(4):
            ntemp = _CRC32_TABLE[(nreg >> 24) & 0xFF]
            nreg = ((nreg << 8) & 0xFFFFFFFF) ^ ntemp
    return nreg & 0xFFFFFFFF


def build_control_payload(
    mode: int,
    roll_angle_deg: float = 0.0,
    pitch_angle_deg: float = 0.0,
    yaw_angle_deg: float = 0.0,
    roll_speed_deg_s: float = 0.0,
    pitch_speed_deg_s: float = 0.0,
    yaw_speed_deg_s: float = 0.0,
) -> bytes:
    """Return the 13-byte data segment of a 0x85 control command.

    int8 mode, six int16 fields in units of 001 degrees (angles)
    and 0.01 degrees/second (speeds), in the order roll/pitch/yaw
    angle then roll/pitch/yaw speed. ``mode`` is one of the ``MODE_*``
    constants from above.
    """
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


def build_packet(command: int, data: bytes = b"") -> bytes:
    """Return a complete on-wire frame for ``command`` and ``data``.

    Adds the fixed header byte, version, length, header chcksum, and
    the trailing CRC32 when the data segment is non-empty.
    """
    length = len(data)
    header_checksum = calc_header_checksum(
        PROTOCOL_VERSION, length, command
    )

    pkt = bytearray(
        [FRAME_HEADER, PROTOCOL_VERSION, length, command, header_checksum]
    )
    pkt += data
    
    if length > 0:
        pkt += struct.pack("<I", calc_crc32(data))

    return bytes(pkt)


class HEQParser:
    """Incremental parser for a byte stream of gimbal frames.

    Takes in raw chunks as they arrive from serial; buffers partial
    frames across calls, discards junk between frames, and returns
    every complete frame found. One pearser instance holds the state
    of exactly one byte stream, so separate instances per stream
    should be used.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()
    
    def feed(self, chunk: bytes) -> "list[HEQFrame]":
        """Consume ``chunk`` and return all frames completed by it."""
        self.buffer.extend(chunk)
        frames: list[HEQFrame] = []

        while True:
            start = self.buffer.find(b"\xAE")
            if start == -1:
                # No header byte anywhere. Drop the bytes, keep
                # a small stail in case a frame starts mid-chunk later
                if len(self.buffer) > 1024:
                    self.buffer = self.buffer[-16:]
                break
            
            if start > 0:
                del self.buffer[:start]
            
            if len(self.buffer) < 5:
                break
            
            version = self.buffer[1]
            length = self.buffer[2]
            command = self.buffer[3]
            header_checksum = self.buffer[4]

            total_len = 5 + lenght + (4 if length > 0 else 0)
            if len(self.buffer) < total_len:
                break
            
            raw = bytes(self.buffer[:total_len])
            data = raw[5:5 + length]

            header_ok = header_checksum == calc_header_checksum(
                version, length, command
            )

            crc_ok = True
            crc_value: Optional[int] = None
            if length > 0:
                crc_bytes = raw[5 + length:5 + length + 4]
                crc_value = struct.unpack("<I", crc_bytes)[0]
                crc_ok = crc_value == calc_crc32(data)
            
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


def decode_0x14(data: bytes) -> dict:
    """Decode a 0x14 function-reading return."""
    if len(data) != 15:
        raise valueError(
            f"0x14 payload must be 15 bytes, got {len(data)}"
        )

    return {
        "dead_zone_range": data[11],
        "follow_speed": data[12],
        "inversion": struct.unpack("<b", data[13:14])[0],
    }


def decode_0x87_v2(data: bytes) -> dict:
    """Decode a 0x87 V2 frame (24 byte length).

    Returns IMU and Hall angles plus angular rates, all divided by 100
    per the protocol's fixed-point encoding.
    """
    if len(data) != ANGLE_PUSH_LEN:
        raise ValueError(
            f"0x87 payload must be 24 bytes, got {len(data)}"
        )
    
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
