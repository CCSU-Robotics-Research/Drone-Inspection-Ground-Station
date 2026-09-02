"""Reads the live telemetry from the gimbal.

Prints each parsed frame (command, length, checksum status, raw hex)
and decodes valid V2 0x87 attitude pushes.  Useful for verifying
wiring, baud, and radio-link health.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import serial

_REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_DIR))

from heq_protocol import (  # noqa: E402
    ANGLE_PUSH_V2_LEN,
    CMD_ANGLE_PUSH,
    HEQParser,
    decode_0x87_v2,
)

_DEFAULT_CONFIG_PATH = _REPO_DIR / "gimbal_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print raw HEQ frames from the gimbal."
    )
    parser.add_argument(
        "--port",
        help="serial port (default: serial.port from config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also append everything printed to this file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        serial_cfg = json.load(f)["serial"]

    port = args.port or serial_cfg["port"]
    ser = serial.Serial(
        port=port,
        baudrate=int(serial_cfg["baud"]),
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1,
    )

    log_file = None
    if args.output is not None:
        log_file = open(args.output, "a", encoding="utf-8")

    def emit(line: str) -> None:
        print(line)
        if log_file is not None:
            log_file.write(line + "\n")

    parser = HEQParser()
    emit(f"# Listening on {port} @ {serial_cfg['baud']} "
         f"({time.strftime('%Y-%m-%d %H:%M:%S')})")

    try:
        while True:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue

            for frame in parser.feed(chunk):
                emit(
                    f"CMD=0x{frame.command:02X} LEN={frame.length} "
                    f"hdr_ok={frame.header_ok} crc_ok={frame.crc_ok}"
                )
                emit(f"  RAW: {frame.raw.hex(' ').upper()}")

                decodable = (
                    frame.command == CMD_ANGLE_PUSH
                    and frame.length == ANGLE_PUSH_V2_LEN
                    and frame.header_ok
                    and frame.crc_ok
                )
                if decodable:
                    telem = decode_0x87_v2(frame.data)
                    emit(
                        f"  IMU r/p/y = "
                        f"{telem['imu_roll']:.2f}, "
                        f"{telem['imu_pitch']:.2f}, "
                        f"{telem['imu_yaw']:.2f}"
                    )
                    emit(
                        f"  Hall r/p/y = "
                        f"{telem['hall_roll']:.2f}, "
                        f"{telem['hall_pitch']:.2f}, "
                        f"{telem['hall_yaw']:.2f}"
                    )
                else:
                    emit("  (frame not decoded)")
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        ser.close()
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
