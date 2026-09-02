"""Test script to move the gimbal directly without the full
teleoperation path.

Bypasses UDP and the controller to validate the protocol layer and
the physical link: return-to-center, speed mode, and angle mode, with
live telemetry printed from the gimbal's 0x87 pushes.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import serial

_REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_DIR))

from heq_protocol import (  # noqa: E402
    ANGLE_PUSH_V2_LEN,
    CMD_ANGLE_PUSH,
    CMD_GIMBAL_CONTROL,
    HEQParser,
    MODE_ANGLE,
    MODE_RETURN_TO_CENTER,
    MODE_SPEED,
    build_control_payload,
    build_packet,
    decode_0x87_v2,
)

_DEFAULT_CONFIG_PATH = _REPO_DIR / "gimbal_config.json"

stop_event = threading.Event()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-serial gimbal motion tests."
    )
    parser.add_argument(
        "--port",
        help="serial port (default: serial.port from config)",
    )
    return parser.parse_args()


def open_serial(port: str, baud: int) -> serial.Serial:
    return serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1,
    )


def send_control(ser: serial.Serial, mode: int, **kwargs) -> None:
    """Build, transmit, and echo one 0x85 control command."""
    payload = build_control_payload(mode=mode, **kwargs)
    pkt = build_packet(CMD_GIMBAL_CONTROL, payload)
    ser.write(pkt)
    ser.flush()
    print("TX:", pkt.hex(" ").upper())


def telemetry_loop(ser: serial.Serial) -> None:
    parser = HEQParser()
    last_print = 0.0

    while not stop_event.is_set():
        try:
            chunk = ser.read(ser.in_waiting or 1)
        except (serial.SerialException, OSError):
            break
        if not chunk:
            continue

        for frame in parser.feed(chunk):
            valid = (
                frame.command == CMD_ANGLE_PUSH
                and frame.length == ANGLE_PUSH_V2_LEN
                and frame.header_ok
                and frame.crc_ok
            )
            if not valid:
                continue

            telem = decode_0x87_v2(frame.data)
            now = time.monotonic()
            if now - last_print > 0.10:
                print(
                    f"[TEL] IMU r/p/y = "
                    f"{telem['imu_roll']:.2f}, "
                    f"{telem['imu_pitch']:.2f}, "
                    f"{telem['imu_yaw']:.2f} | "
                    f"Hall r/p/y = "
                    f"{telem['hall_roll']:.2f}, "
                    f"{telem['hall_pitch']:.2f}, "
                    f"{telem['hall_yaw']:.2f}"
                )
                last_print = now


def test_return_to_center(ser: serial.Serial, wait_s: float = 3.0) -> None:
    print("\n=== Return to center (mode 3) ===")
    send_control(ser, MODE_RETURN_TO_CENTER)
    time.sleep(wait_s)


def test_yaw_speed(
    ser: serial.Serial, speed_deg_s: float, duration_s: float = 2.0
) -> None:
    print(
        f"\n=== Yaw speed {speed_deg_s:+.1f} deg/s "
        f"for {duration_s:.1f}s (mode 1) ==="
    )
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        send_control(ser, MODE_SPEED, yaw_speed_deg_s=speed_deg_s)
        time.sleep(0.1)  # 10 Hz refresh per the HEQ speed-mode note.

    print("Stopping speed command")
    send_control(ser, MODE_SPEED, yaw_speed_deg_s=0.0)
    time.sleep(2.0)


def test_yaw_angle(
    ser: serial.Serial, angle_deg: float, wait_s: float = 4.0
) -> None:
    print(f"\n=== Yaw angle {angle_deg:+.1f} deg (mode 2) ===")
    send_control(ser, MODE_ANGLE, yaw_angle_deg=angle_deg)
    time.sleep(wait_s)


def test_pitch_angle(
    ser: serial.Serial, angle_deg: float, wait_s: float = 4.0
) -> None:
    print(f"\n=== Pitch angle {angle_deg:+.1f} deg (mode 2) ===")
    send_control(ser, MODE_ANGLE, pitch_angle_deg=angle_deg)
    time.sleep(wait_s)


def main() -> None:
    args = parse_args()

    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        serial_cfg = json.load(f)["serial"]

    port = args.port or serial_cfg["port"]
    ser = open_serial(port, int(serial_cfg["baud"]))
    print(f"Opened {port} @ {serial_cfg['baud']}")

    tel_thread = threading.Thread(
        target=telemetry_loop, args=(ser,), daemon=True
    )
    tel_thread.start()

    try:
        time.sleep(1.0)

        test_return_to_center(ser)

        test_yaw_speed(ser, +60.0)
        test_yaw_speed(ser, -60.0)

        test_yaw_angle(ser, +45.0)
        test_yaw_angle(ser, -45.0)

        test_pitch_angle(ser, +15.0)
        test_pitch_angle(ser, -15.0)

        test_return_to_center(ser)

        print("\nAll tests complete.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stop_event.set()
        time.sleep(0.2)
        ser.close()


if __name__ == "__main__":
    main()
