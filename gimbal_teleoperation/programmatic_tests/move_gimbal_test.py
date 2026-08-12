import serial
import struct
import threading
import time

from heq_gimbal import HEQParser, build_packet, decode_0x87_v2

PORT = "COM4"
BAUD = 115200

ser = serial.Serial(
    port=PORT,
    baudrate=BAUD,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=0.1,
)

parser = HEQParser()
stop_event = threading.Event()


def build_0x85_payload(
    mode: int,
    roll_angle_deg: float = 0.0,
    pitch_angle_deg: float = 0.0,
    yaw_angle_deg: float = 0.0,
    roll_speed_deg_s: float = 0.0,
    pitch_speed_deg_s: float = 0.0,
    yaw_speed_deg_s: float = 0.0,
) -> bytes:
    """
    0x85 payload layout from HEQ docs:
      byte 0      : mode (int8)
      bytes 1-2   : roll angle   (int16, 0.01 deg)
      bytes 3-4   : pitch angle  (int16, 0.01 deg)
      bytes 5-6   : yaw angle    (int16, 0.01 deg)
      bytes 7-8   : roll speed   (int16, 0.01 deg/s)
      bytes 9-10  : pitch speed  (int16, 0.01 deg/s)
      bytes 11-12 : yaw speed    (int16, 0.01 deg/s)
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


def send_0x85(
    mode: int,
    roll_angle_deg: float = 0.0,
    pitch_angle_deg: float = 0.0,
    yaw_angle_deg: float = 0.0,
    roll_speed_deg_s: float = 0.0,
    pitch_speed_deg_s: float = 0.0,
    yaw_speed_deg_s: float = 0.0,
) -> None:
    payload = build_0x85_payload(
        mode=mode,
        roll_angle_deg=roll_angle_deg,
        pitch_angle_deg=pitch_angle_deg,
        yaw_angle_deg=yaw_angle_deg,
        roll_speed_deg_s=roll_speed_deg_s,
        pitch_speed_deg_s=pitch_speed_deg_s,
        yaw_speed_deg_s=yaw_speed_deg_s,
    )
    pkt = build_packet(0x85, payload)
    ser.write(pkt)
    ser.flush()

    print("TX:", pkt.hex(" ").upper())
    print(
        f"  mode={mode} | "
        f"angles(r/p/y)=({roll_angle_deg}, {pitch_angle_deg}, {yaw_angle_deg}) deg | "
        f"speeds(r/p/y)=({roll_speed_deg_s}, {pitch_speed_deg_s}, {yaw_speed_deg_s}) deg/s"
    )


def telemetry_loop() -> None:
    while not stop_event.is_set():
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue

        frames = parser.feed(chunk)
        for frame in frames:
            if frame.command == 0x87 and frame.length == 24 and frame.header_ok and frame.crc_ok:
                telem = decode_0x87_v2(frame.data)
                print(
                    f"[TEL] IMU r/p/y = "
                    f"{telem['imu_roll']:.2f}, "
                    f"{telem['imu_pitch']:.2f}, "
                    f"{telem['imu_yaw']:.2f} deg | "
                    f"Hall r/p/y = "
                    f"{telem['hall_roll']:.2f}, "
                    f"{telem['hall_pitch']:.2f}, "
                    f"{telem['hall_yaw']:.2f} deg"
                )


def test_return_to_center(wait_s: float = 3.0) -> None:
    print("\n=== TEST: Return to center (mode 3) ===")
    send_0x85(mode=3)
    time.sleep(wait_s)


def test_yaw_speed(speed_deg_s: float, duration_s: float = 2.0) -> None:
    print(f"\n=== TEST: Yaw speed {speed_deg_s:+.1f} deg/s for {duration_s:.1f}s (mode 1) ===")
    start = time.time()
    while time.time() - start < duration_s:
        send_0x85(mode=1, yaw_speed_deg_s=speed_deg_s)
        time.sleep(0.1)  # 10 Hz, per HEQ note for speed mode

    print("\nStopping speed command")
    send_0x85(mode=1, yaw_speed_deg_s=0.0)
    time.sleep(2.0)


def test_yaw_angle(angle_deg: float, wait_s: float = 4.0) -> None:
    print(f"\n=== TEST: Yaw angle {angle_deg:+.1f} deg (mode 2) ===")
    send_0x85(mode=2, yaw_angle_deg=angle_deg)
    time.sleep(wait_s)


def test_pitch_angle(angle_deg: float, wait_s: float = 4.0) -> None:
    print(f"\n=== TEST: Pitch angle {angle_deg:+.1f} deg (mode 2) ===")
    send_0x85(mode=2, pitch_angle_deg=angle_deg)
    time.sleep(wait_s)


def main():
    tel_thread = threading.Thread(target=telemetry_loop, daemon=True)
    tel_thread.start()

    try:
        print("Starting noticeable gimbal tests...")
        time.sleep(1.0)

        test_return_to_center()

        test_yaw_speed(+60.0, duration_s=2.0)
        test_yaw_speed(-60.0, duration_s=2.0)

        test_yaw_angle(+45.0)
        test_yaw_angle(-45.0)

        test_pitch_angle(+15.0)
        test_pitch_angle(-15.0)

        test_return_to_center()

        print("\nAll tests complete.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        stop_event.set()
        time.sleep(0.2)
        ser.close()


if __name__ == "__main__":
    main()