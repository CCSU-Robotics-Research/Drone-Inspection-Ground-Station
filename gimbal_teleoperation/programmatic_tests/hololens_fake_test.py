import math
import serial
import struct
import threading
import time

from heq_gimbal import HEQParser, build_packet, decode_0x87_v2

PORT = "COM4"
BAUD = 115200
UPDATE_HZ = 10.0
DT = 1.0 / UPDATE_HZ

# Gimbal limits
PITCH_MIN = -135.0
PITCH_MAX = 45.0
YAW_MIN = -135.0
YAW_MAX = 135.0
ROLL_MIN = -45.0
ROLL_MAX = 45.0

ser = serial.Serial(
    port=PORT,
    baudrate=BAUD,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=0.03,
)

parser = HEQParser()
stop_event = threading.Event()


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def build_0x85_payload(
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


def send_angle(roll=0.0, pitch=0.0, yaw=0.0):
    roll = clamp(roll, ROLL_MIN, ROLL_MAX)
    pitch = clamp(pitch, PITCH_MIN, PITCH_MAX)
    yaw = clamp(yaw, YAW_MIN, YAW_MAX)

    payload = build_0x85_payload(
        mode=2,
        roll_angle_deg=roll,
        pitch_angle_deg=pitch,
        yaw_angle_deg=yaw,
    )
    pkt = build_packet(0x85, payload)
    ser.write(pkt)
    ser.flush()


def telemetry_loop():
    last_print = 0.0
    while not stop_event.is_set():
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue

        frames = parser.feed(chunk)
        for frame in frames:
            if frame.command == 0x87 and frame.length == 24 and frame.header_ok and frame.crc_ok:
                telem = decode_0x87_v2(frame.data)
                now = time.time()
                if now - last_print > 0.10:
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
                    last_print = now


def dynamic_hololens_profile(duration_s: float):
    """
    Simulated live head motion with bigger roll/pitch and full-body feel.
    Uses mixed-frequency smooth motion plus periodic expressive bursts.
    """
    print(f"Running dynamic HoloLens-style profile for {duration_s:.1f} s at {UPDATE_HZ:.1f} Hz")

    start = time.time()
    next_tick = start
    last_cmd_print = 0.0

    while True:
        now = time.time()
        t = now - start
        if t >= duration_s:
            break

        # Yaw: broad scanning motion with medium-frequency overlay
        yaw = (
            85.0 * math.sin(2.0 * math.pi * 0.095 * t) +
            28.0 * math.sin(2.0 * math.pi * 0.29 * t + 0.8) +
            10.0 * math.sin(2.0 * math.pi * 0.75 * t + 1.9)
        )

        # Pitch: strongly asymmetric feeling. Base it below zero since your range is mostly downward.
        # This makes it visibly nod and dip.
        pitch = (
            -35.0 +
            45.0 * math.sin(2.0 * math.pi * 0.17 * t + 0.5) +
            20.0 * math.sin(2.0 * math.pi * 0.43 * t + 2.1)
        )

        # Roll: make it much larger and more energetic than before.
        roll = (
            28.0 * math.sin(2.0 * math.pi * 0.21 * t + 1.2) +
            12.0 * math.sin(2.0 * math.pi * 0.62 * t + 0.1) +
            5.0 * math.sin(2.0 * math.pi * 1.10 * t + 2.0)
        )

        # Periodic expressive "look" behaviors
        phase = t % 10.0

        # Lean and look up-right
        if 1.5 < phase < 2.6:
            yaw += 30.0
            pitch += 20.0
            roll += 18.0

        # Heavy look down-left
        elif 4.2 < phase < 5.4:
            yaw -= 40.0
            pitch -= 35.0
            roll -= 20.0

        # Strong side tilt with moderate yaw
        elif 7.0 < phase < 8.1:
            yaw += 18.0
            pitch -= 10.0
            roll += 24.0

        roll = clamp(roll, ROLL_MIN, ROLL_MAX)
        pitch = clamp(pitch, PITCH_MIN, PITCH_MAX)
        yaw = clamp(yaw, YAW_MIN, YAW_MAX)

        send_angle(roll=roll, pitch=pitch, yaw=yaw)

        if now - last_cmd_print > 0.25:
            print(f"[CMD] target r/p/y = {roll:.2f}, {pitch:.2f}, {yaw:.2f} deg")
            last_cmd_print = now

        next_tick += DT
        sleep_time = next_tick - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("Profile complete.")


def return_to_center(wait_s=3.0):
    print("Returning to center...")
    payload = build_0x85_payload(mode=3)
    pkt = build_packet(0x85, payload)
    ser.write(pkt)
    ser.flush()
    time.sleep(wait_s)


def main():
    tel_thread = threading.Thread(target=telemetry_loop, daemon=True)
    tel_thread.start()

    try:
        time.sleep(1.0)

        return_to_center(wait_s=2.0)
        dynamic_hololens_profile(duration_s=30.0)
        return_to_center(wait_s=3.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stop_event.set()
        time.sleep(0.2)
        ser.close()


if __name__ == "__main__":
    main()