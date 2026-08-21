import json
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import serial

from heq_gimbal import HEQParser, build_packet, decode_0x87_v2


CONFIG_PATH = Path(__file__).with_name("gimbal_config.json")

running = True


def handle_sigint(signum, frame):
    global running
    running = False


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def apply_deadband(x, deadband):
    return 0.0 if abs(x) < deadband else x


def smooth(prev, target, alpha):
    return prev + alpha * (target - prev)


def limit_step(prev, target, max_step):
    delta = target - prev
    delta = clamp(delta, -max_step, max_step)
    return prev + delta


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


class GimbalBridge:
    def __init__(self, config):
        self.config = config
        self.dry_run = config["control"]["dry_run"]

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.last_udp_time = 0.0
        self.latest_head = None
        self.lock = threading.Lock()

        self.ser = None
        self.parser = HEQParser()

    def open_serial(self):
        if self.dry_run:
            print("[DRY RUN] Serial not opened. No gimbal motion will occur.")
            return

        s = self.config["serial"]
        self.ser = serial.Serial(
            port=s["port"],
            baudrate=s["baud"],
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.03,
        )
        print(f"[SERIAL] Opened {s['port']} @ {s['baud']}")

    def close_serial(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def parse_udp_pose(self, message):
        parts = message.strip().split(",")
        if len(parts) != 3:
            raise ValueError(f"Expected CSV roll,pitch,yaw but got: {message!r}")

        head_roll = float(parts[0])
        head_pitch = float(parts[1])
        head_yaw = float(parts[2])
        return head_roll, head_pitch, head_yaw

    def update_latest_head_pose(self, pose):
        with self.lock:
            self.latest_head = pose
            self.last_udp_time = time.time()

    def get_latest_head_pose(self):
        with self.lock:
            return self.latest_head, self.last_udp_time

    def map_head_to_gimbal(self, head_roll, head_pitch, head_yaw):
        m = self.config["mapping"]
        limits = self.config["limits"]
        safety = self.config["safety"]

        roll = head_roll if m["enable_roll"] else 0.0
        pitch = head_pitch if m["enable_pitch"] else 0.0
        yaw = head_yaw if m["enable_yaw"] else 0.0

        if m["invert_roll"]:
            roll = -roll
        if m["invert_pitch"]:
            pitch = -pitch
        if m["invert_yaw"]:
            yaw = -yaw

        roll = roll * m["roll_gain"] + m["roll_offset"]
        pitch = pitch * m["pitch_gain"] + m["pitch_offset"]
        yaw = yaw * m["yaw_gain"] + m["yaw_offset"]

        roll = apply_deadband(roll, safety["deadband_deg"])
        pitch = apply_deadband(pitch, safety["deadband_deg"])
        yaw = apply_deadband(yaw, safety["deadband_deg"])

        roll = clamp(roll, limits["roll"][0], limits["roll"][1])
        pitch = clamp(pitch, limits["pitch"][0], limits["pitch"][1])
        yaw = clamp(yaw, limits["yaw"][0], limits["yaw"][1])

        return roll, pitch, yaw

    def send_angle(self, roll, pitch, yaw):
        payload = build_0x85_payload(
            mode=2,
            roll_angle_deg=roll,
            pitch_angle_deg=pitch,
            yaw_angle_deg=yaw,
        )
        pkt = build_packet(0x85, payload)

        if self.dry_run:
            return

        self.ser.write(pkt)
        self.ser.flush()

    def return_to_center(self):
        print("[SAFE] Returning gimbal to center...")

        payload = build_0x85_payload(mode=3)
        pkt = build_packet(0x85, payload)

        if not self.dry_run and self.ser is not None:
            self.ser.write(pkt)
            self.ser.flush()

    def telemetry_loop(self):
        if self.dry_run:
            return

        last_print = 0.0

        while running:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if not chunk:
                continue

            frames = self.parser.feed(chunk)
            for frame in frames:
                if frame.command == 0x87 and frame.length == 24 and frame.header_ok and frame.crc_ok:
                    telem = decode_0x87_v2(frame.data)
                    now = time.time()

                    if now - last_print > 0.20:
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

    def udp_loop(self):
        u = self.config["udp"]

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((u["ip"], u["port"]))
        sock.settimeout(u["timeout_s"])

        print(f"[UDP] Listening on {u['ip']}:{u['port']}")

        try:
            while running:
                try:
                    data, addr = sock.recvfrom(1024)
                    message = data.decode("utf-8", errors="replace")
                    pose = self.parse_udp_pose(message)
                    self.update_latest_head_pose(pose)

                except socket.timeout:
                    continue
                except ValueError as e:
                    print(f"[UDP] Bad packet: {e}")

        finally:
            sock.close()

    def control_loop(self):
        hz = self.config["control"]["update_hz"]
        dt = 1.0 / hz
        next_tick = time.time()

        safety = self.config["safety"]
        timeout_s = safety["command_timeout_s"]
        alpha = safety["smoothing_alpha"]
        max_step = safety["max_step_deg_per_update"]

        last_print = 0.0

        while running:
            now = time.time()
            pose, last_udp_time = self.get_latest_head_pose()

            if pose is None or now - last_udp_time > timeout_s:
                target_roll, target_pitch, target_yaw = 0.0, 0.0, 0.0
                status = "NO RECENT UDP -> CENTER"
            else:
                head_roll, head_pitch, head_yaw = pose
                target_roll, target_pitch, target_yaw = self.map_head_to_gimbal(
                    head_roll, head_pitch, head_yaw
                )
                status = "LIVE"

            target_roll = smooth(self.roll, target_roll, alpha)
            target_pitch = smooth(self.pitch, target_pitch, alpha)
            target_yaw = smooth(self.yaw, target_yaw, alpha)

            self.roll = limit_step(self.roll, target_roll, max_step)
            self.pitch = limit_step(self.pitch, target_pitch, max_step)
            self.yaw = limit_step(self.yaw, target_yaw, max_step)

            self.send_angle(self.roll, self.pitch, self.yaw)

            if now - last_print > 0.20:
                if pose is None:
                    print(
                        f"[CMD] {status} | target r/p/y = "
                        f"{self.roll:.2f}, {self.pitch:.2f}, {self.yaw:.2f}"
                    )
                else:
                    print(
                        f"[CMD] {status} | head r/p/y = "
                        f"{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f} | "
                        f"gimbal target r/p/y = "
                        f"{self.roll:.2f}, {self.pitch:.2f}, {self.yaw:.2f}"
                    )
                last_print = now

            next_tick += dt
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def main():
    global running

    signal.signal(signal.SIGINT, handle_sigint)

    config = load_config()
    bridge = GimbalBridge(config)

    bridge.open_serial()

    udp_thread = threading.Thread(target=bridge.udp_loop, daemon=True)
    telemetry_thread = threading.Thread(target=bridge.telemetry_loop, daemon=True)

    udp_thread.start()
    telemetry_thread.start()

    try:
        print("[BRIDGE] Starting control loop.")
        print("[BRIDGE] Ctrl+C to stop.")
        bridge.control_loop()

    finally:
        running = False

        if config["control"]["return_to_center_on_exit"]:
            bridge.return_to_center()
            time.sleep(0.5)

        bridge.close_serial()
        print("[BRIDGE] Stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()