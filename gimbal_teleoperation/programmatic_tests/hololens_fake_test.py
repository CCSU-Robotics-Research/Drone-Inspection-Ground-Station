"""Bench tool to impersonate the HoloLens over UDP and test the
production path. Sends ``"roll,pitch,yaw"`` datagrams with a dynamic
head-motion profile to the gimbal using the teleoperation bridge.
"""

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_DIR))

_DEFAULT_CONFIG_PATH = _REPO_DIR / "gimbal_config.json"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def head_motion_profile(t: float) -> "tuple":
    """Return simulated head (roll, pitch, yaw) at time ``t`` seconds.

    Mixed-frequency smooth motion plus periodic expressive "look"
    bursts, tuned to feel like live full-body head movement.
    """
    yaw = (
        85.0 * math.sin(2.0 * math.pi * 0.095 * t)
        + 28.0 * math.sin(2.0 * math.pi * 0.29 * t + 0.8)
        + 10.0 * math.sin(2.0 * math.pi * 0.75 * t + 1.9)
    )

    # Biased downward: the gimbal's pitch range is mostly below zero.
    pitch = (
        -35.0
        + 45.0 * math.sin(2.0 * math.pi * 0.17 * t + 0.5)
        + 20.0 * math.sin(2.0 * math.pi * 0.43 * t + 2.1)
    )

    roll = (
        28.0 * math.sin(2.0 * math.pi * 0.21 * t + 1.2)
        + 12.0 * math.sin(2.0 * math.pi * 0.62 * t + 0.1)
        + 5.0 * math.sin(2.0 * math.pi * 1.10 * t + 2.0)
    )

    phase = t % 10.0
    if 1.5 < phase < 2.6:
        # Lean and look up-right.
        yaw += 30.0
        pitch += 20.0
        roll += 18.0
    elif 4.2 < phase < 5.4:
        # Heavy look down-left.
        yaw -= 40.0
        pitch -= 35.0
        roll -= 20.0
    elif 7.0 < phase < 8.1:
        # Strong side tilt with moderate yaw.
        yaw += 18.0
        pitch -= 10.0
        roll += 24.0

    return roll, pitch, yaw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send fake HoloLens head poses to the bridge."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bridge host. Default is 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="bridge UDP port. Default is udp.listen_port from config",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="send rate in Hz, matching the Unity sender. Default is 30",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="seconds to run. Default is 30",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.port is None:
        with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            args.port = int(json.load(f)["udp"]["listen_port"])

    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        limits = json.load(f)["limits"]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = (args.host, args.port)
    period = 1.0 / args.rate

    print(
        f"Sending fake head poses to {args.host}:{args.port} "
        f"at {args.rate:.0f} Hz for {args.duration:.0f} s"
    )

    start = time.monotonic()
    next_tick = start
    last_print = 0.0

    try:
        while True:
            now = time.monotonic()
            t = now - start
            if t >= args.duration:
                break

            roll, pitch, yaw = head_motion_profile(t)
            roll = _clamp(roll, limits["roll"][0], limits["roll"][1])
            pitch = _clamp(pitch, limits["pitch"][0], limits["pitch"][1])
            yaw = _clamp(yaw, limits["yaw"][0], limits["yaw"][1])

            message = f"{roll:.2f},{pitch:.2f},{yaw:.2f}"
            sock.sendto(message.encode("utf-8"), address)

            if now - last_print > 0.25:
                print(f"[TX] {message}")
                last_print = now

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()
        print("Done. The bridge should now hold, then glide to center.")


if __name__ == "__main__":
    main()
