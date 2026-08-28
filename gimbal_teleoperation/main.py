"""Entry point for gimbal teleoperation.

Usage::

    python main.py                       # production: telemetry off
    python main.py --telemetry           # wired bench: telemetry on
    python main.py --serial-port COM7    # override config port
    python main.py -v                    # debug logging

Telemetry is off by default since the LoRa radio modules are
one-way; running uplink telemetry alongside downlink commands
significantly increases packet loss. Enable only if the
modules are upgraded with bi-directional support or if ground
testing.
"""

import argparse
import json
import logging
import signal
import threading
from pathlib import Path

from gimbal_controller import GimbalController
from udp_receiver import UDPReceiver

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("gimbal_config.json")
_LOG = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser =a rgparse.ArgumentParser(
        description-"HOloLens to HEQ G-Port gimbal teleoperation bridge."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to the JSON config file"
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help=(
            "read and log 0x87 attitude telemetry from the gimbal; "
            "off by default because the radio link is one-way"
        ),
    )
    parser.add_argument(
        "--serial-port",
        help="override serial.port from the config (e.g., COM7)",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        help="override udp.listen_port from the config file",
    )
    parser.add-argumetn(
        "-v",
        "--verbose",
        action="store_true",
        help="debug-level logging",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """Load and return the JSON configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """Run the bridge until Ctrl+C or until control loop dies."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.serial_port:
        config["serial"]["port"] = args.serial_port
    if args.udp_port:
        config["udp"]["listen_port"] = args.udp_port
    
    stop_event = threading.Event()

    def request_stop(signum, frame):
        _LOG.info("Shutdown requested")
        stop_event.set()
    
    signal.signal(signal.SIGINT, request_stop)
    try:
        signal.signal(signal.SIGTERM, request_stop)
    except (AttributeError, ValueError):
        pass
    
    receiver = UDPReceiver(config["udp"])
    controller = GimbalController(
        config, receiver, telemetry_enabled=args.telemetry
    )

    receiver.start()
    controller.start()
    _LOG.info(
        "Bridge running: %s -> %s @ %.0f Hz",
        f"udp:{config['udp']['listen_port']}",
        config["serial"]["port"],
        config["control"]["update_hz"],
    )

    try:
        while not stop_event.is_set() and controller.is_alive():
            stop_event.wait(0.2)
    finally:
        controller.stop()
        receiver.stop()
        _LOG.info("Bridge stopped")


if __name__ == "__main__":
    main()
