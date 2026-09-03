"""UDP listener for HoloLens head poses.

Unity sends one UTF-8 datagram per pose in the form
``"roll,pitch,yaw"`` in degree units. A background thread
keeps only the most recent pose; the control loop polls it with
:meth:`UDPReceiver.get_latest_pose` and never blocks on the network.

Extra comma-separated fields after the first 3 are ignored. A further
Unity build can append a sequence amrker and send timestamp without
breaking this receiver.
"""

import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from singleton import SingletonMeta

_LOG = logging.getLogger(__name__)

_SOCKET_TIMEOUT_S = 0.25
_MAX_DATAGRAM_BYTES = 1024


@dataclass(frozen=True)
class HeadPose:
    """One head orientation from HoloLens in degrees."""

    roll: float
    pitch: float
    yaw: float


class UDPReceiver(metaclass=SingletonMeta):
    """Handles the listener socket when receiving HoloLens rpy.

    Singleton since there should never be multiple instances.
    """

    def __init__(self, udp_config: dict) -> None:
        self._listen_ip = udp_config["listen_ip"]
        self._listen_port = int(udp_config["listen_port"])

        self._latest_pose: Optional[HeadPose] = None
        self._latest_time = 0.0

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        """Bind the socket and start the receive thread.

        Binding happens here so that a port conflict fails
        loudly at startup instead of dying silently in the background.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._listen_ip, self._listen_port))
        self._sock.settimeout(_SOCKET_TIMEOUT_S)

        _LOG.info(
            "Listening for head poses on %s:%d",
            self._listen_ip,
            self._listen_port,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._receive_loop, name="udp-receiver", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the receive thread and close the socket."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def get_latest_pose(self) -> Tuple[Optional[HeadPose], float]:
        """Return ``(pose, monotonic_time_received)``.

        ``pose`` is ``None`` until the first valid datagram arrives.
        """
        with self._lock:
            return self._latest_pose, self._latest_time

    @staticmethod
    def _parse_pose(message: str) -> HeadPose:
        """Parse ``"roll,pitch,yaw[,...]"`` into a :class:`HeadPose`."""
        parts = message.strip().split(",")
        if len(parts) < 3:
            raise ValueError(
                f'expected "roll,pitch,yaw" but got {message!r}'
            )

        return HeadPose(
            roll=float(parts[0]),
            pitch=float(parts[1]),
            yaw=float(parts[2]),
        )

    def _receive_loop(self) -> None:
        """Replace the latest pose as datagrams arrive."""
        while not self._stop_event.is_set():
            try:
                data, _addr = self._sock.recvfrom(_MAX_DATAGRAM_BYTES)
            except socket.timeout:
                continue
            except OSError:
                # Socket closed during shutdown.
                break

            try:
                message = data.decode("utf-8", errors="replace")
                pose = self._parse_pose(message)
            except ValueError as exc:
                _LOG.warning("Bad pose packet: %s", exc)
                continue

            with self._lock:
                self._latest_pose = pose
                self._latest_time = time.monotonic()
