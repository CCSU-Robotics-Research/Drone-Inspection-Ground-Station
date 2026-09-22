"""Command listener from HoloLens FOV buttons for recording.

A UDP datagram carries a command (e.g., "record:toggle") from the
HoloLens FOV that is executed here. Unknown commands are ignored.
"""

import logging
import socket
import threading
from typing import Callable, Dict, Optional

_LOG = logging.getLogger(__name__)

COMMAND_HOST = "127.0.0.1"
COMMAND_PORT = 5011
_RECV_BUFFER = 256


class CommandListener:
    """Receives command datagrams and dispatches them to handlers.

    ``handlers`` maps full command strings to callables. Passing ``port=0``
    (tests) lets the OS choose.
    """

    def __init__(
        self,
        handlers: Dict[str, Callable[[], None]],
        host: str = COMMAND_HOST,
        port: int = COMMAND_PORT,
    ) -> None:
        self._handlers = dict(handlers)
        self._host = host
        self._requested_port = port

        self._sock: Optional[socket.socket] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        """The port bound."""
        if self._sock is None:
            return self._requested_port
        return self._sock.getsockname()[1]

    def start(self) -> None:
        """Bind and start dispatching in a background thread."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self._host, self._requested_port))
        self._sock.settimeout(0.5)

        _LOG.info(
            "Listening for commands on %s:%d", self._host, self.port
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen_and_delegate,
            name="command-listener",
            daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop dispatching and close the socket."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _listen_and_delegate(self) -> None:
        while not self._stop_event.is_set():
            try:
                data, _ = self._sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break

            command = data.decode("utf-8", errors="replace").strip()
            handler = self._handlers.get(command)
            if handler is None:
                _LOG.warning("Unknown command: %r", command)
                continue

            _LOG.info("Command: %s", command)
            try:
                handler()
            except Exception:
                _LOG.exception("Command %r handler failed", command)
