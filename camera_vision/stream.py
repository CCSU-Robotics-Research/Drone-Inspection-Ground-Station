"""Frame streaming to Unity.

Encodes each frame as JPEG and sent over a localhost TCP
connection for Unity.

``FrameStreamer`` streams the frames without blocking; a
sender thread transmits the newest one, with older frames
superseded. ``pack_frame`` and ``FrameAssembler`` are for
encoding protocols that ``FrameStreamer`` uses.
"""

import logging
import socket
import struct
import threading
from typing import List, Optional

import cv2

_LOG = logging.getLogger(__name__)

STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5010
MAX_FRAME_BYTES = 8 * 1024 * 1024


def pack_frame(payload: bytes) -> bytes:
    """Return one frame for a JPEG ``payload``."""
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ValueError(
            f"payload of {len(payload)} bytes is too big"
        )
    return struct.Struct("<I").pack(len(payload)) + payload


def encode_jpeg(frame) -> bytes:
    """JPEG-encode one BGR frame."""
    ok, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


class FrameAssembler:
    """Incremental parser for the length-prefixed frame stream.

    Receives raw chunks as they arrive from a socket; buffers
    partial frames across calls and returns every complete payload.
    If payload length is too long (possible stream corrupted), an
    error is raised.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()

    def frames_to_payloads(self, chunk: bytes) -> List[bytes]:
        """Output payloads from a chunk of frames."""
        self.buffer.extend(chunk)
        payloads: List[bytes] = []

        while len(self.buffer) >= 4:
            (length,) = struct.Struct("<I").unpack_from(self.buffer)
            if not 0 < length <= MAX_FRAME_BYTES:
                raise ValueError(
                    f"frame length {length} is too big"
                )

            total = struct.Struct("<I").size + length
            if len(self.buffer) < total:
                break

            payloads.append(
                bytes(self.buffer[4:total])
            )
            del self.buffer[:total]

        return payloads


class FrameStreamer:
    """TCP server that sends frames to a client."""

    def __init__(
        self, host: str = STREAM_HOST, port: int = STREAM_PORT
    ) -> None:
        self._host = host
        self._requested_port = port

        self._listener: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None

        self._lock = threading.Lock()
        self._latest = None
        self._new_frame = threading.Event()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        """The port bound. Differs from requested if 0."""
        if self._listener is None:
            return self._requested_port
        return self._listener.getsockname()[1]

    @property
    def client_connected(self) -> bool:
        return self._conn is not None

    def start(self) -> None:
        """Binds, listens, and starts server in a background thread."""
        self._listener = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self._listener.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        self._listener.bind((self._host, self._requested_port))
        self._listener.listen(1)
        self._listener.settimeout(0.5)

        _LOG.info("Streaming frames on %s:%d", self._host, self.port)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._serve, name="frame-streamer", daemon=True
        )
        self._thread.start()

    def send(self, frame) -> None:
        """Delivers newest frame to the sender (non-blocking)."""
        with self._lock:
            self._latest = frame
        self._new_frame.set()

    def stop(self) -> None:
        """Stops server and closes all sockets."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        _LOG.info("Frame streamer stopped")

    def _serve(self) -> None:
        """Accepts one client at a time and sends frames to it."""
        while not self._stop_event.is_set():
            try:
                conn, addr = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            _LOG.info("Stream client connected from %s:%d", *addr)
            self._conn = conn
            self._send_newest_frame(conn)
            conn.close()
            self._conn = None
            if not self._stop_event.is_set():
                _LOG.info("Stream client disconnected")

    def _send_newest_frame(self, conn: socket.socket) -> None:
        """Send the newest frame whenever one is available."""
        while not self._stop_event.is_set():
            if not self._new_frame.wait(timeout=0.25):
                continue

            with self._lock:
                frame = self._latest
                self._new_frame.clear()
            if frame is None:
                continue

            try:
                conn.sendall(pack_frame(encode_jpeg(frame)))
            except OSError:
                return
