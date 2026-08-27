import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class HeadPose:
    roll: float
    pitch: float
    yaw: float

class UDPReceiver:
    def __init__(self, config):
        self.ip = config['ip']
        self.port = config['port']
        self.timeout_s = config['timeout_s']

        self._latest_pose: Optional[HeadPose] = None
        self._latest_time = 0.0
        
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._sock = None

    def _parse_pose(self, message: str) -> HeadPose:
        parts = message.strip().split(',')

        if len(parts) != 3:
            raise ValueError(f"Expected format \"roll,pitch,yaw\" but got {message!r}")
        
        return HeadPose(
            roll=float(parts[0]),
            pitch=float(parts[1]),
            yaw=float(parts[2]),
        )
    
    def _reecive_loop(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._sock.bind((self.ip, self.port))
        self._sock.settimeout(self.timeout_s)

        print(f"[UDP] Listening on {self.ip}:{self.port}")

        while self._running:
            try:
                data, addr = self._sock.recvfrom(1024)
                message = data.decode("utf-8", errors="replace")
                pose = self._parse_pose(message)

                with self._lock:
                    self._latest_pose = pose
                    self._latest_time = time.monotonic()
            
            except socket.timeout:
                continue
            except ValueError as e:
                print(f"[UDP] Bad packet: {e}")
            
        self._sock.close()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        
    def get_latest_pose(self):
        with self._lock:
            return self._latest_pose, self._latest_time
