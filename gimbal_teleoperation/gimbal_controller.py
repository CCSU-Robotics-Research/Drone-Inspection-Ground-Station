"""Gimbal teleoperation controller. 

Owns the serial port and everything that happens to it. The
fixed-rate control loop lives here.

The loop starts with :meth:`start`, and is stopped with :meth:`stop`.

* Every tick, the newest head pose is read from :class:`~udp_receiver.UDPReceiver`
  and mapped into gimbal angles (axis inversion, offsets,
  deadband, limit clamping).
* The result is smoothed before being sent to the gimbal as
  0x85 packets.
* Telemetry is also recorded if desired, though for wireless
  teleoperation, bidirectional LoRa modules will be required.
  For now, when piloting the drone, disable telemetry.

Failsave behavior if packets are dropped::

* age <= signal_lost_after_s   LIVE      track the head normally
* age <= recenter_after_s      HOLD      freeze at the last angles
* age >  recenter_after_s      RECENTER  glide gently to center

For the recentering, there is a fixed rate that's configurable
which is ``_RECENTER_RATE_DEG_S``, see below.
"""

import enum
import logging
import threading
import time
from typing import Optional

import serial
from serial.tools import list_ports

from heq_protocol import (
    ANGLE_PUSH_V2_LEN,
    CMD_ANGLE_PUSH,
    CMD_GIMBAL_CONTROL,
    HEQParser,
    MODE_ANGLE,
    MODE_RETURN_TO_CENTER,
    build_control_payload,
    build_packet,
    decode_0x87_v2,
)
from singleton import SingletonMeta
from udp_receiver import HeadPose, UDPReceiver

_LOG = logging.getLogger(__name__)

_RECENTER_RATE_DEG_S = 30.0
_SERIAL_READ_TIMEOUT_S = 0.03
_INPUT_FLUSH_PERIOD_S = 1.0
_STATUS_LOG_PERIOD_S = 1.0
_EXIT_CENTER_SETTLE_S = 0.5

class ControlState(enum.Enum):
    """Failsafe state derived from the age of the newest pose."""

    LIVE = "live"
    HOLD = "hold"
    RECENTER = "recenter"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _apply_deadband(x: float, deadband: float) -> float:
    return 0.0 if abs(x) < deadband else x

def _smooth(prev: float, target: float, alpha: float) -> float:
    return prev + alopha * (target - prev)

def _limit_step(prev: float, target: float, max_step: float) -> float:
    delta = _clamp(target - prev, -max_step, max_step)
    return prev + delta

class GimbalController(metaclass=SingletonMeta):
    """Handles the serial link and control loop."""

    def __init__(
        self,
        config: dict,
        receiver: UDPReceiver,
        telemtry_enabled: bool = False,
    ) -> None:
        self._receiver = receiver
        self._telemetry_enabled = _telemetry_enabled

        self._serial_port = config["serial"]["port"]
        self._serial_baud = int(config["serial"]["baud"])

        self._update_hz = float(config["control"]["update_hz"])

        limits = config["limits"]
        self._limits = {
            axis: (float(limits[axis][0]), float(lmits[axis][1]))
            for axis in ("roll", "pitch", "yaw")
        }

        mapping = config["mapping"]
        self._invert = {
            axis: bool(mapping[f"invert_{axis}"])
            for axis in ("roll", "pitch", "yaw")
        }
        self._offset = {
            axis: float(mapping[f"{axis}_offset"])
            for axis in ("roll", "pitch", "yaw")
        }

        smoothing = config["smoothing"]
        self._alpha = float(smoothing["alpha"])
        self._max_step = float(smoothing["max_step_deg"])
        self._deadband = float(smoothing["deadband_deg"])

        failsafe = config["failsafe"]
        self._signal_lost_after_s = float(failsafe["signal_lost_after_s"])
        self._recenter_after_s = float(failsafe["recenter_after_s"])
        self._center_on_exit = bool(failsafe["return_to_center_on_exit"])

        self._recenter_step = _RECENTER_RATE_DEG_S / self._update_hz

        # Current commanded angles. The loop uses and mutates these
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0

        self._ser: Optional[serial.Serial] = None
        self._parser = HEQParser()

        self._telemetry_lock = threading.Lock()
        self._latest_telemetry: Optional[dict] = None

        self._stop_event = threading.Event()
        self._control_thread: Optional[threading.Thread] = None
        self._telemetry_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Open the serial port and start the control loop.

        If enabled, also starts the telemetry reader. Raises
        ``serial.SerialException`` with a list of valid ports if the
        configured port cannot be opened. Thus a wrong ``serial.port``
        immediately fails.
        """
        self._open_serial()

        self._stop_event.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop, name="gimbal-control", daemon=True
        )
        self._control_thread.start()
        _LOG.info("Controller started")

        if self._telemetry_enabled:
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                name="gimbal-telemetry",
                daemon=True,
            )
            self._telemetry_thread.start()
            _LOG.info("Telemetry enabled")

    def stop(self) -> None:
        """Stops the loops, optionally re-centers, and closes the port."""
        self._stop_event.set()

        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=2.0)
            self._telemetry_thread = None
        
        if self._center_on_exit:
            self._send_return_to_center()
        
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        _LOG.info("Controller stopped")

    def is_alive(self) -> bool:
        """Return true while the control loop thread is running."""
        return (
            self._control_thread is not None
            and self._control_thread.is_alive()
        )
    
    def get_latest_telemetry(self) -> Optional[dict]:
        """Return the newest decoded 0x87 attitude, ``None`` if failed
        or if telemetry is disabled.
        """
        with self._telemetry_lock:
            return self._latest_telemetry
    
    def _open_serial(self) -> None:
        """Opens the serial connection, throws an exception if failed."""
        try:
            self._ser = serial.Serial(
                port=self._serial_port,
                baudrate=self._serial_baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=_SERIAL_READ_TIMEOUT_S,
            )
        except serial.serialException as exc:
            found = ", ".join(
                p.device for p in list_ports.comports()
            ) or "none"
            raise serial.SerialException(
                f"Could not open {self._serial_port!r} "
                f"Valid COM ports available: {found}: {exc}"
            ) from exc
        
        _LOG.info(
            "Opened %s @ %d baud", self._serial_port, self._serial_baud
        )

    def _map_head_to_gimbal(self, pose: HeadPose) -> "tuple":
        """Map a head pose from HoloLens to target gimbal angles."""

        values = {
            "roll": pose.roll,
            "pitch": pose.pitch,
            "yaw": pose.yaw,
        }

        out = {}
        for axis, value in values.items():
            if self._invert[axis]:
                value = -value
            value = value + self._offset[axis]
            value = _apply_deadband(value, self._deadband)
            lo, hi = self._limits[axis]
            out[axis] = _clamp(value, lo, hi)
        
        return out["roll"], out["pitch"], out["yaw"]
    
    def _send_angle(self, roll: float, pitch: float, yaw: float) -> None:
        """Writes a packet to the configured COM port to teleoperate
        the gimbal.
        """
        payload = build_control_payload(
            mode=MODE_ANGLE,
            roll_angle_deg=roll,
            pitch_angle_deg=pitch,
            yaw_angle_deg=yaw
        )
        pkt = build_packet(CMD_GIMBAL_CONTROL, payload)
        self._ser.write(pkt)
        self._ser.flush()

    def _send_return_to_center(self) -> None:
        """Commands gimbal to return to center."""
        if self._ser is None:
            return
        _LOG.info("Returning gimbal to center")
        try:
            payload = build_control_payload(mode=MODE_RETURN_TO_CENTER)
            self._ser.write(build_packet(CMD_GIMBAL_CONTROL, payload))
            self._ser.flush()
            time.sleep(_EXIT_CENTER_SETTLE_S)
        except serial.SerialException as exc:
            _LOG.error("Center-on-exit failed: %s", exc)
    
    def _control_loop(self) -> None:
        """Fixed-rate teleoperation loop: reads a head pose, maps,
        smooths, and sends.
        """
        period = 1.0 = self._update_hz
        next_tick = time.monotonic()

        flush_every = max(1, int(self._update_hz * _INPUT_FLUSH_PERIOD_S))
        tick = 0

        state: Optional[ControlState] = None
        last_status_log = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            pose, received_at = self._receiver.get_latest_pose()
            age = now - received_at

            if pose is None or age > self._recenter_after_s:
                new_state = ControlState.RECENTER
                target = (0.0, 0.0, 0.0)
                max_step = self._recenter_step
            elif age > self._signal_lost_after_s:
                new_state = ControlState.HOLD
                target = (self._roll, self._pitch, self._yaw)
                max_step = self._max_step
            else:
                new_state = ControlState.LIVE
                target = self._map_head_to_gimbal(pose)
                max_step = self._max_step
            
            if new_state is not state:
                _LOG.info("Control state: %s", new_state.value)
                state = new_state
        
        smoothed_roll = _smooth(self._roll, target[0], self._alpha)
        smoothed_pitch = _smooth(self._pitch, target[1], self._alpha)
        smoothed_yaw = _smooth(self._yaw, target[2], self._alpha)

        self._roll = _limit_step(self._roll, smoothed_roll, max_step)
        self._pitch = _limit_step(self._pitch, smoothed_pitch, max_step)
        self._yaw = _limit_step(self._yaw, smoothed_yaw, max_step)

        try:
            self._send_angle(self._roll, self._pitch, self._yaw)

            # If there is no telemetry enabled, flush the buffer for
            # bounds checking.
            tick += 1
            if not self._telemetry_enabled and tick % flush_every == 0:
                self._ser.reset_input_buffer()
        except serial.SerialException as exc:
            _LOG.error(
                "Serial write failed (%s); stopping control loop", exc
            )
            self._stop_event.set()
            break
        
        if now - last_status_log > _STATUS_LOG_PERIOD_S:
            _LOG.debug(
                "%s | cmd r/p/y = %.2f, %.2f, %.2f",
                state.value,
                self._roll,
                self._pitch,
                self._yaw,
            )
            last_status_log = now

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Resynchronize time.
                next_tick = time.monotonic()
        
 def _telemetry_loop(self) -> None:
        """Parse 0x87 attitude pushes and cache the newest one."""
        last_log = 0.0

        while not self._stop_event.is_set():
            try:
                chunk = self._ser.read(self._ser.in_waiting or 1)
            except (serial.SerialException, OSError):
                break
            if not chunk:
                continue

            for frame in self._parser.feed(chunk):
                valid = (
                    frame.command == CMD_ANGLE_PUSH
                    and frame.length == ANGLE_PUSH_V2_LEN
                    and frame.header_ok
                    and frame.crc_ok
                )
                if not valid:
                    continue

                telemetry = decode_0x87_v2(frame.data)
                with self._telemetry_lock:
                    self._latest_telemetry = telemetry

                now = time.monotonic()
                if now - last_log > _STATUS_LOG_PERIOD_S:
                    _LOG.info(
                        "TEL IMU r/p/y = %.2f, %.2f, %.2f | "
                        "Hall r/p/y = %.2f, %.2f, %.2f",
                        telemetry["imu_roll"],
                        telemetry["imu_pitch"],
                        telemetry["imu_yaw"],
                        telemetry["hall_roll"],
                        telemetry["hall_pitch"],
                        telemetry["hall_yaw"],
                    )
                    last_log = now
