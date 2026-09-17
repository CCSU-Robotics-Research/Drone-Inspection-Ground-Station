# Gimbal Teleoperation

This subdirectory handles communication from the HoloLens to the station computer via UDP and ultimately to the HEQ G-Port gimbal via UART serial.

HoloLens sends its roll/pitch/yaw packets to the UDPReceiver via UDP, which are parsed into HeadPoses and forwarded to the GimbalController. These orientations are assembled into UART frames to communicate with the gimbal wirelessly.

## Components

[main.py](main.py) - Entry point. Parses the CLI for arguments, loads config, constructs and starts UDPReceiver and GimbalController, handles termination.
[udp_receiver.py](udp_receiver.py) - Owns the listen socket; a background thread keeps only the newest head pose, timestamped with a monotonic clock.
[gimbal_controller.py](gimbal_controller.py) - Serves as an adapter between the UDPReceiver and the serial port for wireless transmission. Maps head pose to gimbal angles (invert, offset, deadbands, and limits), smooths and slew-limits, assembles packets and transmits them, and runs the pose-loss failsafe. Optionally parses telemetry.
[heq_protocol.py](heq_protocol.py) - Protocol library and helper functions for the gimbal, including frame building, vendor-variant CRC32, the stream parser, and payload decoders.
[gimbal_config.json](gimbal_config.json) - Production configuration settings.
[motion_tests/](motion_tests/) - Sample scripts to test the gimbal's motion without HoloLens.
[tests/](tests/) - Unit tests for Python scripts, powered via `pytest`.

## Setup and Usage

***IMPORTANT: You must be on a Windows machine for this to work.***

### Hardware Materials

**These should either be attached on the drone or tested on the ground:**
* An HEQ G-PORT 3-axis Gimbal connected with UART output to a LoRa radio module. Gimbal connected to a power supply (or battery) with 12V output, and the UART LoRa radio connected to the power with a voltage regulator for 3.3-5V. All components should share the same GND.
* A RunCam MicroV2 connected to an HDZero Freestyle V2 VTX, that has a u.FL to SMA antenna and a keypad attached to it. This should be powered with 12V. **NEVER power the VTX without the antenna!**

**For the PC:**
* An HDZero VRX with SMA antennas, plugged in the wall with the barrel connector. Mini HDMI output to a capture card plugged into the PC.
* The other UART LoRa radio module (for the gimbal) plugged into a USB port on the PC. This is where gimbal teleoperation packets will be written/sent.

_It's strongly recommended to study the gimbal protocol documentation PDF file in Teams. This will make comprehension of setup easier._

### HoloLens and Unity

**_Configure the Unity setup by following instructions in the Unity repo's README.md. Once that is done, follow the rest of the instructions below._**

### UDP Teleoperation Bridge

1. Clone this repo to your machine.
2. Create a venv in the root directory.
3. Run `pip install -e ".[dev]"` inside the root directory.
4. Plug in the USB to UART adapter (wired testing or wireless radio link) to your machine. Use Device Manager to locate which COM port matches with the UART adapter.
5. Set the serial port in `gimbal_config.json`, such as `COM5`, `COM7`, etc.
6. Run the teleoperation bridge using these commands:

```py
python main.py                      # normal mode
python main.py --telemetry          # wired testing with telemetry ON
python main.py --serial-port COM7   # one-off port override
python main.py -v                   # debug logging
```

The Unity repo must be running (or `hololens_fake_test.py`) for poses to arrive. On startup with no poses, the controller gently holds center. On shutdown (Ctrl+C), the gimbal returns to its center position.

`--telemetry` is off by default. This is because the current radio modules do not support simultaneous bidirectional traffic well, and uplink telemetry alongside downlink commands increases packet loss. Enable it only with a wired USB connection on the bench or with upgraded LoRa radio modules.

### Config Settings

These are default settings from prior usage; you can tweak them as needed to fit your machine and HoloLens

* `udp` - Listen address for HoloLens pose datagrams.
* `serial` - Gimbal port and baud. Note this gimbal uses 115200 baud with 8 bits and 1 stop bit.
* `control.update_hz` - Control loop rate. Recommended to keep this at 5-20Hz. More Hz will mean smoother motion, but packet drops will be more noticeable.
* `limits` - Mechanical clamps per axis of rotation in degrees.
* `mapping` - Per-axis inversion and offset applied to head angles.
* `smoothing` - Exponential smoothing `alpha`, per-tick slew limit `max_step_deg`, and `deadband_deg` around zero.
* `failsafe` - Pose is considered lost after `signal_lost_after_s`; the gimbal then holds its pose until poses resume. Recentering only happens when the program terminates.

## Testing

### Gimbal Motion Testing

Inside [motion_tests/](motion_tests/), you will find some test scripts for testing gimbal motion without HoloLens.

* [hololens_fake_test.py](hololens_fake_test.py) - Impersonates the HoloLens by sending dynamic head-like motion over UDP to the running bridge. Tests real production path without a headset. Run `python main.py --telemetry` from a terminal, then run this file from another terminal.
* [move_gimbal_test.py](move_gimbal_test.py) - Direct serial exercise of the protocol. Center, speed mode, and angle mode with live telemetry. Run this file directly in the terminal.
* [read_gimbal_telemetry.py](read_gimbal_telemetry.py) - Reads the live telemetry from the gimbal. To save a log, use argument `--output file.log`. Run this file directly in the terminal.

These 3 files read from [gimbal_config.json](gimbal_config.json) with dynamic overrides available for `--port` as needed.

### Software Testing

Automated unit tests live in [tests/](tests/) and cover the gimbal's protocol, the HoloLens-to-gimbal mapping math, the shutdown and failsafe behavior, and the UDP receiver.

To run tests locally:
```bash
pycodestyle --exclude=.venv,venv,__pycache__,.pytest_cache .
pytest
```

Expect no output from pycodestyle (to verify formatting) and all tests passing.
