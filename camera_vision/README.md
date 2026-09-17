# Camera Vision

This subdirectory handles camera processing on the ground station. The camera feed is transmitted from the drone wirelessly via VTX to VRX, which gets read into a capture card on the ground station. The ground station code then processes it and streams it to the HoloLens via the Unity repo. The ground station also plays the video back in an OpenCV window.

**TODO: Incorporate AI annotation and analysis with DINOv3 and YOLO.**

## Components

[main.py](main.py) - Entry point. Display loop with an FPS overlay on the ground station. With `--stream` argument, the video feed is sent to Unity.
[read_feed.py](read_feed.py) - `CameraSource` handles reading from the capture device and produces frames. `FpsCounter` measures the FPS achieved.
[stream.py](stream.py) - Streams the feed TO unity. Includes `FrameStreamer` which streams via TCP, plus the framing helpers and encoder for JPEG.
[tests/](tests/) - Unit tests for the scripts via `pytest`.

## Setup and Usage

1. Create a venv in this directory and activate it.
2. Run `pip install -e ".[dev]"` inside this directory.
3. Run the camera vision program with the following commands

```py
python main.py                      # Playback for default capture device
python main.py --device 0           # Different capture device index
python main.py --device clip.mp4    # Play a video
python main.py --stream             # Send frames to Unity
python main.py -v                   # Debug logging
```

4. To quit, perss `q` or `Esc`.
5. For troubleshooting camera behavior, see _Camera Notes_ section below.

## Streaming to Unity

`python main.py --stream` sends the feed to the Unity repo. The local window's FPS overlay is omitted from the Unity stream so the operator's view is not clutter-heavy. JPEG quality is the constant inside `encode_jpeg()` in `stream.py`. To increase quality of the streamed image, adjust this constant.

### Video Transmission Protocol

These are important things to know about how the video streaming is structured in this repository, which the Unity repository conforms to:

* `camera_vision` is a TCP server that listens on 127.0.0.1:5010. *Since **Holographic Remoting runs on the PC** and not natively on HoloLens hardware, localhost is used.* Unity is the client that connects to this server for receiving a feed.
* The stream of data transmitted is structured as a little-endian uint32 denoting the length of the payload, followed by the JPEG payload itself.
* Length must be a nonzero number <= 8MB (8388608 bytes). Anything larger could possibly indicate a corrupted stream.
* Each payload is one complete JPEG image (BGR source). The frame resolution may change from frame to frame.
* The server sends the newest available frame without queueing or blocking to minimize delay. If no client is connected, frames are dropped adn capture continues unaffected.
* The client does not know any details about the frames being transmitted (such as AI annotations); it simply renders and displays what it receives.

## Camera Notes

* If a camera refuses to open, try a different camera index. This 0-based index should correspond to the order of devices listed in Windows camera settings. Note: you can change this default constant `_CAPTURE_CARD` at the top of `main.py`.
* Ensure that "Allow multiple apps to use camera at the same time" is turned ON in Windows camera settings for the device. Otherwise another app's lock can make an index refuse to open.
* The camera's resolution/FPS are set in the same Advanced camera options page (under "media type"). Windows drivers may ignore programmatic requests, so you will need to configure this here. Keep in mind your camera's hardware limitations for what it can support.
* If a USB capture card misbehaves, switch `BACKEND` in `read_feed.py` to `cv2.CAP_DSHOW`.

## Software Testing

Automated unit tests live in [tests/](tests/) and cover the FPS counter math, `CameraSource` behavior against a temp dummy video file, the stream framing, the JPEG payload encode/decode, and `FrameStreamer` against a real localhost socket.

To run tests locally:
```bash
pycodestyle --exclude=.venv,venv,__pycache__,.pytest_cache .
pytest
```

Expect no output from pycodestyle (to verify formatting) and all tests passing.
