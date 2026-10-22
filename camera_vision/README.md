# Camera Vision

This subdirectory handles camera processing on the ground station. The camera feed is transmitted from the drone wirelessly via VTX to VRX, which gets read into a capture card on the ground station. The ground station code then processes it and streams it to the HoloLens via the Unity repo. The ground station also plays the video back in an OpenCV window.

The ground station also anntoates the feed live with AI defect detection using DINOv3 and YOLO models, shown in both the HoloLens and the local window. The unannotated feed can be recorded to disk for offline analysis.

_NOTE: See [ai/README.md](ai/README.md) for setup, datasets, training, video analysis, and credits._

## Components

* [main.py](main.py) - Entry point. Display loop with an FPS overlay on the ground station; streams to Unity and runs live AI annotation
* [read_feed.py](read_feed.py) - `CameraSource` handles reading from the capture device and produces frames. `FpsCounter` measures the FPS achieved.
* [stream.py](stream.py) - Streams the feed TO unity. Includes `FrameStreamer` which streams via TCP, plus the framing helpers and encoder for JPEG.
* [commands.py](commands.py) - UDP listener for HoloLens FOV commands to toggle recording and annotation.
* [ai/](ai/) - Defect detection with DINOv3 and YOLO. See [ai/README.md](ai/README.md) for details.
* [recordings/](recordings/) - Storage for offline recordings. This is gitignored so it may not exist until a recording is captured.
* [models/](models/) - Model weights and trained models. See the [ai/README.md](ai/README.md) file for details on what goes here.
* [datasets/](datasets/) - Training datasets (gitignored). See [ai/README.md](ai/README.md) for what goes here.
* [tests/](tests/) - Unit tests for the scripts via `pytest`.

## Setup and Usage

1. Create a venv in this directory and activate it.
2. Install PyTorch with CUDA support using the selector at https://pytorch.org/get-started/locally (Windows, pip, CUDA 12.x).
3. Install everything else with `pip install -e ".[dev,ai]"`
4. Setup the AI module with its setup instructiosn in [ai/README.md](ai/README.md).
5. To run the camera vision program, use the following commands:

```bash
python main.py                      # Playback + stream to Unity
python main.py --device 0           # Different capture device index
python main.py --device clip.mp4    # Play a video
python main.py --record             # Start recording immediately
python main.py --no-stream          # Playback only, no Unity stream
python main.py -v                   # Debug logging
python main.py --annotate           # Start annotating immediately
```

6. To start/stop recording, press `r`. An indicator will show recording in progress.
7. To turn live AI annotation on/off, press `a`. The AI models load when the program starts. If they can't load, the camera runs without annotation. For model setup, see [ai/README.md](ai/README.md).
8. To quit, press `q` or `Esc`.
9. For troubleshooting camera behavior, see _Camera Notes_ section below.

## Streaming to Unity

`python main.py` sends the feed to the Unity repo by default with an FPS counter. To omit streaming, use `python main.py --no-stream`. Annotations appear in both HoloLens stream and the local window. Recordings are unannotated and have no overlays/FPS counter.

### Video Transmission Protocol

These are important things to know about how the video streaming is structured in this repository, which the Unity repository conforms to:

* `camera_vision` is a TCP server that listens on 127.0.0.1:5010. *Since **Holographic Remoting runs on the PC** and not natively on HoloLens hardware, localhost is used.* Unity is the client that connects to this server for receiving a feed.
* The stream of data transmitted is structured as a little-endian uint32 denoting the length of the payload, followed by the JPEG payload itself.
* Length must be a nonzero number <= 8MB (8388608 bytes). Anything larger could possibly indicate a corrupted stream.
* Each payload is one complete JPEG image (BGR source). The frame resolution may change from frame to frame.
* The server sends the newest available frame without queueing or blocking to minimize delay. If no client is connected, frames are dropped and capture continues unaffected.
* The client does not know any details about the frames being transmitted (such as AI annotations); it simply renders and displays what it receives.

## Video Recording

Use video recording for offline analysis with AI.

The recorder writes MJPG at quality 95 in `.avi` format. Storage outputs expected 0.3-0.5GB per minute of 720p30 video. For lossless recording, switch `_FOURCC` in [record.py](record.py) to `"FFV1"`. This will yield larger files.

* Recordings capture the raw feed from the camera without any annotations or overlays.
* Files are auto-named `rec_YYYYMMDD_HHMMSS.avi` and can be found in `recordings/`.

### Command Protocol

Things to know about the HoloLens to ground station remote control for video recording and annotation; the Unity repo conforms to this:

* `camera_vision` listens for UDP datagrams on `127.0.0.1:5011` for commands from Unity.
* Commands are in UTF-8 format: `record:start`, `record:stop`, `record:toggle`, `annotate:on`, `annotate:off`, `annotate:toggle`.
* Unknown commands are ignored.

## Camera Notes

* If a camera refuses to open, try a different camera index. This 0-based index should correspond to the order of devices listed in Windows camera settings. Note: you can change this default constant `_CAPTURE_CARD` at the top of `main.py`.
* Ensure that "Allow multiple apps to use camera at the same time" is turned ON in Windows camera settings for the device. Otherwise another app's lock can make an index refuse to open.
* The camera's resolution/FPS are set in the same Advanced camera options page (under "media type"). Windows drivers may ignore programmatic requests, so you will need to configure this here. Keep in mind your camera's hardware limitations for what it can support.
* If a USB capture card misbehaves, switch `BACKEND` in `read_feed.py` to `cv2.CAP_DSHOW`.

## Software Testing

Automated unit tests live in [tests/](tests/) and cover each of the different components of the camera vision component.

To run tests locally:
```bash
pycodestyle --exclude=.venv,venv,__pycache__,.pytest_cache .
pytest
```

Expect no output from pycodestyle (to verify formatting) and all tests passing.
