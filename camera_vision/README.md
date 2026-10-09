# Camera Vision

This subdirectory handles camera processing on the ground station. The camera feed is transmitted from the drone wirelessly via VTX to VRX, which gets read into a capture card on the ground station. The ground station code then processes it and streams it to the HoloLens via the Unity repo. The ground station also plays the video back in an OpenCV window.

The camera feed can also be recorded to the ground station disk for offline analysis.

**TODO: Incorporate AI annotation and analysis with DINOv3 and YOLO.**

## Components

* [main.py](main.py) - Entry point. Display loop with an FPS overlay on the ground station and streams to Unity. With `--no-stream` argument, the video feed is not sent to Unity.
* [read_feed.py](read_feed.py) - `CameraSource` handles reading from the capture device and produces frames. `FpsCounter` measures the FPS achieved.
* [stream.py](stream.py) - Streams the feed TO unity. Includes `FrameStreamer` which streams via TCP, plus the framing helpers and encoder for JPEG.
* [commands.py](commands.py) - UDP listener for HoloLens FOV commands. Namely for recording start/stop
* [ai/](ai/) - Defect detection package, including DINOv3. To access this, install the appropriate dependencies via `pip install -e ".[ai]"`.
* [recordings/](recordings/) - Storage for offline recordings. This is gitignored so it may not exist until a recording is captured.
* [models/](models/) - Storage for DINOv3 model weight files (\*.pth) and the linear classifier probes (\*.joblib). This is gitignored so it may not exist; you need to copy the *.pth manually.
* [tests/](tests/) - Unit tests for the scripts via `pytest`.

## Setup and Usage

1. Create a venv in this directory and activate it.
2. Install PyTorch with CUDA support using the selector at https://pytorch.org/get-started/locally (Windows, pip, CUDA 12.x).
3. Install everything else with `pip install -e ".[dev,ai]"`
4. Clone [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) next to this repo.
5. Place the `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` file inside `models/`. To get this file, you will need to request a license from Meta. You cannot find these weights in the repo. You will get an email with downloadable links for weights. Do not commit or redistribute the weights.
6. Run `python -m ai.check_env` once after setup. Expect the GPU name, `loaded dinov3_vits16`, and a per-pass inference time in ms. This number sizes how many frames per second the live annotator can analyze. Proceed forward once this is ready. For further usage of the AI module, see below.
7. To run the camera vision program, use the following commands:

```bash
python main.py                      # Playback + stream to Unity
python main.py --device 0           # Different capture device index
python main.py --device clip.mp4    # Play a video
python main.py --record             # Start recording immediately
python main.py --no-stream          # Playback only, no Unity stream
python main.py -v                   # Debug logging
```

8. To start/stop recording, press `r`. An indicator will show recording in progress.
9. To quit, press `q` or `Esc`.
10. For troubleshooting camera behavior, see _Camera Notes_ section below.

## Streaming to Unity

`python main.py` sends the feed to the Unity repo by default. To omit streaming, use `python main.py --no-stream`. Note the FPS indicator is not visible via HoloLens, but recording indicator is.

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

### Recording Command Protocol

Things to know about the HoloLens to ground station remote control for video recording; the Unity repo conforms to this:

* `camera_vision` listens for UDP datagrams on `127.0.0.1:5011` for commands from Unity.
* Commands are in UTF-8 format: `record:start`, `record:stop`, `record:toggle`.
* Unknown commands are ignored.

## Camera Notes

* If a camera refuses to open, try a different camera index. This 0-based index should correspond to the order of devices listed in Windows camera settings. Note: you can change this default constant `_CAPTURE_CARD` at the top of `main.py`.
* Ensure that "Allow multiple apps to use camera at the same time" is turned ON in Windows camera settings for the device. Otherwise another app's lock can make an index refuse to open.
* The camera's resolution/FPS are set in the same Advanced camera options page (under "media type"). Windows drivers may ignore programmatic requests, so you will need to configure this here. Keep in mind your camera's hardware limitations for what it can support.
* If a USB capture card misbehaves, switch `BACKEND` in `read_feed.py` to `cv2.CAP_DSHOW`.

## AI Module

The [ai/](ai/) package detects concrete defects in video using a DINOv3 backbone and a trained probe (a small classifier on those features). For more information on this terminology (backbone, head, linear probe), refer to Meta's DINOv3 repo.

Per the Setup and Usage (see above), make sure that `python -m ai.check_env` prints the appropriate values (the GPU name with CUDA drivers, `loaded dinov3_vits16`, and a per-pass inference time).

_More AI refinements with YOLO and other types of defects will come soon!_

### Crack Training Dataset

Training uses the Kaggle "Surface Crack Detection" dataset (Ozgenel's METU concrete tiles). These are 40,000 labeled 227x227 images. Download from Kaggle and unzip so `datasets\kaggle\Positive` and `datasets\kaggle\Negative` exist. Never commit the datasets to the remote.

### Commands

* `python -m ai.probe_train` - Trains the probe by creating `models/probe.joblib` and prints validation accuracy.
* `python -m ai.analyze_video recordings\clip.MOV` - Annotates a video offline with DINOv3. Writes `<name>_annotated.avi` plus a `.json` of heat map statistics per analyzed frame. This data can be used to create charts with `seaborn` or `matplotlib`.
* For `analyze_video`, the `--frame-interval` parameter sets the sampling rate at which a frame will be analyzed and the video will be annotated for defects. The default is 5, so 5 samples at 30 FPS video is 6 samples per second. `--heat-threshold` sets how confident a tile must be (for anomaly detection with DINOv3) before it is annotated.

## Software Testing

Automated unit tests live in [tests/](tests/) and cover each of the different components of the camera vision component.

To run tests locally:
```bash
pycodestyle --exclude=.venv,venv,__pycache__,.pytest_cache .
pytest
```

Expect no output from pycodestyle (to verify formatting) and all tests passing.
