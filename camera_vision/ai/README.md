# AI Analysis Module for Defect Inspection

This package detects concrete defects in video. 2 models work together on each frame:
* **DINOv3 + trained classifier probe** - Meta's DINOv3 backbone is a frozen feature extractor, and a trained classifier "probe" is layered on top of it. This answers where something is wrong, and the video is annotated with red heatmap.
* **YOLO** - A detector trained to draw labeled boxes around 5 defect types. It answers what the defect is.

A YOLO box is kept only if DINOv3's heat is present inside it. For background on the terminology, see Meta's DINOv3 repo.

The install is shared with camera_vision.

## Components

* [detect.py](detect.py) - Includes: the `Detections` class for each frame's output, including boxes, labels, scores, and heatmap; the `Detector` protocol, and `fuse()` to combine DINOv3's heat with YOLO's boxes
* [annotate.py](annotate.py) - Draws the heatmap and boxes onto a frame
* [dino.py](dino.py) - Loads the backbone for DINOv3. ``DinoDefectDetector`` uses a 224 px window over the frame and scores each tile with the trained probe.
* [yolo.py](yolo.py) - Houses ``YoloDetector``, which wraps an Ultralytics YOLO model.
* [probe_train.py](probe_train.py) - Trains the custom DINOv3 probe.
* [yolo_train.py](yolo_train.py) - Trains the YOLO model.
* [analyze_video.py](analyze_video.py) - Analyzes and annotates a recorded video offline, outputs the annotated video and a JSON file of detected defects that can be plotted.
* [plot_json.py](plot_json.py) - Charts the JSON output of `analyze_video` with `seaborn` and `matplotlib`.
* [check_env.py](check_env.py) - Verifies `torch`, the GPU, and DINOv3 loading on the machine.

## Setup

1. Clone https://github.com/facebookresearch/dinov3 next to this repo. The repo contains only code
2. Place `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` inside `camera_vision/models`. The weights are distributed by Meta undder the DINOv3 License via a request form. You will get an email with time-limited download links. Do not commute or redistribute the weights.
3. Run `python -m ai.check_env`. You should see the GPU name, `loaded dinov3_vits16`, and a per-pass inference time in ms.

`models/` and `datasets/` are gitignored. Once all the training is complete, `models/` contains:
* `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`, the DINOv3 weights from Meta
* `probe.joblib`, the custom trained DINOv3 classifier probe
* `yolo11n.pt`, aka YOLO's starting weights. Downloaded automatically on first use
* `yolo_codebrim.pt`, the custom trained YOLO model
* `yolo_runs/`, YOLO's training logs and charts

## Datasets

Both datasets should be located in `camera_vision/datasets/`:
* [Kaggle "Surface Crack Detection"](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) - 40,000 227x227 concrete tiles. Unzip so `datasets/kaggle/Positive` and `datasets/kaggle/Negative` exist
* [CODEBRIM](https://zenodo.org/records/2620293) - 1,590 bridge photos with defect boxes. From Zenodo, download only CODEBRIM_original_images.zip, and extract it with 7-Zip. You should see `datasets/codebrim/original_dataset/images` and `.../annotations`. You can delete the __MACOSX folder.

Note there are about 400 CODEBRIM photos without annotations. `yolo_train.py` skips them to prevent false negatives.

The classes for box colors are: Crack (orange), Spallation (blue), Efflorescence (green), ExposedBars (red), CorrosionStain (magenta).

## Training

Once the datasets are ready, you can train your models.

* `python -m ai.probe_train` - Trains the DINOv3 probe on Kaggle and CODEBRIM together, prints validation F1 per defect, and saves `moels/probe.joblib`
* `python -m ai.yolo_train` - The first run copies CODEBRIM into `datasets/codebrim_yolo` in YOLO's format, shrunk to 1024 px, with EXIF rotation removed. This is becasue CODEBRIM's boxes were drawn on the unrotated photos. It then trains for 100 epochs and saves the best checkpoint as `models/yolo_codebrim.pt`. To redo the copy step, delete `datasets/codebrim_yolo`.

## Analyzing Video

* `python -m ai.analyze_video recordings\clip.MOV` - Runs both models and merges their annotations (DINOv3's heatmap and YOLO's boxes), then outputs an annotated video file.
* `--frame-interval` sets the sampling rate of frames for analysis. The default is 5, so that means 6 frames will be analyzed every second of a 30 FPS video. Frames in between show the most recent result. This is for computation power and readability purposes.
* `--heat-threshold` sets how confident DINOv3 must be before a tile is drawn red. It' salso the bar a YOLO box's confidence must meet for it to be shown. Default is 0.5
* `--python -m ai.plot_json recordings\clip_annotated.json --title "Title"` - Charts `heat_mean` and `tiles_above_threshold` over the video, saved as a PNG beside the JSON. Uses `seaborn` and `matplotlib`.

Each JSON record holds the `frame`, `heat_max`, `heat_mean`, `tiles_above_threshold`, and `boxes` (each with `label`, `score`, and `xyx` pixel coordinates).

## Datasets and Credits

* Ozgenel, C. F. (2019). *Concrete Crack Images for Classification*. Mendeley Data. Distributed on Kaggle as "Surface Crack Detection".
* Mundt, M., Majumder, S., Murali, S., Panetsos, P., & Ramesh, V. (2019). Meta-learning Convolutional Neural Architectures for Multi-target Concrete Defect Classification with the COncrete DEfect BRidge IMage Dataset. *CVPR*. Non-commercial research use; see the Zenodo page.
* Simeoni, O., et al. (2025). DINOv3. arXiv:2508.10104. Weights under the DINOv3 License.
* Jocher, G., & Qiu, J. (2024). Ultralytics YOLO11. AGPL-3.0.
* Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV*. YOLO's starting weights were trained on COCO.
