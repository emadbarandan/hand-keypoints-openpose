# OpenPose Hand (CPU-only, OpenCV DNN)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![OpenCV DNN](https://img.shields.io/badge/OpenCV-DNN-informational)](https://opencv.org/)

Lightweight **hand keypoint detection** using the **OpenPose hand** Caffe model, executed via **OpenCV DNN** on **CPU only**.  
This repo reproduces the functionality you might have in Colab (downloading the model, running inference on images, saving keypoints/skeleton overlays, and writing a CSV report) — but locally, with **Anaconda** and **VS Code**.

---

## Features
- No Caffe installation required — uses `cv2.dnn.readNetFromCaffe`.
- CPU‑only by default (works on Windows/macOS/Linux).
- Batch processing of a folder (recursively handles subfolders).
- Saves two images per input (`*_keypoints.jpg` and `*_skeleton.jpg`) and a CSV report.
- Clean CLI with tunable parameters (`--inH`, thresholds, paths).

---

## Quickstart

### 0) Prerequisites
- **Anaconda** (recommended) and **VS Code** (optional but handy).
- **Git** (to clone or push to GitHub).

### 1) Create and activate a conda environment
```bash
conda create -n openpose-hand python=3.10 -y
conda activate openpose-hand
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If you prefer conda YAML: `conda env create -f environment.yml && conda activate openpose-hand`

### 2) Download model files (automatically)
```bash
python prepare_models.py --models models/hand
```
This script fetches:
- `pose_deploy.prototxt` (OpenPose hand deploy prototxt)
- `pose_iter_102000.caffemodel` (~147 MB hand weights)

> **Note:** We do **not** commit large model files to GitHub. They’re downloaded on demand.

### 3) Put your images in `data/5/`
Example:
```
data/
└─ 5/
   ├─ img001.jpg
    ├─ img002.png
    └─ ...your images...
```

### 4) Run inference
```bash
python hand_infer.py --images data/5 --output outputs --models models/hand
```

Outputs:
- `outputs/5/<image>_keypoints.jpg`
- `outputs/5/<image>_skeleton.jpg`
- `outputs/hand_detection_report.csv`

---

## CLI options
```text
python hand_infer.py [--images PATH] [--output PATH] [--models PATH]
                     [--inH 368] [--pt-thresh 0.20] [--mean-thresh 0.22]
```
- `--images`: Folder containing images (default: `data/5`). If it contains subfolders, all are processed.
- `--output`: Where to write results (default: `outputs`).
- `--models`: Folder containing `pose_deploy.prototxt` & `pose_iter_102000.caffemodel` (default: `models/hand`).
- `--inH`: Network input height (default `368`). Lower to `256` or `224` for faster CPU runs.
- `--pt-thresh`: Per-keypoint detection threshold (default `0.20`).
- `--mean-thresh`: Mean confidence threshold over detected keypoints (default `0.22`).

---

## Performance & tips
- **CPU speed:** If it’s slow, run with `--inH 256` or pre‑resize very large images.
- **Windows:** No extra deps needed.
- **Linux:** If you see `libGL.so.1` errors: `sudo apt-get install -y libgl1`.
- **Left/right detection:** simple heuristic using thumb vs. index x‑coordinates. Mirrored images may flip results.

---

## How it works (short)
- `prepare_models.py` downloads the OpenPose hand prototxt and Caffe weights from trusted mirrors.
- `hand_infer.py` loads the model via `cv2.dnn`, detects 21 hand keypoints, draws keypoints/skeleton, infers left/right if the hand is confidently present, and writes a CSV report.

---

## Folder structure
```
.
├─ prepare_models.py      # downloads model files
├─ hand_infer.py          # runs inference on a folder of images
├─ models/hand/           # model files will be stored here
├─ data/5/                # put your images here
└─ outputs/               # results (images + CSV)
```

---

## Acknowledgements & licenses
- Hand model and prototxt are from the CMU Perceptual Computing Lab’s **OpenPose** project.  
  The **pretrained weights/prototxt** have their **own upstream license/terms** — review them before any commercial use.  
- This repo’s **code** is MIT‑licensed (see `LICENSE`).  
- Do **not** commit large model files (e.g., `.caffemodel`) to GitHub; they are fetched at runtime.

---

## Troubleshooting
- `ModuleNotFoundError: cv2`: reinstall dependencies inside your active conda env:  
  `pip install -r requirements.txt`
- `cv2.imshow` not working from terminals: this repo **saves** outputs to files instead of live windows.
- Very slow on CPU: try `--inH 256` or reduce image sizes.
- Corrupted model download: re‑run `prepare_models.py`. The script checks file size.

---

## Contributing
Pull requests are welcome for:
- Better hand presence heuristics or handedness logic
- ONNX conversion, OpenVINO, or TensorRT backends
- Tests and CI

---

## Citation
If you use this, please cite OpenPose and the CMU Perceptual Computing Lab accordingly.
