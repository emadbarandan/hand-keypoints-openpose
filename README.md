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

