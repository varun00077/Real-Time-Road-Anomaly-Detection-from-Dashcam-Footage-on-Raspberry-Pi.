# Real-Time Road Anomaly Detection from Dashcam Footage on Raspberry Pi

Real-time road anomaly detection from dashcam footage using a YOLOv8n model exported to ONNX, containerised with Docker for easy simulation on a PC and deployment on a Raspberry Pi 4.

**Detected classes:** cracks, potholes, open manholes, good road

---

## Results

Evaluated on 480 validation images:

| Metric | Value |
|---|---|
| mAP@50 | 0.498 |
| Precision | 0.778 |
| Recall | 0.670 |
| F1 | 0.720 |
| Avg latency (ONNX, RPi4 sim) | ~83 ms |
| Avg FPS (ONNX, RPi4 sim) | ~12 FPS |

---

## Project Structure

```
├── infer.py              # Main inference script (image / video / folder / camera / benchmark)
├── evaluate.py           # Full mAP@50 evaluation with per-class AP, precision, recall, F1
├── convert_to_onnx.py    # Export best.pt → best.onnx
├── best.onnx             # YOLOv8n model weights (ONNX, opset 20, 12 MB)
├── Dockerfile            # Container for PC simulation and RPi4 deployment
├── requirements.txt      # Python dependencies
└── .dockerignore
```

---

## Quickstart

### 1 — Clone and build

```bash

git clone https://github.com/varun00077/Real-Time-Road-Anomaly-Detection-from-Dashcam-Footage-on-Raspberry-Pi..git

docker build -t yolo-rpi .
```

### 2 — Simulate RPi4 on your PC (benchmark)

```bash
docker run --rm --cpus=4 --memory=4g yolo-rpi
```

Runs 100 synthetic frames at 640×640 through the ONNX model with 4 CPU threads, printing latency and FPS.

### 3 — Run on a folder of images

```bash
# Windows
docker run --rm --cpus=4 --memory=4g ^
    -v D:/rpi_yolo_test:/app/project yolo-rpi ^
    python infer.py ^
        --source /app/project/dataset/test/images ^
        --outdir /app/project/output

# Linux / macOS
docker run --rm --cpus=4 --memory=4g \
    -v $(pwd):/app/project yolo-rpi \
    python infer.py \
        --source /app/project/dataset/test/images \
        --outdir /app/project/output
```

Annotated images and a `results.csv` are saved to `output/`.

### 4 — Run on a single image

```bash
docker run --rm --cpus=4 --memory=4g \
    -v $(pwd):/app/project yolo-rpi \
    python infer.py --source /app/project/road.jpg --outdir /app/project/output
```

---

## Evaluation (mAP)

```bash
# Requires images + YOLO-format .txt labels at dataset/valid/
docker run --rm --cpus=4 --memory=4g \
    -v $(pwd):/app/project yolo-rpi \
    python /app/project/evaluate.py \
        --images /app/project/dataset/valid/images \
        --labels /app/project/dataset/valid/labels \
        --outdir /app/project/eval_output
```

Outputs `summary.txt`, `eval_results.csv`, and annotated images.

---

## Deploy on Actual Raspberry Pi 4

### Install Docker on Pi

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

### Build natively on Pi (arm64)

```bash
git clone https://github.com/YOUR_USERNAME/Real-Time-Road-Anomaly-Detection-from-Dashcam-Footage-on-Raspberry-Pi.git
cd Real-Time-Road-Anomaly-Detection-from-Dashcam-Footage-on-Raspberry-Pi
docker build -t yolo-rpi .
```

> Building directly on the Pi uses native arm64 — no emulation, no QEMU issues.

### Run with camera

```bash
# USB camera or Pi camera (via v4l2)
docker run --rm --cpus=4 --memory=4g \
    --device /dev/video0 \
    -v $(pwd)/output:/app/output \
    yolo-rpi python infer.py --camera --outdir /app/output
```

Records annotated video to `output/camera_<timestamp>.mp4`. Press `Ctrl+C` to stop.

### Run without Docker (directly on Pi)

```bash
pip install -r requirements.txt
python infer.py --camera --outdir output/
python infer.py --source road.jpg --outdir output/
python infer.py --benchmark --frames 100
```

---

## CLI Reference — `infer.py`

| Argument | Default | Description |
|---|---|---|
| `--source PATH` | — | Image, video, or folder of images |
| `--camera` | false | Live camera inference |
| `--device PATH` | `/dev/video0` | Camera device (Linux) |
| `--benchmark` | false | Synthetic benchmark (no input needed) |
| `--frames N` | 100 | Number of benchmark frames |
| `--outdir PATH` | `.` | Directory to save output |
| `--show` | false | Display window (requires display) |

---

## Convert to ONNX (optional — `best.onnx` already included)

```bash
pip install ultralytics onnx onnxslim
python convert_to_onnx.py
```

Requires `best.pt` (not included in repo — provide your own trained weights).

---

## Requirements

- Docker (PC simulation or Pi deployment), **or**
- Python 3.8+ with `pip install -r requirements.txt` (direct install)

```
numpy>=1.23.0,<2.0.0
opencv-python-headless>=4.8.0
onnxruntime>=1.16.0
```

---

## Model

- Architecture: YOLOv8n (nano)
- Input: 640×640 RGB
- Output: 4 classes — `cracks`, `good_road`, `open_manhole`, `pothole`
- Format: ONNX (opset 20, dynamic batch)
- Size: 12 MB
