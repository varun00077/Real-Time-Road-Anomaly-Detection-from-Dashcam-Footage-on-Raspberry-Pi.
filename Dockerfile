FROM python:3.9-slim
# NOTE: No --platform flag here.
# • On PC (x86_64): builds natively as amd64 — stable for simulation
# • On Raspberry Pi 4 (arm64): builds natively as arm64 — no emulation needed
# To force arm64 emulation on PC (slow/unstable): docker build --platform linux/arm64 .

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    # v4l2 camera support (USB + CSI camera on RPi)
    libv4l-dev \
    v4l-utils \
    # video encoding
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── HOW TO RUN ──────────────────────────────────────────────────────────────
# Simulate RPi4 (no camera needed):
#   docker run --rm --cpus=4 --memory=4g yolo-rpi
#
# Simulate RPi4 with benchmark frames:
#   docker run --rm --cpus=4 --memory=4g yolo-rpi python infer.py --benchmark --frames 200
#
# Run with Pi camera (USB or CSI via v4l2) on actual Raspberry Pi:
#   docker run --rm --cpus=4 --memory=4g --device /dev/video0 yolo-rpi python infer.py --camera
#
# Run with camera and save annotated video to host:
#   docker run --rm --cpus=4 --memory=4g --device /dev/video0 \
#       -v $(pwd)/output:/app/output \
#       yolo-rpi python infer.py --camera --outdir /app/output
# ────────────────────────────────────────────────────────────────────────────

CMD ["python", "infer.py", "--benchmark", "--frames", "100"]
