# Vehicle Detection and Counting System

A computer vision pipeline using **YOLOv8**, **OpenCV**, and **ByteTrack** for real-time vehicle detection, multi-object tracking, directional line-crossing counting, and automated batch video processing.

---

## Features

- **Object Detection**: Powered by Ultralytics YOLOv8 for accurate multi-class detection (`Cars`, `Motorcycles`, `Buses`, `Trucks`).
- **Object Tracking**: High-performance tracking via ByteTrack (`bytetrack.yaml`).
- **Line-Crossing Counting**: Directional line-crossing counting using 2D geometric segment intersection tests.
- **Batch Processing**: Automatically iterates across videos in `input_videos/`, applies scene-tailored counting lines, saves annotated video into `outputs/`, and exports a comparative CSV report to `results/`.
- **4K Dynamic Scaling**: Automatically scales high-resolution 4K frames for fast, memory-efficient inference while producing clean HD video outputs.
- **Live Visual Dashboard**: Real-time HUD banner displaying current and average FPS, frame progress, total counts, and per-category breakdown.
- **Streamlit Web App**: Interactive dashboard with chart analytics, in-browser video playback, and individual video processing.

---

## Project Structure

```text
Vehicle-Detection-System/
├── input_videos/                   # Source videos (untouched)
│   ├── traffic1.mp4                # 4K Overpass bridge view
│   ├── traffic2.mp4                # 4K Roadside street view
│   └── traffic3.mp4                # 4K Aerial overhead road view
├── outputs/                        # Annotated processed videos
│   ├── traffic1_counted.mp4
│   ├── traffic2_counted.mp4
│   └── traffic3_counted.mp4
├── results/                        # Reports and statistics
│   ├── traffic_comparison_report.csv
│   └── traffic_summary.json
├── vehicle_counter.py              # Single-video processor & module API
├── batch_process.py                # Automated batch processing pipeline
├── app.py                          # Streamlit web dashboard
├── requirements.txt                # Python dependencies
├── yolov8n.pt                      # YOLOv8 pre-trained weights
└── README.md                       # Project documentation
```

---

## Installation

```powershell
pip install -r requirements.txt
```

*Required packages: `ultralytics`, `opencv-python`, `lapx`, `streamlit`, `pandas`, `torch`.*

---

## Usage

### 1. Batch Processing All Videos
Automatically process all videos inside `input_videos/` and generate comparative reports:

```powershell
python batch_process.py
```

### 2. Single Video Processing
Process an individual video:

```powershell
python vehicle_counter.py --source input_videos/traffic1.mp4 --output outputs/traffic1_test.mp4
```

### 3. Launch Web Dashboard
Open the interactive Streamlit dashboard:

```powershell
streamlit run app.py
```

---

## Batch Benchmark Results

Summary from processing `traffic1.mp4`, `traffic2.mp4`, and `traffic3.mp4`:

| Video | Raw Resolution | Processed Resolution | Frames | Duration | Processing FPS | Total Counted | Cars | Motorcycles | Buses | Trucks | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **traffic1.mp4** | 2160x3840 (4K) | 720x1280 (HD) | 553 | 18.6s | 6.32 | **16** | 14 | 0 | 1 | 1 | Success |
| **traffic2.mp4** | 3840x2160 (4K) | 1280x720 (HD) | 280 | 9.3s | 6.44 | **6** | 5 | 0 | 0 | 1 | Success |
| **traffic3.mp4** | 2160x3840 (4K) | 720x1280 (HD) | 642 | 10.7s | 7.57 | **0** | 0 | 0 | 0 | 0 | Success |
