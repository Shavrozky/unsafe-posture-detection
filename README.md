# 🦴 Unsafe Posture Detection

Real-time human posture monitoring system using **YOLOv8 Pose Estimation**. Detects unsafe postures from a webcam or RTSP stream and logs alert events to CSV.

---

## 🎯 Features

- **Real-time pose estimation** via YOLOv8n-pose
- **Multi-person support** — classifies each detected person independently
- **5 posture labels**: `standing`, `sitting`, `bending`, `lying_fall`, `hands_up`
- **Temporal smoothing** — filters noisy keypoint fluctuations across frames
- **Alert system** — triggers when an unsafe posture is held beyond a configurable duration
- **Debounced event logging** — writes unsafe events to CSV with a 5-second cooldown
- **FPS overlay** — real-time performance display

---

## 🗂️ Project Structure

```
unsafe-posture-detection/
├── main.py                  # Entry point
├── config.yaml              # All runtime configuration
├── requirements.txt
├── yolov8n-pose.pt          # Pre-trained YOLOv8 pose weights
├── outputs/
│   └── events.csv           # Logged alert events
└── src/
    ├── pose_estimator.py    # YOLOv8 inference wrapper
    ├── posture_rules.py     # Rule-based posture classifier
    ├── temporal_smoothing.py# Majority-vote label smoother
    ├── event_logger.py      # CSV event logger
    └── visualizer.py        # Bounding box + keypoint renderer
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
camera:
  source: 0          # 0 = webcam, or RTSP URL string
  width: 1280
  height: 720

model:
  weights: yolov8n-pose.pt
  conf: 0.35         # Detection confidence threshold
  imgsz: 640

posture:
  fall_duration_sec: 3.0       # Alert after 3s of lying_fall
  bending_duration_sec: 5.0    # Alert after 5s of bending
  hands_up_duration_sec: 2.0   # Alert after 2s of hands_up

display:
  window_name: Unsafe Posture Detection
  show_fps: true

logging:
  event_csv: outputs/events.csv
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
python main.py
```

Press **`Q`** to quit.

---

## 🧠 Posture Classification Logic

Classification is done rule-based using **COCO 17-keypoint** skeleton (shoulder, hip, knee, ankle, wrist midpoints).

| Label | Condition |
|---|---|
| `hands_up` | At least one wrist is above the shoulder by ≥10% of torso height |
| `lying_fall` | Horizontal body span > 1.4× vertical span |
| `bending` | Horizontal body span > 0.55× vertical span |
| `sitting` | Hip-to-knee distance < 75% of knee-to-ankle distance |
| `standing` | None of the above |

Priority order: `hands_up → lying_fall → bending → sitting → standing`

---

## 📋 Alert Logging

Unsafe events (`lying_fall`, `bending`, `hands_up`) are logged to `outputs/events.csv` when the posture duration exceeds the configured threshold. A **5-second debounce** prevents spam logging per label.

Example CSV output:

```
timestamp,camera_id,posture,status,duration_sec
2026-05-03 00:01:12,cam_01,lying_fall,unsafe,3.42
2026-05-03 00:01:28,cam_01,hands_up,unsafe,2.11
```

---

## 📦 Dependencies

| Package | Version |
|---|---|
| `ultralytics` | 8.3.159 |
| `opencv-python` | 4.10.0.84 |
| `numpy` | 1.26.4 |
| `PyYAML` | 6.0.2 |

---

## 🔌 RTSP Stream Support

To use an IP camera, change `source` in `config.yaml`:

```yaml
camera:
  source: "rtsp://user:password@192.168.1.100:554/stream"
```
