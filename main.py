import time
import cv2
import yaml

from src.pose_estimator import PoseEstimator
from src.posture_rules import PostureRules
from src.temporal_smoothing import TemporalSmoothing
from src.event_logger import EventLogger
from src.visualizer import Visualizer


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def open_camera(source, width=None, height=None):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {source}")

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def get_alert_threshold(label, config):
    posture_cfg = config["posture"]

    if label == "lying_fall":
        return posture_cfg["fall_duration_sec"]

    if label == "bending":
        return posture_cfg["bending_duration_sec"]

    if label == "hands_up":
        return posture_cfg["hands_up_duration_sec"]

    return None


def log_alert_if_needed(logger, last_event_time, label, duration, frame, config):
    """Log an unsafe posture event with debounce and optional snapshot."""
    event_cfg = config.get("event", {})

    camera_id = event_cfg.get("camera_id", "cam_01")
    cooldown_sec = event_cfg.get("cooldown_sec", 5)
    save_snapshot = event_cfg.get("save_snapshot", True)
    snapshot_dir = event_cfg.get("snapshot_dir", "outputs/clips")

    now = time.time()

    if now - last_event_time.get(label, 0) > cooldown_sec:
        snapshot_path = ""

        if save_snapshot:
            snapshot_path = logger.save_snapshot(
                frame=frame,
                camera_id=camera_id,
                posture=label,
                snapshot_dir=snapshot_dir
            )

        logger.log(
            camera_id=camera_id,
            posture=label,
            status="unsafe",
            duration_sec=duration,
            snapshot_path=snapshot_path
        )

        last_event_time[label] = now
        print(f"[ALERT] {label} detected for {duration:.2f}s | snapshot={snapshot_path}")


def process_detection(detection, posture_rules, smoother, logger, last_event_time, config, frame):
    """Classify a single detection and trigger an alert if the threshold is exceeded."""
    raw_label = posture_rules.classify(detection["keypoints"])
    label, duration = smoother.update(raw_label)

    threshold = get_alert_threshold(label, config)
    alert = threshold is not None and duration >= threshold

    if alert:
        log_alert_if_needed(
            logger=logger,
            last_event_time=last_event_time,
            label=label,
            duration=duration,
            frame=frame,
            config=config
        )

    return {"label": label, "duration": duration, "alert": alert}


def compute_fps(prev_time):
    """Return (fps, updated_prev_time)."""
    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    return fps, now


def run_loop(cap, pose_estimator, posture_rules, smoother, logger, visualizer, config):
    """Main capture-and-display loop."""
    window_name = config["display"]["window_name"]
    show_fps = config["display"].get("show_fps", True)
    last_event_time = {}
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame.")
            break

        detections = pose_estimator.predict(frame)
        labels_info = [
            process_detection(
                detection=d,
                posture_rules=posture_rules,
                smoother=smoother,
                logger=logger,
                last_event_time=last_event_time,
                config=config,
                frame=frame
            )
            for d in detections
        ]

        frame = visualizer.draw(frame, detections, labels_info)

        fps, prev_time = compute_fps(prev_time)
        if show_fps:
            visualizer.draw_fps(frame, fps)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    config = load_config()

    cap = open_camera(
        config["camera"]["source"],
        config["camera"].get("width"),
        config["camera"].get("height")
    )

    pose_estimator = PoseEstimator(
        weights=config["model"]["weights"],
        conf=config["model"]["conf"],
        imgsz=config["model"]["imgsz"]
    )

    posture_rules = PostureRules()
    smoother = TemporalSmoothing(window_size=5)
    logger = EventLogger(config["logging"]["event_csv"])
    visualizer = Visualizer()

    print("[INFO] Starting camera...")
    print("[INFO] Press Q to quit.")

    run_loop(cap, pose_estimator, posture_rules, smoother, logger, visualizer, config)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()