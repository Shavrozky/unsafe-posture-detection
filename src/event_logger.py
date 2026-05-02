import csv
import os
from datetime import datetime
import cv2


class EventLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if not os.path.exists(csv_path):
            with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp",
                    "camera_id",
                    "posture",
                    "status",
                    "duration_sec",
                    "snapshot_path"
                ])

    def log(self, camera_id, posture, status, duration_sec, snapshot_path=""):
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                camera_id,
                posture,
                status,
                round(duration_sec, 2),
                snapshot_path
            ])

    def save_snapshot(self, frame, camera_id, posture, snapshot_dir):
        os.makedirs(snapshot_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_id}_{posture}_{timestamp}.jpg"
        path = os.path.join(snapshot_dir, filename)

        cv2.imwrite(path, frame)

        return path