import cv2


class Visualizer:
    def draw(self, frame, detections, labels_info):
        """
        labels_info: list of dict
        [
            {
                "label": "standing",
                "duration": 1.2,
                "alert": False
            }
        ]
        """

        for detection, info in zip(detections, labels_info):
            bbox = detection["bbox"]
            keypoints = detection["keypoints"]

            x1, y1, x2, y2 = map(int, bbox)

            label = info["label"]
            duration = info["duration"]
            alert = info["alert"]

            color = (0, 0, 255) if alert else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} | {duration:.1f}s"
            cv2.putText(
                frame,
                text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            self._draw_keypoints(frame, keypoints, color)

        return frame

    def draw_fps(self, frame, fps):
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    def _draw_keypoints(self, frame, keypoints, color):
        for point in keypoints:
            x, y = int(point[0]), int(point[1])
            if x <= 0 or y <= 0:
                continue
            cv2.circle(frame, (x, y), 3, color, -1)