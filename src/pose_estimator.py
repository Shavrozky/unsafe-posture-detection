from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, weights: str, conf: float = 0.35, imgsz: int = 640):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.keypoints is None or result.boxes is None:
            return detections

        keypoints_xy = result.keypoints.xy.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for idx in range(len(keypoints_xy)):
            detections.append({
                "keypoints": keypoints_xy[idx],
                "bbox": boxes_xyxy[idx],
                "confidence": float(confs[idx])
            })

        return detections