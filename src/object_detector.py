from ultralytics import YOLO


class ObjectDetector:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

        self.target_classes = {
            "cell phone",
            "book"
        }

        self.confidence_threshold = 0.35

    def detect(self, frame):

        results = self.model(
            frame,
            verbose=False
        )

        detections = []

        for result in results:

            for box in result.boxes:

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                if (
                    class_name in self.target_classes
                    and confidence >= self.confidence_threshold
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2)
                    })

        return detections