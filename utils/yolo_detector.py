import os
import cv2
import numpy as np
import logging
from typing import List, Tuple

class YoloV7Detector:
    """Lightweight wrapper around a YOLOv7 ONNX model."""

    def __init__(self, model_path: str = "models/yolov7.onnx", conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.net = None
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                self.logger.info(f"Loaded YOLOv7 model from {self.model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load YOLOv7 model: {e}")
        else:
            self.logger.warning(f"YOLOv7 model file not found: {self.model_path}")

    def detect(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Detect objects in an image.

        Returns a list of tuples: (label, confidence, bbox).
        Bbox is (x1, y1, x2, y2) in pixel coordinates.
        """
        if self.net is None:
            return []

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        detections = []
        for output in outputs[0]:
            scores = output[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > self.conf_threshold:
                cx, cy, w, h = output[0:4]
                x1 = int((cx - w / 2) * image.shape[1])
                y1 = int((cy - h / 2) * image.shape[0])
                x2 = int((cx + w / 2) * image.shape[1])
                y2 = int((cy + h / 2) * image.shape[0])
                detections.append((str(class_id), float(confidence), (x1, y1, x2, y2)))
        return detections
