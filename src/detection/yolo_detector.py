from ultralytics import YOLO
import os
import time

# src/detection/yolo_detector.py

import time
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8s.pt", conf_threshold=0.25):
        """
        conf_threshold: minimum confidence for a detection to be kept.
        """
        self.model = YOLO(model_path)
        self.model.overrides["verbose"] = False
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Runs YOLO on a single frame and returns:
        - detections: list of {label, confidence}
        - t_infer: inference time in seconds
        Only detections with confidence >= self.conf_threshold are kept.
        """
        t0 = time.time()
        # pass conf to model so it also prunes low-conf boxes internally
        res = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        t_infer = time.time() - t0

        detections = []
        for b in res.boxes:
            cls = int(b.cls.cpu().numpy()[0])
            conf = float(b.conf.cpu().numpy()[0])
            if conf < self.conf_threshold:
                continue
            detections.append({"label": res.names[cls], "confidence": conf})

        return detections, t_infer
