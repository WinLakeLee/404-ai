import os
from typing import List, Dict

from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path: str, device: str = "cuda", conf: float = 0.25, imgsz: int = 640):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)

    def detect(self, image_path, conf_override: float = None) -> List[Dict]:
        """Detect boxes. If `conf_override` is provided, use it as the model confidence threshold.
        """
        conf_to_use = float(conf_override) if conf_override is not None else self.conf
        results = self.model.predict(
            source=str(image_path),
            conf=conf_to_use,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        regions: List[Dict] = []
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    regions.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "class_id": cls_id,
                    })
        return regions
