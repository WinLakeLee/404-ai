from typing import List, Dict, Tuple, Union

import cv2
from ultralytics import FastSAM


class SAMDetector:
    def __init__(self, model_path: str = "FastSAM-s.pt", prompt: str = "toy car", device: str = "cuda", conf: float = 0.25, imgsz: int = 640):
        self.model_path = model_path
        self.prompt = prompt
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.model = FastSAM(model_path)

    def detect(self, image_path, return_image: bool = False) -> Union[List[Dict], Tuple[List[Dict], "cv2.Mat"]]:
        """Run FastSAM and return detections (and optionally an annotated BGR image).

        return_image=True gives you a copy of the original image with only boxes drawn,
        so colors stay unchanged instead of using FastSAM's RGB render output.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        results = self.model(
            source=str(image_path),
            texts=[self.prompt],
            device=str(self.device),
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
        )

        regions: List[Dict] = []
        annotated = image.copy() if return_image else None

        if len(results) > 0:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    regions.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "class_id": 1,
                    })

                    if annotated is not None:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if return_image:
            return regions, annotated
        return regions
