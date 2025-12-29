import logging
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import torch.nn as nn

logger = logging.getLogger(__name__)


# PDN 모델 정의 (efficient_ad_model.py에서 이동)
class PDN(nn.Module):
    def __init__(self, out_channels=384, padding=False):
        super(PDN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 128, kernel_size=4, stride=1, padding=3 if padding else 0
        )
        self.conv2 = nn.Conv2d(
            128, 256, kernel_size=4, stride=1, padding=3 if padding else 0
        )
        self.conv3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1 if padding else 0
        )
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class PDNBackend:
    def __init__(self, checkpoint_path: str, device: str = "cuda", anomaly_threshold: float = 0.5):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.anomaly_threshold = anomaly_threshold
        if PDN is None:
            logger.error("EfficientAD model definition (PDN) not found")
            self.model = None
        else:
            self.model = PDN(out_channels=384).to(self.device).eval()
            if Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded EfficientAD checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_crop(self, bgr_image, bbox, threshold: Optional[float] = None):
        if self.model is None:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "threshold": threshold or self.anomaly_threshold,
                "backend": "efficientad_missing_model",
            }
        x1, y1, x2, y2 = bbox
        crop = bgr_image[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            # 단일 모델의 출력값으로 anomaly score 계산 (예시: max 값)
            score = float(output.max())
        th = threshold if threshold is not None else self.anomaly_threshold
        return {
            "is_anomaly": bool(score >= th),
            "score": score,
            "threshold": th,
            "backend": "efficientad",
        }
