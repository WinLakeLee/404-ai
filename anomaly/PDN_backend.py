import logging
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

# We'll expect the model definition to be provided or implemented in efficient_ad_model.py
try:
    from .efficient_ad_model import PDN, AutoEncoder
except ImportError:
    # Use relative import if running as part of a package, or absolute otherwise
    try:
        from anomaly.efficient_ad_model import PDN, AutoEncoder
    except ImportError:
        PDN = None
        AutoEncoder = None

logger = logging.getLogger(__name__)


class PDNBackend:
    def __init__(
        self, checkpoint_path: str, device: str = "cuda", anomaly_threshold: float = 0.5
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.anomaly_threshold = anomaly_threshold

        # Load models
        if PDN is None or AutoEncoder is None:
            logger.error("EfficientAD model definitions (PDN, AutoEncoder) not found")
            self.teacher = None
            self.student = None
            self.autoencoder = None
        else:
            # Initialize models with research-standard output channels
            self.teacher = PDN(out_channels=384).to(self.device).eval()
            self.student = PDN(out_channels=384).to(self.device).eval()
            self.autoencoder = AutoEncoder(out_channels=384).to(self.device).eval()

            if Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                # Handle potential different checkpoint formats
                if "teacher" in checkpoint:
                    self.teacher.load_state_dict(checkpoint["teacher"])
                if "student" in checkpoint:
                    self.student.load_state_dict(checkpoint["student"])
                if "autoencoder" in checkpoint:
                    self.autoencoder.load_state_dict(checkpoint["autoencoder"])
                logger.info(f"Loaded EfficientAD checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict_crop(self, bgr_image, bbox, threshold: Optional[float] = None):
        if self.teacher is None:
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
            teacher_output = self.teacher(tensor)
            student_output = self.student(tensor)
            ae_output = self.autoencoder(tensor)

            # Distance between teacher and student
            dist_st = torch.mean(
                (teacher_output - student_output) ** 2, dim=1, keepdim=True
            )

            # Distance between teacher and autoencoder (via student-like comparison if applicable)
            # This logic depends on the specific EfficientAD implementation
            dist_ae = torch.mean((teacher_output - ae_output) ** 2, dim=1, keepdim=True)

            # Simple combined score (example)
            score_map = dist_st + dist_ae
            score = float(score_map.max())

        th = threshold if threshold is not None else self.anomaly_threshold
        return {
            "is_anomaly": bool(score >= th),
            "score": score,
            "threshold": th,
            "backend": "efficientad",
        }
