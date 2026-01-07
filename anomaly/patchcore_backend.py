import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from torchvision.models import Wide_ResNet50_2_Weights

    _WIDE_RESNET_WEIGHTS = Wide_ResNet50_2_Weights.DEFAULT
except Exception:
    _WIDE_RESNET_WEIGHTS = None

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


class PatchCoreOptimized:
    def __init__(self, backbone_name: str = "wide_resnet50_2", sampling_ratio: float = 0.01, use_fp16: bool = True):
        self.sampling_ratio = sampling_ratio
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if backbone_name == "wide_resnet50_2":
            if _WIDE_RESNET_WEIGHTS is not None:
                self.backbone = models.wide_resnet50_2(weights=_WIDE_RESNET_WEIGHTS)
            else:
                self.backbone = models.wide_resnet50_2(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)

        self.backbone.eval().to(self.device)

        if self.use_fp16:
            self.backbone.half()
            logger.info("FP16 enabled")

        if hasattr(torch, "compile") and HAS_TRITON:
            try:
                self.backbone = torch.compile(self.backbone)
                logger.info("torch.compile enabled")
            except Exception as e:
                logger.warning(f"torch.compile skipped: {e}")

        self.features = []
        self._register_hooks()

        self.memory_bank = None
        self.knn = None
        self.faiss_index = None
        self.n_neighbors = 9

    def _register_hooks(self):
        self.backbone.layer2.register_forward_hook(self._hook_fn)
        self.backbone.layer3.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features.append(output)

    def to(self, device):
        self.device = torch.device(device)
        self.backbone.to(self.device)
        return self

    def _build_index(self):
        dim = self.memory_bank.shape[1]
        if HAS_FAISS:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_factory(dim, "Flat", faiss.METRIC_L2)
                if torch.cuda.is_available():
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                index.add(self.memory_bank)
                self.faiss_index = index
                return
            except Exception as e:
                logger.warning(f"FAISS GPU failed, fallback to CPU: {e}")
                self.faiss_index = faiss.IndexFlatL2(dim)
                self.faiss_index.add(self.memory_bank)
        else:
            self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
            self.knn.fit(self.memory_bank)

    def extract_features_for_predict(self, x):
        self.features = []
        x = x.to(self.device)
        if self.use_fp16:
            x = x.half()
        with torch.no_grad():
            self.backbone(x)
        f2, f3 = self.features[0], self.features[1]
        f3_resized = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=True)
        concat = torch.cat([f2, f3_resized], dim=1)
        pooled = F.avg_pool2d(concat, kernel_size=3, stride=1, padding=1)
        pooled = pooled.permute(0, 2, 3, 1)
        b, h, w, c = pooled.shape
        patches_per_image = int(h * w)
        feats = pooled.reshape(-1, c)
        return feats.float().cpu(), patches_per_image

    def predict(self, x, score_type: str = "max"):
        if self.memory_bank is None:
            raise RuntimeError("memory_bank is not initialized")
        feats_cpu, patches_per_image = self.extract_features_for_predict(x)
        feats_np = feats_cpu.numpy().astype("float32")
        total_patches, _ = feats_np.shape
        if patches_per_image <= 0 or total_patches % patches_per_image != 0:
            raise RuntimeError("Invalid patch layout")
        batch_size = total_patches // patches_per_image
        k = min(self.n_neighbors, self.memory_bank.shape[0])
        if self.faiss_index is not None:
            D, _ = self.faiss_index.search(feats_np, k)
            dists = D.astype("float32")
        elif self.knn is not None:
            dists, _ = self.knn.kneighbors(feats_np, n_neighbors=k)
            dists = dists.astype("float32")
        else:
            raise RuntimeError("No index available")
        patch_scores = dists.mean(axis=1).reshape(batch_size, patches_per_image)
        img_scores = patch_scores.mean(axis=1) if score_type == "mean" else patch_scores.max(axis=1)
        return torch.from_numpy(img_scores)


class PatchCoreBackend:
    def __init__(self, checkpoint_dir: str, device: str = "cuda", anomaly_threshold: float = 33.08):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.anomaly_threshold = anomaly_threshold

        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir must not be None")
        if not isinstance(checkpoint_dir, Path):
            checkpoint_dir = Path(str(checkpoint_dir))
        memory_bank_path = checkpoint_dir / "memory_bank.npy"
        meta_path = checkpoint_dir / "meta.json"
        if not memory_bank_path.exists():
            raise FileNotFoundError(f"Memory bank not found: {memory_bank_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.model = PatchCoreOptimized(
            backbone_name="wide_resnet50_2",
            sampling_ratio=meta.get("sampling_ratio", 0.01),
            use_fp16=meta.get("use_fp16", True),
        ).to(self.device)
        self.model.memory_bank = np.load(str(memory_bank_path)).astype("float32")
        self.model.n_neighbors = int(meta.get("n_neighbors", self.model.n_neighbors))
        self.model._build_index()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict_crop(self, bgr_image, bbox, threshold: Optional[float] = None):
        x1, y1, x2, y2 = bbox
        crop = bgr_image[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = self.transform(pil_img).unsqueeze(0)
        scores = self.model.predict(tensor, score_type="max")
        score = float(scores[0])
        th = threshold if threshold is not None else self.anomaly_threshold
        return {
            "is_anomaly": bool(score >= th),
            "score": score,
            "threshold": th,
            "backend": "patchcore",
        }


# OpenCV import placed at bottom to avoid circular deps in some environments
import cv2  # noqa: E402  # isort:skip
