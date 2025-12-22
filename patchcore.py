"""
YOLO + PatchCore 통합 이상 감지 파이프라인

Stage 1: YOLO로 차량 영역 감지
Stage 2: PatchCore로 anomaly detection (스크래치/파손/분리)

사용법:
    python yolo_training/detect_anomaly_pipeline.py --image path/to/image.jpg
    python yolo_training/detect_anomaly_pipeline.py --source path/to/images/ --save-dir results/
"""

import os
import sys
import cv2
import torch
import logging
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

# PyTorch 2.6+ sets torch.load(weights_only=True) by default. Allow YOLO checkpoints safely.
os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
try:
    from torch.serialization import add_safe_globals
    import ultralytics.nn.tasks as _ultra_tasks

    _safe_list = []
    for _name in ["DetectionModel", "SegmentationModel", "ClassificationModel", "PoseModel", "OBBModel"]:
        if hasattr(_ultra_tasks, _name):
            _safe_list.append(getattr(_ultra_tasks, _name))
    if _safe_list:
        add_safe_globals(_safe_list)

    # Torch 2.6+ defaults weights_only=True; force False for trusted local checkpoints.
    _ORIG_TORCH_LOAD = torch.load

    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs or kwargs.get("weights_only", False):
            kwargs["weights_only"] = False
        return _ORIG_TORCH_LOAD(*args, **kwargs)

    torch.load = _safe_torch_load
except Exception:
    # ultralytics import may fail here if not installed; YOLO import below will surface errors.
    pass

import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    import faiss

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

logger = logging.getLogger(__name__)

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
    def __init__(
        self, backbone_name="wide_resnet50_2", sampling_ratio=0.01, use_fp16=True
    ):
        """
        Args:
            sampling_ratio (float): 메모리 뱅크 샘플링 비율.
            use_fp16 (bool): True일 경우 FP16(반정밀도) 모드를 사용하여 속도 향상 및 메모리 절약.
        """
        self.sampling_ratio = sampling_ratio
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Backbone 로드
        if backbone_name == "wide_resnet50_2":
            if _WIDE_RESNET_WEIGHTS is not None:
                self.backbone = models.wide_resnet50_2(weights=_WIDE_RESNET_WEIGHTS)
            else:
                self.backbone = models.wide_resnet50_2(pretrained=True)
        else:
            self.backbone = models.resnet18(pretrained=True)

        self.backbone.eval()
        self.backbone.to(self.device)

        # ---------------------------------------------------------
        # 2. 성능 최적화: FP16 모드 (메모리 절반, 속도 증가)
        # ---------------------------------------------------------
        if self.use_fp16:
            self.backbone.half()
            logger.info("🚀 FP16(Half Precision) 모드가 활성화되었습니다.")

        # ---------------------------------------------------------
        # 3. 성능 최적화: torch.compile (PyTorch 2.x 이상, triton 필요)
        # ---------------------------------------------------------
        if hasattr(torch, "compile") and HAS_TRITON:
            try:
                self.backbone = torch.compile(self.backbone)
                logger.info("🚀 PyTorch 2.0 Compilation이 적용되었습니다.")
            except Exception as e:
                logger.warning(f"Compilation 실패 (무시 가능): {e}")
        else:
            logger.info("torch.compile 건너뜀 (triton 없음 또는 환경 미지원)")

        # 특징 추출을 위한 Hook 설정
        self.features = []
        self._register_hooks()

        self.memory_bank = None
        self.knn = None
        self.faiss_index = None
        self.n_neighbors = 9

    def _build_index(self):
        """KNN 또는 Faiss 인덱스 빌드"""
        dim = self.memory_bank.shape[1]

        if HAS_FAISS:
            # ---------------------------------------------------------
            # 4. 성능 최적화: FAISS IndexFactory 사용 (자동 최적화)
            # ---------------------------------------------------------
            # 데이터가 매우 많다면 'IVF1024,Flat' 등을 사용하여 근사 검색(속도↑) 가능
            # 여기서는 정확도를 위해 FlatL2를 쓰되 GPU 자원을 활용
            index_str = "Flat"

            try:
                # GPU 리소스 사용 시도
                res = faiss.StandardGpuResources()
                # 인덱스 생성
                index = faiss.index_factory(dim, index_str, faiss.METRIC_L2)

                # GPU로 이동 (메모리가 허용하는 경우)
                if torch.cuda.is_available():
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info("🚀 FAISS: GPU 인덱싱 성공")

                index.add(self.memory_bank)
                self.faiss_index = index

            except Exception as e:
                logger.warning(f"FAISS GPU 설정 실패 ({e}). CPU 모드로 전환합니다.")
                self.faiss_index = faiss.IndexFlatL2(dim)
                self.faiss_index.add(self.memory_bank)
        else:
            logger.info("Faiss 없음: Scikit-Learn KNN 사용.")
            self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
            self.knn.fit(self.memory_bank)

    def to(self, device):
        """Move backbone and update internal device tracking."""
        self.device = torch.device(device)
        self.backbone.to(self.device)
        return self

    def _hook_fn(self, module, input, output):
        # FP16 모드일 경우 Hook 출력도 FP16일 수 있으므로 필요시 처리 가능
        self.features.append(output)

    def _register_hooks(self):
        self.backbone.layer2.register_forward_hook(self._hook_fn)
        self.backbone.layer3.register_forward_hook(self._hook_fn)

    def extract_features(self, x):
        """이미지 배치를 입력받아 (N_patches, Dim) 형태의 특징 벡터 반환"""
        self.features = []

        # 입력 데이터 장치 및 타입 변환
        x = x.to(self.device)
        if self.use_fp16:
            x = x.half()

        with torch.no_grad():
            self.backbone(x)

        # Feature Map 가져오기
        f2 = self.features[0]
        f3 = self.features[1]

        # Upsampling & Concatenation
        # F.interpolate는 FP16에서 동작하지만, 안정성을 위해 float32로 변환해서 계산하는 경우도 있음.
        # 여기서는 속도를 위해 그대로 진행하되 align_corners=True는 유지
        f3_resized = F.interpolate(
            f3, size=f2.shape[-2:], mode="bilinear", align_corners=True
        )
        concat_features = torch.cat([f2, f3_resized], dim=1)

        # Average Pooling (Smoothing)
        patch_features = F.avg_pool2d(
            concat_features, kernel_size=3, stride=1, padding=1
        )

        # (Batch, C, H, W) -> (Batch, H, W, C) -> (N, C)
        patch_features = patch_features.permute(0, 2, 3, 1)
        output_features = patch_features.reshape(-1, patch_features.shape[-1])

        # 주의: Faiss(CPU)나 Sklearn은 float32만 받습니다.
        # 따라서 반환 시에는 float32로 캐스팅하여 CPU로 보냅니다.
        return output_features.float().cpu()

    # -----------------------------------------------------
    # 추가: 예측용 feature 추출 + anomaly score 계산
    # -----------------------------------------------------
    def extract_features_for_predict(self, x):
        """Return (features_cpu, patches_per_image).

        features_cpu: torch.Tensor (total_patches, dim)
        patches_per_image: int
        """
        self.features = []

        x = x.to(self.device)
        if self.use_fp16:
            x = x.half()

        with torch.no_grad():
            self.backbone(x)

        f2 = self.features[0]
        f3 = self.features[1]

        f3_resized = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=True)
        concat_features = torch.cat([f2, f3_resized], dim=1)
        patch_features = F.avg_pool2d(concat_features, kernel_size=3, stride=1, padding=1)

        patch_features = patch_features.permute(0, 2, 3, 1)
        b, h, w, c = patch_features.shape
        patches_per_image = int(h * w)
        feats = patch_features.reshape(-1, c)
        return feats.float().cpu(), patches_per_image

    def predict(self, x, score_type="max"):
        """Compute per-image anomaly scores for input batch x.

        score_type: 'max' (default) or 'mean' over patch scores.
        Returns: torch.Tensor shape (batch_size,)
        """
        if self.memory_bank is None:
            raise RuntimeError("memory_bank is not initialized")

        feats_cpu, patches_per_image = self.extract_features_for_predict(x)
        feats_np = feats_cpu.numpy().astype("float32")
        total_patches, dim = feats_np.shape
        if patches_per_image <= 0:
            raise RuntimeError("Invalid patches_per_image computed")
        if total_patches % patches_per_image != 0:
            raise RuntimeError("Mismatch in patches_per_image and total_patches")

        batch_size = total_patches // patches_per_image

        k = min(self.n_neighbors, self.memory_bank.shape[0])

        if self.faiss_index is not None:
            D, _ = self.faiss_index.search(feats_np, k)
            dists = D.astype("float32")
        elif self.knn is not None:
            dists, _ = self.knn.kneighbors(feats_np, n_neighbors=k)
            dists = dists.astype("float32")
        else:
            raise RuntimeError("No index available for nearest neighbor search")

        patch_scores = dists.mean(axis=1)  # (total_patches,)
        patch_scores = patch_scores.reshape(batch_size, patches_per_image)

        if score_type == "mean":
            img_scores = patch_scores.mean(axis=1)
        else:
            img_scores = patch_scores.max(axis=1)

        return torch.from_numpy(img_scores)


class AnomalyDetectionPipeline:
    def __init__(
        self,
        yolo_model_path=os.path.join("models", "yolo_weights", "best.pt"),
        patchcore_checkpoint=os.path.join("models", "patch_core"),
        device="cuda",
        conf_threshold=0.25,
        anomaly_threshold=33.08,  # PatchCore 임계값
        anomaly_model="patchcore",  # 'patchcore' or 'gan' or 'efficientad'
        gan_generator_path=None,
        efficientad_checkpoint=None,
        efficientad_image_size=256,
    ):
        """
        이상 감지 파이프라인 초기화

        Args:
            yolo_model_path: YOLO 세그멘테이션 모델 경로
            patchcore_checkpoint: PatchCore 체크포인트 디렉토리
            device: 'cuda' or 'cpu'
            conf_threshold: YOLO 신뢰도 임계값
            anomaly_threshold: PatchCore anomaly 점수 임계값
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.anomaly_threshold = anomaly_threshold
        # PatchCore 기본값 보관 (meta 누락 시 사용)
        self.sampling_ratio = 0.01
        self.use_fp16_default = True
        # 클래스별 임계값 (없으면 기본 anomaly_threshold 사용)
        self.class_anomaly_thresholds = {
            1: float("inf"),  # car: PatchCore로 결함 판단하지 않음
            3: float("inf"),  # car_floor: PatchCore로 결함 판단하지 않음
            4: float("inf"),  # car_housing: PatchCore로 결함 판단하지 않음
        }

        # 차량 관련 클래스 ID (data.yaml 기준)
        # 0: objects, 1: car, 2: car_broken_area, 3: car_floor, 4: car_housing, 5: car_scratch, 6: car_separated
        # 비정상 후보: car_broken_area(2), car_separated(6)
        self.car_class_ids = [1, 2, 3, 4, 5, 6]
        self.class_names = {
            1: "car",
            2: "car_broken_area",
            3: "car_floor",
            4: "car_housing",
            5: "car_scratch",
            6: "car_separated",
        }

        # 시각화 색상 팔레트 (BGR)
        self.viz_colors = {
            "normal": (40, 200, 40),
            "broken_yolo": (0, 165, 255),
            "separated_yolo": (255, 0, 255),
            "patchcore_anomaly": (0, 0, 255),
        }

        print(f"🖥️  Device: {self.device}")

        # 1. YOLO 모델 로드
        print(f"📦 YOLO 모델 로드: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)

        # 2. anomaly 모델 로드 (PatchCore 또는 GAN)
        self.anomaly_model = "patchcore"
        self.patchcore = self._load_patchcore(patchcore_checkpoint)

        # 3. PatchCore용 이미지 전처리
        self.patchcore_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_patchcore(self, checkpoint_dir):
        """PatchCore 모델 로드"""
        checkpoint_dir = Path(checkpoint_dir)
        import json

        # pkl 경로는 더 이상 사용하지 않음: 항상 memory_bank.npy + meta.json 사용
        memory_bank_path = checkpoint_dir / "memory_bank.npy"
        if not memory_bank_path.exists():
            raise FileNotFoundError(f"Memory bank not found: {memory_bank_path}")

        meta_path = checkpoint_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # 모델 생성
        model = PatchCoreOptimized(
            backbone_name="wide_resnet50_2",
            sampling_ratio=meta.get("sampling_ratio", getattr(self, "sampling_ratio", 0.01)),
            use_fp16=meta.get("use_fp16", getattr(self, "use_fp16_default", True)),
        ).to(self.device)

        # 메모리 뱅크 로드
        model.memory_bank = np.load(str(memory_bank_path))
        model.n_neighbors = int(meta.get("n_neighbors", model.n_neighbors))
        model._build_index()

        return model

    def _color_for_detection(self, is_broken_yolo, is_separated_yolo, is_anomaly_pc):
        if is_broken_yolo:
            return self.viz_colors["broken_yolo"], 3
        if is_separated_yolo:
            return self.viz_colors["separated_yolo"], 3
        if is_anomaly_pc:
            return self.viz_colors["patchcore_anomaly"], 3
        return self.viz_colors["normal"], 2

    def _place_label_box(self, x1, y1, label_size, occupied_boxes, img_h):
        """Place label to reduce overlap; returns (top, bottom)."""
        height = label_size[1] + 10
        top = max(y1 - height, 5)
        bottom = top + height

        def _overlaps(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

        for _ in range(8):
            box = (x1, top, x1 + label_size[0], bottom)
            has_overlap = any(_overlaps(box, other) for other in occupied_boxes)
            if not has_overlap and bottom < img_h - 5:
                break
            # move label downward to avoid overlap; clamp near bottom
            top = min(img_h - height - 5, top + height + 6)
            bottom = top + height

        occupied_boxes.append((x1, top, x1 + label_size[0], bottom))
        return top, bottom

    def _draw_legend(self, image):
        entries = [
            ("Broken (YOLO)", self.viz_colors["broken_yolo"]),
            ("Separated (YOLO)", self.viz_colors["separated_yolo"]),
            ("Anomaly (PatchCore)", self.viz_colors["patchcore_anomaly"]),
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        padding = 8
        swatch = 14
        line_gap = 6

        text_sizes = [cv2.getTextSize(lbl, font, scale, thickness)[0] for lbl, _ in entries]
        max_w = max(sz[0] for sz in text_sizes)
        text_h = max(sz[1] for sz in text_sizes)

        box_w = padding * 3 + swatch + max_w
        box_h = padding * 2 + len(entries) * text_h + (len(entries) - 1) * line_gap

        h, w = image.shape[:2]
        x1 = int(w - box_w - 10)
        y1 = int(h - box_h - 10)
        x2 = int(w - 10)
        y2 = int(h - 10)

        cv2.rectangle(image, (x1, y1), (x2, y2), (245, 245, 245), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 80, 80), 1)

        y_cursor = y1 + padding + text_h
        for (label, color), sz in zip(entries, text_sizes):
            swatch_top = y_cursor - text_h
            # Draw color using outline to match box borders
            cv2.rectangle(
                image,
                (x1 + padding, swatch_top),
                (x1 + padding + swatch, swatch_top + text_h),
                color,
                2,
            )
            cv2.putText(
                image,
                label,
                (x1 + padding * 2 + swatch, y_cursor),
                font,
                scale,
                (0, 0, 0),
                thickness,
            )
            y_cursor += text_h + line_gap

    def detect_car_regions(self, image_path):
        """
        YOLO로 차량 영역 감지

        Returns:
            list of dict: [{'bbox': [x1,y1,x2,y2], 'conf': score, 'class_id': id}, ...]
        """
        results = self.yolo_model.predict(
            source=str(image_path), conf=self.conf_threshold, verbose=False
        )

        car_regions = []

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])

                    # 차량 관련 클래스만 처리
                    if class_id in self.car_class_ids:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])

                        car_regions.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "conf": conf,
                                "class_id": class_id,
                            }
                        )

        return car_regions

    def detect_anomaly_in_region(self, image, bbox, threshold=None):
        """크롭된 차량 영역에서 PatchCore로 anomaly 점수 계산"""
        x1, y1, x2, y2 = bbox

        # 크롭
        cropped = image[y1:y2, x1:x2]

        # BGR to RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)

        # 전처리 및 배치 차원 추가
        img_tensor = self.patchcore_transform(pil_img).unsqueeze(0)

        scores = self.patchcore.predict(img_tensor, score_type="max")
        anomaly_score = float(scores[0])
        th = threshold if threshold is not None else self.anomaly_threshold
        is_anomaly = anomaly_score >= th

        return {
            "is_anomaly": bool(is_anomaly),
            "score": anomaly_score,
            "threshold": th,
        }

    def process_image(self, image_path, save_path=None):
        """
        단일 이미지 처리

        Args:
            image_path: 입력 이미지 경로
            save_path: 결과 저장 경로 (None이면 표시만)

        Returns:
            dict: 감지 결과
        """
        print(f"\n🔍 이미지 처리 중: {image_path}")

        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        result_image = image.copy()
        label_boxes = []  # text disabled; kept for potential future use

        # Stage 1: YOLO로 차량 영역 감지
        car_regions = self.detect_car_regions(image_path)
        print(f"   📦 감지된 차량 영역: {len(car_regions)}개")

        results = {
            "image_path": str(image_path),
            "car_regions": [],
            "anomaly_detected": False,
            "scratch_detected": False,
            "broken_detected": False,
            "separated_detected": False,
        }

        # Stage 2: 각 차량 영역에서 스크래치 검사
        for i, region in enumerate(car_regions):
            bbox = region["bbox"]
            x1, y1, x2, y2 = bbox

            cls_id = region["class_id"]
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

            # PatchCore로 anomaly 감지 (클래스별 임계값)
            class_threshold = self.class_anomaly_thresholds.get(
                cls_id, self.anomaly_threshold
            )

            if class_threshold == float("inf"):
                anomaly_result = {
                    "is_anomaly": False,
                    "score": 0.0,
                    "threshold": class_threshold,
                    "skipped": True,
                }
            else:
                anomaly_result = self.detect_anomaly_in_region(
                    image, bbox, threshold=class_threshold
                )

            # 결함 판정 로직
            is_broken_yolo = cls_id == 2
            is_separated_yolo = cls_id == 6
            is_anomaly_pc = anomaly_result["is_anomaly"]

            region_result = {
                "bbox": bbox,
                "yolo_conf": region["conf"],
                "class_id": cls_id,
                "class_name": cls_name,
                "anomaly": anomaly_result,
                "broken_by_yolo": is_broken_yolo,
                "separated_by_yolo": is_separated_yolo,
                "anomaly_by_patchcore": is_anomaly_pc,
            }
            results["car_regions"].append(region_result)

            # 시각화
            is_defect = is_broken_yolo or is_separated_yolo or is_anomaly_pc
            color, thickness = self._color_for_detection(
                is_broken_yolo, is_separated_yolo, is_anomaly_pc
            )

            # Bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)

            # 라벨 표시는 비활성화 (글씨 미표시)

            if is_anomaly_pc:
                results["anomaly_detected"] = True
            if is_broken_yolo or is_anomaly_pc:
                results["broken_detected"] = True
            if is_separated_yolo:
                results["separated_detected"] = True

            if is_defect:
                print(
                    f"   ⚠️  영역 {i+1}: 결함 감지! (cls={cls_name}, 점수={anomaly_result['score']:.2f})"
                )
            else:
                print(
                    f"   ✅ 영역 {i+1}: 정상 (cls={cls_name}, 점수={anomaly_result['score']:.2f})"
                )

        # 결과 저장 또는 표시 (범례는 하단 우측에 표시)
        self._draw_legend(result_image)
        if save_path:
            cv2.imwrite(str(save_path), result_image)
            print(f"   💾 결과 저장: {save_path}")

        return results, result_image


def main():
    parser = argparse.ArgumentParser(
        description="YOLO + PatchCore 이상 감지 파이프라인"
    )
    parser.add_argument("--image", type=str, help="단일 이미지 경로")
    parser.add_argument("--source", type=str, help="이미지 디렉토리 경로")
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo_training/runs/seg_toycar3/weights/best.pt",
        help="YOLO 모델 경로",
    )
    parser.add_argument(
        "--patchcore-checkpoint",
        type=str,
        default="models/patchcore_scratch",
        help="PatchCore 체크포인트 디렉토리",
    )
    parser.add_argument("--save-dir", type=str, help="결과 저장 디렉토리")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 신뢰도 임계값")
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=33.08,
        help="PatchCore anomaly 임계값",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    args = parser.parse_args()

    # 파이프라인 초기화
    pipeline = AnomalyDetectionPipeline(
        yolo_model_path=args.yolo_model,
        patchcore_checkpoint=args.patchcore_checkpoint,
        device=args.device,
        conf_threshold=args.conf,
        anomaly_threshold=args.anomaly_threshold,
    )

    # 단일 이미지 처리
    if args.image:
        results, result_img = pipeline.process_image(args.image, args.save_dir)

        # 결과 출력
        if results["scratch_detected"]:
            print(f"\n⚠️  최종 결과: 스크래치 감지됨!")
        else:
            print(f"\n✅ 최종 결과: 정상")

    # 디렉토리 처리
    elif args.source:
        pipeline.process_directory(args.source, args.save_dir)

    else:
        print("❌ --image 또는 --source를 지정하세요.")
        parser.print_help()


if __name__ == "__main__":
    # Example usage
    main()
