"""
메인 파이프라인 엔트리포인트.
- Detection backend: YOLO or FastSAM (env DETECTION_BACKEND, CLI --det-backend)
- Anomaly backend: PatchCore 또는 EfficientAD placeholder (env ANOMALY_BACKEND, CLI --anomaly-backend)
- 단일 이미지 중심 처리. 필요 시 --source 로 디렉토리 순회.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import cv2

from detection.yolo_detector import YOLODetector
from detection.sam_detector import SAMDetector
from anomaly.patchcore_backend import PatchCoreBackend


class DetectionFactory:
    @staticmethod
    def create(backend: str, device: str, conf: float, imgsz: int, sam_model: Optional[str], sam_prompt: Optional[str], yolo_model: str):
        backend = backend.lower()
        if backend == "sam":
            return SAMDetector(
                model_path=sam_model or "FastSAM-s.pt",
                prompt=sam_prompt or "car",
                device=device,
                conf=conf,
                imgsz=imgsz,
            )
        return YOLODetector(
            model_path=yolo_model,
            device=device,
            conf=conf,
            imgsz=imgsz,
        )


class AnomalyFactory:
    @staticmethod
    def create(backend: str, device: str, patchcore_ckpt: Optional[str], anomaly_threshold: float):
        backend = backend.lower()
        if backend == "efficientad":
            return None  # placeholder
        return PatchCoreBackend(
            checkpoint_dir=patchcore_ckpt or "models/patch_core",
            device=device,
            anomaly_threshold=anomaly_threshold,
        )


class Pipeline:
    def __init__(
        self,
        det_backend: str,
        anomaly_backend: str,
        device: str,
        det_conf: float,
        det_imgsz: int,
        yolo_model: str,
        sam_model: Optional[str],
        sam_prompt: Optional[str],
        anomaly_threshold: float,
        patchcore_ckpt: Optional[str],
    ):
        self.detector = DetectionFactory.create(
            backend=det_backend,
            device=device,
            conf=det_conf,
            imgsz=det_imgsz,
            sam_model=sam_model,
            sam_prompt=sam_prompt,
            yolo_model=yolo_model,
        )
        self.anomaly_backend = anomaly_backend.lower()
        self.anomaly = AnomalyFactory.create(
            backend=anomaly_backend,
            device=device,
            patchcore_ckpt=patchcore_ckpt,
            anomaly_threshold=anomaly_threshold,
        )
        self.anomaly_threshold = anomaly_threshold

    def run_image(self, image_path: Path, save_path: Optional[Path] = None):
        print(f"\n[Input] {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        regions = self.detector.detect(image_path)
        print(f" Detected regions: {len(regions)}")

        results: List[Dict] = []
        for i, reg in enumerate(regions):
            bbox = reg["bbox"]
            cls_id = reg.get("class_id", 0)
            out = {
                "bbox": bbox,
                "conf": reg.get("conf"),
                "class_id": cls_id,
                "anomaly": None,
            }

            if self.anomaly and self.anomaly_backend == "patchcore":
                anomaly = self.anomaly.predict_crop(image, bbox, threshold=self.anomaly_threshold)
                out["anomaly"] = anomaly
                tag = "DEFECT" if anomaly["is_anomaly"] else "OK"
                print(f"  [{i}] cls={cls_id} score={anomaly['score']:.2f} -> {tag}")
            else:
                print(f"  [{i}] cls={cls_id} (anomaly backend skipped)")

            results.append(out)

            # 시각화: 녹색 박스
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image)
            print(f" Saved: {save_path}")
        else:
            cv2.imshow("result", image)
            cv2.waitKey(1)

        return results

    def run_directory(self, source_dir: Path, save_dir: Optional[Path] = None):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in exts])
        if not images:
            print(f"❌ 디렉토리에 이미지가 없습니다: {source_dir}")
            return
        for img in images:
            out = save_dir / img.name if save_dir else None
            self.run_image(img, out)


def main():
    parser = argparse.ArgumentParser(description="Detection + Anomaly pipeline")
    parser.add_argument("--image", type=str, help="단일 이미지 경로")
    parser.add_argument("--source", type=str, help="이미지 디렉토리 경로")
    parser.add_argument("--save-dir", type=str, help="결과 저장 디렉토리")

    parser.add_argument("--det-backend", type=str, choices=["yolo", "sam"], default=os.getenv("DETECTION_BACKEND", "yolo"))
    parser.add_argument("--yolo-model", type=str, default="models/yolo_weights/best.pt")
    parser.add_argument("--sam-model", type=str, help="FastSAM 모델 경로")
    parser.add_argument("--sam-prompt", type=str, help="SAM 텍스트 프롬프트")
    parser.add_argument("--conf", type=float, default=float(os.getenv("DETECTION_CONF", 0.25)))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("DETECTION_IMGSZ", 640)))
    parser.add_argument("--device", type=str, default=os.getenv("DEVICE", "cuda"))

    parser.add_argument("--anomaly-backend", type=str, choices=["patchcore", "efficientad"], default=os.getenv("ANOMALY_BACKEND", "patchcore"))
    parser.add_argument("--patchcore-checkpoint", type=str, default="models/patch_core")
    parser.add_argument("--anomaly-threshold", type=float, default=float(os.getenv("ANOMALY_THRESHOLD", 33.08)))

    args = parser.parse_args()

    pipeline = Pipeline(
        det_backend=args.det_backend,
        anomaly_backend=args.anomaly_backend,
        device=args.device,
        det_conf=args.conf,
        det_imgsz=args.imgsz,
        yolo_model=args.yolo_model,
        sam_model=args.sam_model,
        sam_prompt=args.sam_prompt,
        anomaly_threshold=args.anomaly_threshold,
        patchcore_ckpt=args.patchcore_checkpoint,
    )

    save_dir = Path(args.save_dir) if args.save_dir else None

    if args.image:
        out_path = save_dir / Path(args.image).name if save_dir else None
        pipeline.run_image(Path(args.image), out_path)
    elif args.source:
        pipeline.run_directory(Path(args.source), save_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
