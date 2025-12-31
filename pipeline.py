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
import numpy as np
from _daily_logger import DailyLogger

from detection.yolo_detector import YOLODetector
from detection.sam_detector import SAMDetector
from anomaly.patchcore_backend import PatchCoreBackend
from anomaly.PDN_backend import PDNBackend


class DetectionFactory:
    @staticmethod
    def create(
        backend: str,
        device: str,
        conf: float,
        imgsz: int,
        sam_model: Optional[str],
        sam_prompt: Optional[str],
        yolo_model: str,
    ):
        backend = backend.lower()
        if backend == "sam":
            return SAMDetector(
                model_path=sam_model or "models/sam/FastSAM-s.pt",
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
    def create(
        backend: str,
        device: str,
        patchcore_ckpt: Optional[str],
        anomaly_threshold: float,
    ):
        backend = backend.lower()
        if backend == "efficientad":
            return PDNBackend(
                checkpoint_path=os.path.join(
                    patchcore_ckpt or "models", "Efficient_AD", "best_checkpoint.pth"
                ),
                device=device,
                anomaly_threshold=anomaly_threshold,
            )
        return PatchCoreBackend(
            checkpoint_dir=patchcore_ckpt or "models/patch_core",
            device=device,
            anomaly_threshold=anomaly_threshold,
        )


class Pipeline:
    def run_images(self, image_paths, save_dir: Optional[Path] = None):
        """
        여러 장 또는 한 장의 이미지를 받아 image:[] 형태로 처리
        image_paths: Path 또는 List[Path]
        """
        if isinstance(image_paths, (str, Path)):
            image_paths = [Path(image_paths)]
        all_results = []
        image_names = []
        for image_path in image_paths:
            save_path = (save_dir / Path(image_path).name) if save_dir else None
            results = self.run_image(Path(image_path), save_path)
            all_results.append({"image": str(image_path), "results": results})
            image_names.append(str(image_path))
        # 전체 결과를 한 번에 저장 (image: [] 형태)
        self.logger.save_result({"images": image_names, "results": all_results})
        return all_results

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
        self.logger = DailyLogger(log_dir="logs", log_prefix="pipeline")
        # toy_car 선택 시 클래스 3/6에 대해 최소 박스 면적 비율 요구 (env TOY_CAR_MIN_AREA_RATIO)
        self.toycar_min_area_ratio = float(os.getenv("TOY_CAR_MIN_AREA_RATIO", 0.0))
        # 클래스별 신뢰도 매핑: env `DETECTION_CONF_MAP` 을 파싱합니다. 형식 예: "1:0.4,2:0.6,4:0.3"
        conf_map_raw = os.getenv("DETECTION_CONF_MAP", "")
        self.class_conf_map = {}
        if conf_map_raw:
            try:
                for item in conf_map_raw.split(","):
                    if not item.strip():
                        continue
                    k, v = item.split(":", 1)
                    self.class_conf_map[int(k.strip())] = float(v.strip())
                self.logger.log(f"Loaded class_conf_map={self.class_conf_map}")
            except Exception as e:
                self.logger.log(f"Failed to parse DETECTION_CONF_MAP='{conf_map_raw}': {e}", level="error")
        # NMS IoU 임계값 (클래스 무관). 0이면 NMS 비활성화
        self.nms_iou = float(os.getenv("DETECTION_NMS_IOU", 0.5))

    def run_image(self, image_path: Path, save_path: Optional[Path] = None):
        self.logger.log(f"[Input] {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.log(f"이미지를 로드할 수 없습니다: {image_path}", level="error")
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        # 1. YOLO 탐색
        # If per-class thresholds exist and are lower than detector's configured conf,
        # request the detector to run with the lower threshold so pipeline filtering can take effect.
        conf_override = None
        if self.class_conf_map:
            try:
                min_thr = min(self.class_conf_map.values())
                det_conf = getattr(self.detector, "conf", None)
                if det_conf is not None and min_thr < float(det_conf):
                    conf_override = float(min_thr)
            except Exception:
                conf_override = None
        regions = self.detector.detect(image_path, conf_override=conf_override)
        self.logger.log(f"Detected regions (pre-filter): {len(regions)}")
        # 클래스별 임계값이 설정되어 있으면 필터링 적용
        if self.class_conf_map:
            filtered = []
            for r in regions:
                cls = r.get("class_id")
                conf = float(r.get("conf") or 0.0)
                thr = self.class_conf_map.get(cls)
                if thr is None:
                    filtered.append(r)
                else:
                    if conf >= thr:
                        filtered.append(r)
                    else:
                        self.logger.log(f"Filtered out cls={cls} conf={conf:.3f} < thr={thr}")
            regions = filtered
        self.logger.log(f"Detected regions (post-filter): {len(regions)}")

        # Apply class-agnostic NMS to prevent overlapping boxes
        if regions and self.nms_iou and float(self.nms_iou) > 0.0:
            boxes = [r.get("bbox") for r in regions]
            scores = [float(r.get("conf") or 0.0) for r in regions]
            idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
            keep = []
            while idxs:
                cur = idxs.pop(0)
                keep.append(cur)
                rem = []
                for i in idxs:
                    # compute IoU
                    xA = max(boxes[cur][0], boxes[i][0])
                    yA = max(boxes[cur][1], boxes[i][1])
                    xB = min(boxes[cur][2], boxes[i][2])
                    yB = min(boxes[cur][3], boxes[i][3])
                    interW = max(0, xB - xA)
                    interH = max(0, yB - yA)
                    interArea = interW * interH
                    areaA = max(0, boxes[cur][2] - boxes[cur][0]) * max(0, boxes[cur][3] - boxes[cur][1])
                    areaB = max(0, boxes[i][2] - boxes[i][0]) * max(0, boxes[i][3] - boxes[i][1])
                    denom = float(areaA + areaB - interArea)
                    iou_val = interArea / denom if denom > 0 else 0.0
                    if iou_val <= float(self.nms_iou):
                        rem.append(i)
                idxs = rem
            regions = [regions[i] for i in keep]
            self.logger.log(f"Detected regions (after NMS): {len(regions)}")

        # 2. 차량 존재 여부 판단: select_toy_car_class에 위임
        toy_car_class, present_cls = self.select_toy_car_class(regions, image_shape=image.shape[:2])
        toy_car_exists = toy_car_class is not None
        # SHOW_IGNORED_CLASSES 환경변수가 true이면 cls 3,4를 임시로 legend와 bbox에 표시
        show_ignored = os.getenv("SHOW_IGNORED_CLASSES", "true").lower() in ("1", "true", "yes")

        class_colors = {
            0: (0, 255, 0),  # green
            1: (0, 0, 255),  # red
            2: (255, 0, 0),  # blue
            3: (0, 255, 255),  # yellow
            4: (255, 0, 255),  # magenta
            5: (255, 255, 0),  # cyan
            6: (128, 128, 128),  # gray
        }
        # 클래스 이름 매핑: 1->toy_car, 4->case, 3->car_floor, 2->broken, 5->scratch, 6->separated
        class_names = {1: "toy_car", 4: "case", 3: "car_floor", 2: "broken", 5: "scratch", 6: "separated"}

        # pipeline은 이제 감지 결과(클래스, bbox, conf)만 반환합니다.
        # 애플리케이션 레이어에서 이미지 수준 판정이나 anomaly 실행을 담당합니다.
        results: List[Dict] = []
        for i, reg in enumerate(regions):
            bbox = reg["bbox"]
            cls_id = reg.get("class_id", 0)
            out = {
                "bbox": bbox,
                "conf": reg.get("conf"),
                "class_id": cls_id,
            }
            self.logger.log(f"[{i}] cls={cls_id} (detected, anomaly skipped in pipeline)")
            results.append(out)

            # bbox 그리기 규칙:
            # - 항상 cls 5,6(scratch/separated)는 그림
            # - cls 1(toy_car)와 cls 4(case)는 표시 (단, toy_car가 감지된 경우 floor/case는 무시)
            # - cls 3(car_floor)는 cls 4가 없을 때만 표시
            # - cls 2는 기본적으로 무시(표시하려면 SHOW_IGNORED_CLASSES)
            draw_cls3 = (cls_id == 3 and 4 not in {r.get("class_id") for r in regions})
            # if toy_car exists, ignore floor(3) and case(4) for drawing
            if toy_car_exists and cls_id in (3, 4):
                continue
            if (
                cls_id in (5, 6)
                or cls_id in (1, 4)
                or draw_cls3
                or (show_ignored and cls_id == 2)
            ):
                x1, y1, x2, y2 = bbox
                color = class_colors.get(cls_id, (0, 255, 0))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # legend(범례) 추가: 이미지 우측하단 box, 각 class별 색상/이름
        h, w = image.shape[:2]
        legend_w = int(w * 0.3)
        # legend에 표시할 클래스: 이미지에서 감지된 모든 클래스 ID를 표시
        legend_cls_ids = {reg.get("class_id") for reg in regions if reg.get("class_id") is not None}
        # 사용자 규칙: 기본적으로 class 3(car_floor)은 legend에서 제외
        # SHOW_IGNORED_CLASSES가 true이면 포함시켜 임시 표시
        if not show_ignored:
            legend_cls_ids = {cid for cid in legend_cls_ids if cid not in (3,)}
        # class 3과 4가 모두 있을 경우 4만 남겨서 toy_car로 단일 표기
        if 3 in legend_cls_ids and 4 in legend_cls_ids:
            legend_cls_ids.discard(3)
        legend_cls_ids = sorted(legend_cls_ids)
        n_legend = len(legend_cls_ids)
        if n_legend == 0:
            n_legend = 1  # 최소 높이 확보
        line_height = max(h // 20, 18)
        legend_h = int(
            line_height * (n_legend + 1.2)
        )  # 표시할 class 수에 따라 세로 크기 자동 조정
        legend_x1 = w - legend_w - 10
        legend_y1 = h - legend_h - 10
        legend_x2 = w - 10
        legend_y2 = h - 10
        cv2.rectangle(
            image, (legend_x1, legend_y1), (legend_x2, legend_y2), (255, 255, 255), -1
        )  # legend 배경
        cv2.rectangle(
            image, (legend_x1, legend_y1), (legend_x2, legend_y2), (0, 0, 0), 2
        )  # legend 테두리
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = (line_height / 32.0) * (2 / 3)
        thickness = max(int(font_scale * 2), 1)
        y_offset = legend_y1 + line_height
        color_box_w = max(int(line_height * 0.7), 12)
        color_box_h = max(int(line_height * 0.7), 12)
        for cid in legend_cls_ids:
            # If class 4 exists alone (no class 1), show it as 'toy_car'
            if cid == 4 and 1 not in present_cls:
                cname = "toy_car"
            elif cid == 1:
                cname = "toy_car"
            else:
                cname = class_names.get(cid, str(cid))
            color = class_colors.get(cid, (0, 255, 0))
            box_y1 = y_offset - color_box_h // 2
            box_y2 = y_offset + color_box_h // 2 - 1
            cv2.rectangle(
                image,
                (legend_x1 + 15, box_y1),
                (legend_x1 + 15 + color_box_w, box_y2),
                color,
                -1,
            )
            cv2.putText(
                image,
                f"{cname}",
                (legend_x1 + 25 + color_box_w, y_offset + line_height // 3),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )
            y_offset += line_height

        # regions가 비어 있어도 결과 이미지를 항상 저장
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image)
            self.logger.log(f"Saved: {save_path}")
        else:
            cv2.imshow("result", image)
            cv2.waitKey(1)

        # 결과를 날짜별 파일에 자동 저장 (이미지별 원시 감지 결과)
        self.logger.save_result({"image": str(image_path), "results": results})
        return results

    def select_toy_car_class(self, regions: List[Dict], image_shape: Optional[tuple] = None, image_path: Optional[Path] = None):
        """주어진 감지 결과에서 toy_car_class 우선순위(1,4,3,6)를 적용하여 선택합니다.
        선택 시 환경변수 `TOY_CAR_MIN_AREA_RATIO`가 설정되어 있으면 클래스 3 또는 6에 대해
        해당 비율 이상인 박스가 존재해야 선택합니다.
        반환: (toy_car_class or None, present_cls set)
        """
        present_cls = {r.get("class_id") for r in regions if r.get("class_id") is not None}
        # 기본 우선순위
        toy_car_class = None
        if 1 in present_cls:
            toy_car_class = 1
        elif 4 in present_cls:
            toy_car_class = 4
        elif 3 in present_cls or 6 in present_cls:
            # 3 또는 6은 이미지 내에서 충분히 큰 박스가 있을 때만 차량으로 간주할 수 있음
            if self.toycar_min_area_ratio and (3 in present_cls or 6 in present_cls):
                # determine image area
                ih, iw = None, None
                if image_shape:
                    ih, iw = image_shape[0], image_shape[1]
                elif image_path:
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        ih, iw = img.shape[:2]
                if ih and iw:
                    img_area = ih * iw
                    # find max area among cls 3 or 6
                    max_area = 0
                    max_cls = None
                    for r in regions:
                        cid = r.get("class_id")
                        if cid in (3, 6):
                            x1, y1, x2, y2 = r.get("bbox", [0, 0, 0, 0])
                            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                            if area > max_area:
                                max_area = area
                                max_cls = cid
                    if max_area >= img_area * self.toycar_min_area_ratio:
                        toy_car_class = max_cls
                    else:
                        toy_car_class = None
                else:
                    # 이미지 크기를 알 수 없으면 보수적으로 선택하지 않음
                    toy_car_class = None
            else:
                # no area restriction -> choose 3 if present else 6
                if 3 in present_cls:
                    toy_car_class = 3
                elif 6 in present_cls:
                    toy_car_class = 6
        # else toy_car_class remains None
        self.logger.log(f"[SELECT] present_cls={present_cls} -> toy_car_class={toy_car_class}")
        return toy_car_class, present_cls

    def predict_anomalies_for(self, image_path: Path, regions: List[Dict], targets: Optional[set] = None):
        """주어진 regions에 대해 anomaly backend를 실행하여 'anomaly' 필드를 추가한 결과를 반환합니다.
        targets: 클래스 ID 집합만 처리 (None이면 모든 클래스 처리)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        augmented = []
        for i, r in enumerate(regions):
            out = dict(r)
            cls_id = out.get("class_id")
            if targets is None or cls_id in targets:
                if self.anomaly and (self.anomaly_backend in ("patchcore", "efficientad")):
                    anomaly = self.anomaly.predict_crop(image, out["bbox"], threshold=self.anomaly_threshold)
                    out["anomaly"] = anomaly
                    tag = "DEFECT" if anomaly.get("is_anomaly") else "OK"
                    self.logger.log(f"[anomaly] idx={i} cls={cls_id} score={anomaly.get('score'):.2f} -> {tag}")
                else:
                    out["anomaly"] = None
            else:
                out["anomaly"] = None
            augmented.append(out)
        return augmented

    def run_directory(self, source_dir: Path, save_dir: Optional[Path] = None):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in exts])
        if not images:
            self.logger.log(
                f"❌ 디렉토리에 이미지가 없습니다: {source_dir}", level="error"
            )
            return
        for img in images:
            out = save_dir / img.name if save_dir else None
            self.run_image(img, out)


def main():
    parser = argparse.ArgumentParser(description="Detection + Anomaly pipeline")
    parser.add_argument("--image", type=str, help="단일 이미지 경로")
    parser.add_argument("--source", type=str, help="이미지 디렉토리 경로")
    parser.add_argument("--save-dir", type=str, help="결과 저장 디렉토리")

    parser.add_argument(
        "--det-backend",
        type=str,
        choices=["yolo", "sam"],
        default=os.getenv("DETECTION_BACKEND", "yolo"),
    )
    parser.add_argument("--yolo-model", type=str, default="models/yolo_weights/best.pt")
    parser.add_argument("--sam-model", type=str, help="FastSAM 모델 경로")
    parser.add_argument("--sam-prompt", type=str, help="SAM 텍스트 프롬프트")
    parser.add_argument(
        "--conf", type=float, default=float(os.getenv("DETECTION_CONF", 0.25))
    )
    parser.add_argument(
        "--imgsz", type=int, default=int(os.getenv("DETECTION_IMGSZ", 640))
    )
    parser.add_argument("--device", type=str, default=os.getenv("DEVICE", "cuda"))

    parser.add_argument(
        "--anomaly-backend",
        type=str,
        choices=["patchcore", "efficientad"],
        default=os.getenv("ANOMALY_BACKEND", "patchcore"),
    )
    parser.add_argument("--patchcore-checkpoint", type=str, default="models/patch_core")
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=float(os.getenv("ANOMALY_THRESHOLD", 33.08)),
    )

    args = parser.parse_args()

