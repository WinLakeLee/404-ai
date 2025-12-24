import pathlib
from pathlib import Path

import cv2

from detection.sam_detector import SAMDetector


detector = SAMDetector(model_path="FastSAM-s.pt", prompt="toy car", device="cuda", conf=0.85, imgsz=640)


def run_on_image(source: Path) -> None:
    print(f"\n=== {source} ===")

    regions, annotated = detector.detect(source, return_image=True)

    if not regions:
        print("No car detections.")
    else:
        print(f"Detected cars: {len(regions)}")
        for i, r in enumerate(regions):
            bbox = r["bbox"]
            conf = r.get("conf", 0)
            cls = r.get("class_id", 0)
            print(f"[{i}] bbox={bbox}, conf={conf:.3f}, class={cls}")

    # Save color-preserved annotated image
    out_dir = Path("debug")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"sam_{source.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    sources = sorted(pathlib.Path("img").glob("*.jpg"))
    if not sources:
        sources = [pathlib.Path("img/1.jpg")]
    for src in sources:
        run_on_image(src)
