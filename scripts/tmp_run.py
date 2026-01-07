import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import Pipeline

def main():
    imgs = ['img/broken.jpg','img/scratch.jpg']
    device = 'cpu'
    det_backend = os.getenv('DETECTION_BACKEND','yolo')
    anomaly_backend = os.getenv('ANOMALY_BACKEND','patchcore')
    det_conf = float(os.getenv('DETECTION_CONF', 0.25))
    det_imgsz = int(os.getenv('DETECTION_IMGSZ', 640))
    yolo_model = os.getenv('YOLO_MODEL','models/yolo_weights/best.pt')
    sam_model = os.getenv('SAM_MODEL','FastSAM-s.pt')
    sam_prompt = os.getenv('SAM_PROMPT','toy car')
    anomaly_threshold = float(os.getenv('ANOMALY_THRESHOLD', 33.08))
    patchcore_ckpt = os.getenv('PATCHCORE_CHECKPOINT','models/patch_core')

    pl = Pipeline(
        det_backend=det_backend,
        anomaly_backend=anomaly_backend,
        device=device,
        det_conf=det_conf,
        det_imgsz=det_imgsz,
        yolo_model=yolo_model,
        sam_model=sam_model,
        sam_prompt=sam_prompt,
        anomaly_threshold=anomaly_threshold,
        patchcore_ckpt=patchcore_ckpt,
    )

    out_dir = Path('debug/output')
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in imgs:
        p = Path(img)
        if not p.exists():
            print('Image not found:', img)
            continue
        print('\n=== Processing', img)
        save_path = out_dir / p.name
        res = pl.run_image(p, save_path)
        print('Saved annotated image to', save_path)
        print('Results:')
        from pprint import pprint
        pprint(res)

if __name__ == '__main__':
    main()
