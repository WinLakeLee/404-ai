"""
404-AI orchestrator (Flask): proxies image uploads to vision  services.
Inference service: app.py (YOLO + PatchCore)
"""

import os
import json
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
import uuid
from flask import Flask, jsonify, request
from mqtt_utils import (
    create_paho_client,
    publish_with_client,
    publish_mqtt,
    is_client_connected,
)
import threading
from datetime import datetime
import time
from dotenv import load_dotenv
from pipeline import Pipeline
import tempfile
import numpy as np
from PIL import Image
import io
import re
import base64

# Load environment variables from .env (or DOTENV_PATH) before reading any settings
load_dotenv(dotenv_path=os.environ.get("DOTENV_PATH", ".env"), override=True)

app = Flask(__name__)
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("APP_WORKERS", 4)))

# MQTT settings (single broker for pub/sub)
_MQTT_BROKER = os.environ.get("MQTT_BROKER") or "localhost"

_MQTT_PORT = int(os.environ.get("MQTT_PORT") or 1883)
_MQTT_TLS = (os.environ.get("MQTT_TLS") or "0").lower() in ("1", "true", "yes")
_MQTT_KEEPALIVE = int(os.environ.get("MQTT_KEEPALIVE", 60))
_IN_TOPIC = os.environ.get("IN_MQTT_TOPIC") or "camera01/control"
_OUT_TOPIC = os.environ.get("MQTT_TOPIC") or "camera01/result"
_OUT_QOS = int(os.environ.get("OUT_MQTT_QOS") or os.environ.get("MQTT_QOS") or 1)

app.config["MQTT_BROKER_URL"] = _MQTT_BROKER
app.config["MQTT_BROKER_PORT"] = _MQTT_PORT
app.config["MQTT_KEEPALIVE"] = _MQTT_KEEPALIVE
app.config["MQTT_TLS_ENABLED"] = _MQTT_TLS
app.config["MQTT_CLEAN_SESSION"] = True

# Create a persistent paho client and wire callbacks. If broker is down,
# create_paho_client will return a client (and log connection failure) but
# publishing will be best-effort.
_MQTT_CLIENT = None
try:

    def _on_message(client, userdata, message):

        def _task():
            # Normalize/decode MQTT payload (handle hex, JSON with base64, raw base64)
            def _normalize_payload(data: bytes) -> bytes:
                # If not decodable as text, return raw bytes
                try:
                    s = data.decode("utf-8").strip()
                except Exception:
                    return data

                # JSON object containing a base64 image
                if s.startswith("{") or s.startswith("["):
                    try:
                        obj = json.loads(s)
                        if isinstance(obj, dict):
                            for key in ("image", "payload", "data"):
                                if key in obj and isinstance(obj[key], str):
                                    b64 = obj[key].strip()
                                    # data URI
                                    if b64.startswith("data:") and "base64," in b64:
                                        try:
                                            return base64.b64decode(
                                                b64.split("base64,", 1)[1]
                                            )
                                        except Exception:
                                            pass
                                    try:
                                        return base64.b64decode(b64, validate=True)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                # Hex string (e.g., '7b226361' -> b'{"ca')
                if re.fullmatch(r"[0-9a-fA-F]+", s) and len(s) % 2 == 0:
                    try:
                        return bytes.fromhex(s)
                    except Exception:
                        pass

                # Raw base64 string
                try:
                    return base64.b64decode(s, validate=True)
                except Exception:
                    pass

                return data

            payload = _normalize_payload(message.payload)

            # Optional ACK/heartbeat to indicate message received
            try:
                publish_with_client(
                    _MQTT_CLIENT,
                    {"id": _MQTT_BROKER[0], "timestamp": datetime.now().isoformat()},
                    topic=_OUT_TOPIC,
                    qos=_OUT_QOS,
                )
            except Exception:
                publish_mqtt(payload={"error": ConnectionRefusedError()})

            # 포맷 검증은 강제하지 않음 — 가능한 경우 메타정보를 얻고, 실패 시 기본값을 사용
            img_info = validate_image_format(payload)
            if not img_info.get("valid"):
                img_info = {
                    "valid": True,
                    "format": "jpg",
                    "extension": ".jpg",
                    "mime_type": "image/jpeg",
                    "size": len(payload),
                    "width": 0,
                    "height": 0,
                }

            # 유효한 이미지 처리
            filename = f"mqtt_{message.topic.replace('/', '_')}{img_info['extension']}"
            print(
                f"✅ MQTT 이미지 수신: {filename} ({img_info['width']}x{img_info['height']}, {img_info['size']} bytes)"
            )

            # use normalized payload (may have been decoded from hex/base64/JSON)
            resp = process_image(payload, filename, img_info["mime_type"])
            # result가 'pass'가 아니면 publish
            if resp.get("detection", {}).get("result") != "pass":
                try:
                    publish_with_client(
                        _MQTT_CLIENT, resp, topic=_OUT_TOPIC, qos=_OUT_QOS
                    )
                except Exception:
                    # fallback to ephemeral publish if persistent client fails
                    publish_mqtt(resp)
            else:
                publish_mqtt(resp)

        _EXECUTOR.submit(_task)

    _MQTT_CLIENT = create_paho_client(
        on_message_cb=_on_message,
        broker=_MQTT_BROKER,
        port=_MQTT_PORT,
        use_tls=_MQTT_TLS,
        subscribe_topic=_IN_TOPIC,
        qos=_OUT_QOS,
        start_loop=True,
    )
except Exception as e:
    _MQTT_CLIENT = None
    print(f"MQTT client init failed: {e}")


# Start a background monitor that periodically prints MQTT connection info.
def _start_mqtt_monitor(interval: int = 10):
    def _monitor():
        while True:
            try:
                connected = is_client_connected(_MQTT_CLIENT)
                now = datetime.now().isoformat()
                print(
                    f"[{now}] MQTT status: connected={connected} broker={_MQTT_BROKER}:{_MQTT_PORT} in_topic={_IN_TOPIC} out_topic={_OUT_TOPIC}"
                )
            except Exception:
                print(f"[{datetime.now().isoformat()}] MQTT status: check failed")
            time.sleep(interval)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()


_start_mqtt_monitor()


# Image format validation functions
def validate_image_format(data: bytes) -> dict:
    """
    이미지 바이트 데이터의 형식을 검증하고 파일 확장자, MIME 타입 반환

    Returns:
        {
            'valid': bool,
            'format': str (jpg, png, bmp, etc),
            'extension': str (.jpg, .png, etc),
            'mime_type': str,
            'size': int (bytes),
            'width': int,
            'height': int,
            'error': str (if valid=False)
        }
    """
    if not data or len(data) == 0:
        return {"valid": False, "error": "이미지 데이터가 비어있습니다"}

    # 파일 시그니처(매직 넘버)로 형식 검증
    magic_numbers = {
        b"\xff\xd8\xff": ("jpg", ".jpg", "image/jpeg"),  # JPEG
        b"\x89PNG": ("png", ".png", "image/png"),  # PNG
        b"BM": ("bmp", ".bmp", "image/bmp"),  # BMP
        b"GIF87a": ("gif", ".gif", "image/gif"),  # GIF87a
        b"GIF89a": ("gif", ".gif", "image/gif"),  # GIF89a
    }

    detected_format = None
    for magic, (fmt, ext, mime) in magic_numbers.items():
        if data.startswith(magic):
            detected_format = (fmt, ext, mime)
            break

    if not detected_format:
        return {
            "valid": False,
            "size": len(data),
            "error": f"지원하지 않는 이미지 형식입니다 (처음 4바이트: {data[:4].hex()})",
        }

    fmt, ext, mime = detected_format

    # PIL로 이미지 검증 및 크기 확인
    try:
        img = Image.open(io.BytesIO(data))
        width, height = img.size

        return {
            "valid": True,
            "format": fmt,
            "extension": ext,
            "mime_type": mime,
            "size": len(data),
            "width": width,
            "height": height,
        }
    except Exception as e:
        return {
            "valid": False,
            "size": len(data),
            "format": fmt,
            "error": f"이미지 검증 실패: {str(e)}",
        }


# Initialize Scratch Detection Pipeline (configurable backends)
print("🚀 Scratch Detection Pipeline 초기화 중...")
try:
    _SCRATCH_PIPELINE = Pipeline(
        det_backend=os.environ.get("DETECTION_BACKEND", "yolo"),
        anomaly_backend=os.environ.get("ANOMALY_BACKEND", "patchcore"),
        device=os.environ.get("DEVICE", "cuda"),
        det_conf=float(os.environ.get("DETECTION_CONF", 0.25)),
        det_imgsz=int(os.environ.get("DETECTION_IMGSZ", 640)),
        yolo_model=os.environ.get(
            "YOLO_MODEL_PATH", os.path.join("models", "yolo_weights", "best.pt")
        ),
        sam_model=os.environ.get("SAM_MODEL_PATH", "FastSAM-s.pt"),
        sam_prompt=os.environ.get("SAM_PROMPT", "car"),
        anomaly_threshold=float(os.environ.get("ANOMALY_THRESHOLD", 33.08)),
        patchcore_ckpt=os.environ.get(
            "PATCHCORE_CHECKPOINT", os.path.join("models", "patch_core")
        ),
    )
    print("✅ Scratch Detection Pipeline 준비 완료!")
except Exception as e:
    print(f"❌ Scratch Detection Pipeline 초기화 실패: {e}")
    _SCRATCH_PIPELINE = None

# Load configuration object
env = os.environ.get("FLASK_ENV", "development")


@app.route("/")
def index():
    return jsonify(
        {
            "message": "Welcome to 404-AI Factory Defect Recognition System",
            "status": "running",
        }
    )


@app.route("/health")
def health():
    deps = {}
    try:
        import flask  # noqa: F401

        deps["flask"] = "installed"
    except ImportError:
        deps["flask"] = "not installed"
    try:
        import cv2  # noqa: F401

        deps["opencv"] = "installed"
    except ImportError:
        deps["opencv"] = "not installed"
    try:
        import ultralytics  # noqa: F401

        deps["ultralytics"] = "installed"
    except ImportError:
        deps["ultralytics"] = "not installed"

    all_ok = all(v == "installed" for v in deps.values())
    return jsonify(
        {"status": "healthy" if all_ok else "degraded", "dependencies": deps}
    )


def process_image(
    data: bytes, filename: str = "image.jpg", mimetype: str | None = None
):
    """
    이미지 바이트를 처리하고 모든 서비스 호출
    """
    # 가능한 경우 포맷/크기 정보를 얻되, 실패해도 계속 진행
    img_info = validate_image_format(data)
    if not img_info.get("valid"):
        img_info = {
            "valid": True,
            "format": "jpg",
            "extension": ".jpg",
            "mime_type": "image/jpeg",
            "size": len(data),
            "width": 0,
            "height": 0,
        }

    # 유효한(또는 기본값) 이미지 정보 로깅
    print(
        f"📸 이미지 수신: {filename} ({img_info['width']}x{img_info['height']}, format={img_info.get('format')}, size={img_info['size']} bytes)"
    )

    # 실제 감지(inference) 수행
    if "_SCRATCH_PIPELINE" in globals() and _SCRATCH_PIPELINE is not None:
        try:
            # 임시 파일로 저장 후 run_image 사용
            with tempfile.NamedTemporaryFile(
                suffix=img_info["extension"], delete=False
            ) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            # 결과 이미지 저장 경로 (디버그용)
            debug_dir = os.path.join(os.getcwd(), "debug")
            os.makedirs(debug_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 파일명에 원본 이미지 이름(확장자 제거) 추가하여 중복 방지
            base_name = os.path.splitext(filename)[0]
            debug_img_path = os.path.join(debug_dir, f"scratch_result_{ts}_{base_name}.jpg")
            # 감지 수행
            from pathlib import Path

            results = _SCRATCH_PIPELINE.run_image(Path(tmp_path), Path(debug_img_path))
            # results: List[Dict], 시각화 이미지는 debug_img_path에 저장됨
            # 결과 이미지 base64 인코딩
            bytes_img = b""
            try:
                with open(debug_img_path, "rb") as f:
                    bytes_img = f.read()
            except Exception:
                bytes_img = b""
            img_base64 = (
                base64.b64encode(bytes_img).decode("utf-8") if bytes_img else ""
            )
            # 결과 집계
            car_regions = results if isinstance(results, list) else []
            # 자동차(cls=1,2)가 감지되지 않으면 pass 처리
            car_detected = any(r.get("class_id") in (1, 2) for r in car_regions)
            if not car_detected:
                scratch_result = {
                    "success": False,
                    "result_image": f"data:image/jpeg;base64,{img_base64}",
                    "car_regions": car_regions,
                    "result": "pass",
                    "reason": "no car (cls=1,2) detected",
                }
                print("[DEBUG] pipeline summary: no car (cls=1,2) detected, pass")
            else:
                scratch_count = sum(1 for r in car_regions if r.get("class_id") == 5)
                broken_count = 0  # 필요시 클래스별로 집계
                separated_count = sum(1 for r in car_regions if r.get("class_id") == 6)
                scratch_result = {
                    "success": True,
                    "result_image": f"data:image/jpeg;base64,{img_base64}",
                    "scratch_detected": bool(scratch_count),
                    "broken_detected": bool(broken_count),
                    "separated_detected": bool(separated_count),
                    "anomaly_detected": bool(scratch_count),
                    "scratch_count": scratch_count,
                    "broken_count": broken_count,
                    "separated_count": separated_count,
                    "car_regions": car_regions,
                    "result": ("defect" if scratch_count > 0 else "ok"),
                }
                print(f"[DEBUG] pipeline summary: scratch_count={scratch_count}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        except Exception as e:
            scratch_result = {"error": "detection_exception", "detail": str(e)}
    else:
        scratch_result = {"skipped": True, "reason": "scratch_pipeline_not_configured"}

    overall_result = "ok"
    try:
        if isinstance(scratch_result, dict) and (
            scratch_result.get("scratch_detected")
            or scratch_result.get("broken_detected")
            or scratch_result.get("separated_detected")
            or scratch_result.get("anomaly_detected")
        ):
            overall_result = "detected"
    except Exception:
        overall_result = "ok"

    return {
        "id": str(uuid.uuid4()),
        "result": overall_result,
        "detection": scratch_result,
        "timestamp": datetime.now().isoformat(),
    }


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    files = request.files.getlist("image")
    result_images = []
    total_scratch_count = 0
    total_broken_count = 0
    total_separated_count = 0
    total_car_regions = []
    detected_any = False
    reasons = []
    for file in files:
        if file.filename == "":
            continue
        data = file.read()
        resp = process_image(
            data,
            filename=file.filename or datetime.now().isoformat(),
            mimetype=file.mimetype,
        )
        det = resp.get("detection", {})
        car_regions = det.get("car_regions", [])
        # car_regions가 없으면 이 이미지는 완전히 무시
        if not car_regions:
            continue
        # result_image만 리스트로 모음 (항상 prefix 보장)
        img = det.get("result_image")
        if img:
            if not img.startswith("data:image/jpeg;base64,"):
                img = f"data:image/jpeg;base64,{img.lstrip()}"
            result_images.append(img)
        else:
            result_images.append("")
        # count 합산
        total_scratch_count += det.get("scratch_count", 0)
        total_broken_count += det.get("broken_count", 0)
        total_separated_count += det.get("separated_count", 0)
        total_car_regions.extend(car_regions)
        if det.get("result") != "pass":
            detected_any = True
        if "reason" in det:
            reasons.append(det["reason"])
        # result가 'pass'가 아니면 publish
        if det.get("result") != "pass":
            if _MQTT_CLIENT is not None:
                try:
                    publish_with_client(
                        _MQTT_CLIENT, resp, topic=_OUT_TOPIC, qos=_OUT_QOS
                    )
                except Exception:
                    publish_mqtt(resp)
            else:
                publish_mqtt(resp)
    # 최종 응답 dict 구성
    response = {
        "result": "defect" if detected_any else "ok",
        "scratch_count": total_scratch_count,
        "broken_count": total_broken_count,
        "separated_count": total_separated_count,
        "result_image": result_images,
        "car_regions": total_car_regions,
        "reason": "; ".join(reasons) if reasons else None,
        "timestamp": datetime.now().isoformat(),
    }
    return jsonify(response)


if __name__ == "__main__":
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, use_reloader=False, host=host, port=port)
