"""
404-AI orchestrator (Flask): proxies image uploads to vision and inference services.
Vision service: detector_app.py (FastAPI or Flask-based YOLO/SAM3/RealSense)
Inference service: inference_app.py (GAN reconstruction + PatchCore)
"""

import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
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
from patchcore import AnomalyDetectionPipeline
import tempfile
import numpy as np
from PIL import Image
import io

# Load environment variables from .env (or DOTENV_PATH) before reading any settings
load_dotenv(dotenv_path=os.environ.get("DOTENV_PATH", ".env"), override=True)

app = Flask(__name__)
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("APP_WORKERS", 4)))

# MQTT settings (single broker for pub/sub)
_MQTT_BROKER = os.environ.get("MQTT_BROKER") or "192.168.0.25"

_MQTT_PORT = int(os.environ.get("MQTT_PORT") or 1883)
_MQTT_TLS = (os.environ.get("MQTT_TLS") or "0").lower() in ("1", "true", "yes")
_MQTT_KEEPALIVE = int(os.environ.get("MQTT_KEEPALIVE", 60))
_IN_TOPIC = os.environ.get("IN_MQTT_TOPIC") or "camera01/control"
_OUT_TOPIC = app.config.get("MQTT_TOPIC", "camera01/result")
_OUT_QOS = int(os.environ.get("OUT_MQTT_QOS") or os.environ.get("MQTT_QOS") or 1)

# Detection mode: 'internal' (default), 'external', or 'both'
# - internal: use local `_SCRATCH_PIPELINE` only
# - external: call remote detector service only
# - both: run both and combine results

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
        # delegate processing to executor
        def _task():
            # 이미지 형식 검증
            img_info = validate_image_format(message.payload)

            if not img_info["valid"]:
                error_resp = {
                    "status": "error",
                    "source": "mqtt",
                    "error": img_info.get("error", "알 수 없는 오류"),
                    "payload_size": img_info.get("size", 0),
                    "timestamp": datetime.now().isoformat(),
                }
                print(f"❌ MQTT 이미지 검증 실패: {img_info.get('error')}")
                try:
                    publish_with_client(
                        _MQTT_CLIENT, error_resp, topic=_OUT_TOPIC, qos=_OUT_QOS
                    )
                except Exception:
                    publish_mqtt(error_resp)
                return

            # 유효한 이미지 처리
            filename = f"mqtt_{message.topic.replace('/', '_')}{img_info['extension']}"
            print(
                f"✅ MQTT 이미지 수신: {filename} ({img_info['width']}x{img_info['height']}, {img_info['size']} bytes)"
            )

            resp = process_image_bytes(message.payload, filename, img_info["mime_type"])
            try:
                publish_with_client(_MQTT_CLIENT, resp, topic=_OUT_TOPIC, qos=_OUT_QOS)
            except Exception:
                # fallback to ephemeral publish if persistent client fails
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


# Initialize Scratch Detection Pipeline
print("🚀 Scratch Detection Pipeline 초기화 중...")
try:
    _SCRATCH_PIPELINE = AnomalyDetectionPipeline(
        yolo_model_path=os.environ.get(
            "YOLO_MODEL_PATH",
            os.path.join("models", "yolo_weights", "best.pt"),
        ),
        patchcore_checkpoint=os.environ.get(
            "PATCHCORE_CHECKPOINT",
            os.path.join(
                "models",
                "patch_core",
            ),
        ),
        device=os.environ.get("DEVICE", "cuda"),
        conf_threshold=float(os.environ.get("YOLO_CONF_THRESHOLD", 0.25)),
        anomaly_threshold=float(os.environ.get("ANOMALY_THRESHOLD", 33.08)),
    )
    print("✅ Scratch Detection Pipeline 준비 완료!")
except Exception as e:
    print(f"❌ Scratch Detection Pipeline 초기화 실패: {e}")
    _SCRATCH_PIPELINE = None

# Load configuration object
env = os.environ.get("FLASK_ENV", "development")
# Support both a `config` dict (mapping env->obj) or the config module itself

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


def process_image_bytes(
    data: bytes, filename: str = "image.jpg", mimetype: str | None = None
):
    """이미지 바이트를 처리하고 모든 서비스 호출"""

    # 이미지 형식 검증
    img_info = validate_image_format(data)

    if not img_info["valid"]:
        return {
            "scratch_detection": {
                "error": "invalid_image_format",
                "detail": img_info.get("error", "알 수 없는 오류"),
                "payload_size": img_info.get("size", 0),
            },
            "vision": {"error": "invalid_image_format"},
            "reconstruct": {"error": "invalid_image_format"},
            "patchcore": {"error": "invalid_image_format"},
            "timestamp": datetime.now().isoformat(),
        }

    # 유효한 이미지 정보 로깅
    print(
        f"📸 이미지 수신: {filename} ({img_info['width']}x{img_info['height']}, format={img_info['format']}, size={img_info['size']} bytes)"
    )

    files = {
        "image": (filename or "image.jpg", data, mimetype or img_info["mime_type"])
    }
    # Detector service (HTTP) - only called in 'external' or 'both' modes
    detector_result = {"error": "detector_not_configured"}

    try:
        with tempfile.NamedTemporaryFile(
            suffix=img_info["extension"], delete=False
        ) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        result, result_image = _SCRATCH_PIPELINE.process_image(tmp_path)

        import base64

        # Try to use OpenCV for encoding; if not available, fall back to PIL
        try:
            import cv2 as _cv2

            success, buffer = _cv2.imencode(".jpg", result_image)
            if success:
                bytes_img = buffer.tobytes()
            else:
                bytes_img = b""
        except Exception:
            try:
                bio = io.BytesIO()
                Image.fromarray(result_image).save(bio, format="JPEG")
                bytes_img = bio.getvalue()
            except Exception:
                bytes_img = b""

        img_base64 = base64.b64encode(bytes_img).decode("utf-8") if bytes_img else ""

        scratch_result = {
            "result": [
                (
                    "ok"
                    if not result.get("scratch_detected")
                    and not result.get("broken_detected")
                    and not result.get("separated_detected")
                    and not result.get("anomaly_detected")
                    else "detected"
                )
            ],
            "scratch_detected": result.get("scratch_detected"),
            "broken_detected": result.get("broken_detected"),
            "separated_detected": result.get("separated_detected"),
            "anomaly_detected": result.get("anomaly_detected"),
            "car_regions": [
                {
                    "bbox": r["bbox"],
                    "yolo_conf": r["yolo_conf"],
                    "class_id": r["class_id"],
                    "class_name": r["class_name"],
                    "anomaly": {
                        "is_anomaly": r["anomaly"]["is_anomaly"],
                        "score": round(r["anomaly"]["score"], 4),
                        "threshold": round(r["anomaly"]["threshold"], 4),
                    },
                    "defect_flags": {
                        "broken_by_yolo": r["broken_by_yolo"],
                        "separated_by_yolo": r["separated_by_yolo"],
                        "anomaly_by_patchcore": r["anomaly_by_patchcore"],
                    },
                }
                for r in result.get("car_regions", [])
            ],
            "result_image": f"data:image/jpeg;base64,{img_base64}",
        }

        os.unlink(tmp_path)
    except Exception as e:
        scratch_result = {"error": "scratch_detection_exception", "detail": str(e)}

    # Combine results according to detection mode. Priority: if any configured source reports detection, mark detected.
    overall_result = "ok"
    try:
        if isinstance(scratch_result, dict) and scratch_result.get("scratch_detected"):
            overall_result = "detected"
    except Exception:
        overall_result = "ok"

    return {
        "result": overall_result,
        "detector": detector_result,
        "scratch_detection": scratch_result,
        "timestamp": datetime.now().isoformat(),
    }


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    data = file.read()
    resp = process_image_bytes(data, filename=file.filename, mimetype=file.mimetype)
    if _MQTT_CLIENT is not None:
        try:
            publish_with_client(_MQTT_CLIENT, resp, topic=_OUT_TOPIC, qos=_OUT_QOS)
        except Exception:
            publish_mqtt(resp)
    else:
        publish_mqtt(resp)
    return jsonify(resp)


if __name__ == "__main__":
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, use_reloader=False, host=host, port=port)
