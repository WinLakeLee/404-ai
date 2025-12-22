"""
404-AI orchestrator (Flask): proxies image uploads to vision  services.
Inference service: app.py (YOLO + PatchCore)
"""

import os
import json
import requests
import sys
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
import re
import base64
import json

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
        # delegate processing to executor
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

            # 악성코드 검사만 수행, 포맷 검증은 스킵
            malware_check = scan_for_malware(payload)
            if malware_check.get("malware"):
                error_resp = {
                    "status": "error",
                    "source": "mqtt",
                    "error": "malware_detected",
                    "detail": malware_check.get("reason"),
                    "payload_size": len(payload),
                    "timestamp": datetime.now().isoformat(),
                }
                print(f"❌ MQTT 악성 페이로드 차단: {malware_check.get('reason')}")
                try:
                    publish_with_client(
                        _MQTT_CLIENT, error_resp, topic=_OUT_TOPIC, qos=_OUT_QOS
                    )
                except Exception:
                    publish_mqtt(error_resp)
                return

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


def scan_for_malware(data: bytes) -> dict:
    """간단한 악성코드/의심 페이로드 검사.

    완전한 악성코드 검사 도구는 아니며, 일반적으로 악성으로 보이는 파일 헤더나
    스크립트 키워드(예: powershell, cmd, eval, base64_decode 등)를 확인합니다.
    """
    if not data:
        return {"malware": False}

    s = None
    try:
        s = data.decode("utf-8", errors="ignore").lower()
    except Exception:
        s = ""

    # PE/EXE header (Windows), ELF (Linux), Mach-O (macOS)
    if data.startswith(b"MZ") or data.startswith(b"\x7fELF") or data[:4] in (
        b"\xfe\xed\fa\xcf",
        b"\xcf\xfa\xed\xfe",
    ):
        return {"malware": True, "reason": "executable_header"}

    # common script indicators
    suspicious_terms = [
        "powershell",
        "Invoke-Expression".lower(),
        "cmd.exe",
        "eval(",
        "base64",
        "base64_decode",
        "wget ",
        "curl ",
        "exec(",
        "os.system",
    ]
    for t in suspicious_terms:
        if t in s:
            return {"malware": True, "reason": f"suspicious_string:{t}"}

    # zip file may contain executables; mark suspicious if zip and contains exe strings
    if data.startswith(b"PK"):
        if b".exe" in data.lower() or b"powershell" in data.lower():
            return {"malware": True, "reason": "zip_contains_exe_or_script"}

    return {"malware": False}


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
            os.path.join("models", "patch_core"),
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
    """이미지 바이트를 처리하고 모든 서비스 호출

    포맷 검증을 엄격히 수행하지 않고, 먼저 악성코드 여부만 검사합니다.
    """
    # 악성코드 검사: 악성으로 판단되면 즉시 응답
    malware_check = scan_for_malware(data)
    if malware_check.get("malware"):
        return {
            "result": "malware_blocked",
            "detection": {
                "error": "malware_detected",
                "detail": malware_check.get("reason"),
                "payload_size": len(data),
            },
            "timestamp": datetime.now().isoformat(),
        }

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

    # Run internal scratch pipeline if available
    scratch_result = {"skipped": True, "reason": "scratch_pipeline_not_configured"}
    try:
        if _SCRATCH_PIPELINE is not None:
            with tempfile.NamedTemporaryFile(suffix=img_info["extension"], delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            # try path first, then array
            try:
                result, result_image = _SCRATCH_PIPELINE.process_image(tmp_path)
                import cv2

                # Save the result image to disk
                out_path = os.path.join(os.getcwd(), "output.jpg")
                cv2.imwrite(out_path, result_image)

                # Try to open with the default OS image viewer (works for local desktop)
                try:
                    if sys.platform.startswith("win"):
                        os.startfile(out_path)
                    elif sys.platform == "darwin":
                        import subprocess

                        subprocess.Popen(["open", out_path])
                    else:
                        import subprocess

                        subprocess.Popen(["xdg-open", out_path])
                except Exception:
                    # ignore viewer errors in headless/server environments
                    pass

            except Exception as e_path:
                try:
                    pil_img = Image.open(tmp_path).convert("RGB")
                    arr = np.array(pil_img)
                    result, result_image = _SCRATCH_PIPELINE.process_image(arr)
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                    raise e_path

            # encode result_image to jpeg base64
            bytes_img = b""
            try:
                import cv2 as _cv2

                success, buffer = _cv2.imencode(".jpg", result_image)
                if success:
                    bytes_img = buffer.tobytes()
            except Exception:
                try:
                    bio = io.BytesIO()
                    Image.fromarray(result_image).save(bio, format="JPEG")
                    bytes_img = bio.getvalue()
                except Exception:
                    bytes_img = b""

            img_base64 = base64.b64encode(bytes_img).decode("utf-8") if bytes_img else ""

            scratch_result = {
                "success": True,
                "scratch_detected": result.get("scratch_detected"),
                "broken_detected": result.get("broken_detected"),
                "separated_detected": result.get("separated_detected"),
                "anomaly_detected": result.get("anomaly_detected"),
                "car_regions": [
                    {
                        "bbox": r.get("bbox"),
                        "yolo_conf": r.get("yolo_conf"),
                        "class_id": r.get("class_id"),
                        "class_name": r.get("class_name"),
                        "anomaly": r.get("anomaly"),
                        "defect_flags": {
                            "broken_by_yolo": r.get("broken_by_yolo"),
                            "separated_by_yolo": r.get("separated_by_yolo"),
                            "anomaly_by_patchcore": r.get("anomaly_by_patchcore"),
                        },
                    }
                    for r in result.get("car_regions", [])
                ],
                "result_image": f"data:image/jpeg;base64,{img_base64}",
                "pipeline_result": result if isinstance(result, dict) else {},
            }

            # Debug: save pipeline result image and a short pipeline_result summary
            try:
                debug_dir = os.path.join(os.getcwd(), "debug")
                os.makedirs(debug_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_img_path = os.path.join(debug_dir, f"scratch_result_{ts}.jpg")

                # Prefer writing encoded bytes if available, otherwise save result_image directly
                wrote = False
                if bytes_img:
                    try:
                        with open(debug_img_path, "wb") as df:
                            df.write(bytes_img)
                        wrote = True
                        print(f"[DEBUG] saved scratch result image")
                    except Exception:
                        wrote = False

                if not wrote:
                    try:
                        import cv2 as _cv2_save

                        _cv2_save.imwrite(debug_img_path, result_image)
                        wrote = True
                        print(f"[DEBUG] saved scratch result image via cv2")
                    except Exception:
                        try:
                            bio2 = io.BytesIO()
                            Image.fromarray(result_image).convert("RGB").save(bio2, format="JPEG")
                            with open(debug_img_path, "wb") as df2:
                                df2.write(bio2.getvalue())
                            wrote = True
                            print(f"[DEBUG] saved scratch result image via PIL")
                        except Exception as e_save:
                            print(f"[DEBUG] failed to write scratch result image: {e_save}")

                # print compact pipeline result summary
                try:
                    summary = {
                        "scratch_detected": bool(result.get("scratch_detected")),
                        "car_regions_count": len(result.get("car_regions", [])),
                    }
                except Exception:
                    summary = {"scratch_detected": result.get("scratch_detected")}
                print(f"[DEBUG] pipeline summary: {json.dumps(summary)}")
            except Exception as e:
                print(f"[DEBUG] failed to write debug artifacts: {e}")

            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        scratch_result = {"error": "detection_exception", "detail": str(e)}

    overall_result = "ok"
    try:
        if isinstance(scratch_result, dict) and scratch_result.get("scratch_detected"):
            overall_result = "detected"
    except Exception:
        overall_result = "ok"

    return {
        "result": overall_result,
        "detection": scratch_result,
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
    resp = process_image(data, filename=file.filename, mimetype=file.mimetype)
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
