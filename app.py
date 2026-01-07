"""
404-AI 오케스트레이터(Flask): 이미지 업로드를 비전 서비스로 전달합니다.
추론 서비스 엔트리포인트: app.py (YOLO + PatchCore)
"""

import os
import json
import requests
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor
import uuid
from flask import Flask, jsonify, request
from mqtt_utils import (
    publish_with_client,
    publish_mqtt,
    is_client_connected,
    start_paho_listener,
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
import copy
import re
import base64
from _daily_logger import DailyLogger
from itertools import count

# Load environment variables from .env (or DOTENV_PATH) before reading any settings
load_dotenv(dotenv_path=os.environ.get("DOTENV_PATH", ".env"), override=True)

app = Flask(__name__)
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("APP_WORKERS", 4)))
dlogger = DailyLogger()

# Process start time for uptime reporting
_START_TIME = datetime.now()

# Create a per-process upload session directory at app start to avoid
# filename collisions when multiple images arrive with the same timestamps.
# Files will be saved under debug/session_<YYYYMMDD>T<HHMMSS>/ and named 1.ext, 2.ext, ...
_UPLOAD_SESSION = datetime.now().strftime("session_%Y%m%dT%H%M%S")
_UPLOAD_DIR = os.path.join(os.getcwd(), "debug", _UPLOAD_SESSION)
os.makedirs(_UPLOAD_DIR, exist_ok=True)
# thread-safe counter for filenames
_UPLOAD_COUNTER = count(1)
_UPLOAD_COUNTER_LOCK = threading.Lock()

# MQTT settings (single broker for pub/sub)
_MQTT_BROKER = os.environ.get("MQTT_BROKER") or "localhost"

_MQTT_PORT = int(os.environ.get("MQTT_PORT") or 1883)
_MQTT_TLS = (os.environ.get("MQTT_TLS") or "0").lower() in ("1", "true", "yes")
_MQTT_KEEPALIVE = int(os.environ.get("MQTT_KEEPALIVE", 60))
_IN_TOPIC = os.environ.get("IN_MQTT_TOPIC") or "camera01/control"
_OUT_TOPIC = os.environ.get("MQTT_TOPIC") or "camera01/result"
_OUT_QOS = int(os.environ.get("OUT_MQTT_QOS") or os.environ.get("MQTT_QOS") or 1)

# ACK behavior: don't publish ACKs to the main result topic by default.
# Set `MQTT_SEND_ACK=1` to enable ACKs, and `MQTT_ACK_TOPIC` to change the ack topic.
_MQTT_SEND_ACK = (os.environ.get("MQTT_SEND_ACK") or "0").lower() in (
    "1",
    "true",
    "yes",
)
_MQTT_ACK_TOPIC = os.environ.get("MQTT_ACK_TOPIC") or None

# Whether to force `data:image/jpeg;base64,` prefix on published result_image
_FORCE_DATA_URI_PREFIX = (os.environ.get("FORCE_DATA_URI_HEADER") or "1").lower() in (
    "1",
    "true",
    "yes",
)

app.config["MQTT_BROKER_URL"] = _MQTT_BROKER
app.config["MQTT_BROKER_PORT"] = _MQTT_PORT
app.config["MQTT_KEEPALIVE"] = _MQTT_KEEPALIVE
app.config["MQTT_TLS_ENABLED"] = _MQTT_TLS
app.config["MQTT_CLEAN_SESSION"] = True

# MQTT client will be started after `process_image` is defined to avoid
# forward-reference issues with static analyzers (e.g., Pylance).
_MQTT_CLIENT = None


# MQTT 연결 상태를 주기적으로 출력하는 백그라운드 모니터를 시작합니다.
def _start_mqtt_monitor(interval: int = 10):
    def _monitor():
        prev_connected = None
        while True:
            try:
                connected = is_client_connected(_MQTT_CLIENT)
                now = datetime.now().isoformat()
                # Log only on state change (or first check)
                if prev_connected is None:
                    dlogger.log(
                        f"[{now}] MQTT status: connected={connected} broker={_MQTT_BROKER}:{_MQTT_PORT} in_topic={_IN_TOPIC} out_topic={_OUT_TOPIC}",
                        level="info",
                    )
                elif connected != prev_connected:
                    dlogger.log(
                        f"[{now}] MQTT status changed: connected={connected} broker={_MQTT_BROKER}:{_MQTT_PORT}",
                        level="info",
                    )
                prev_connected = connected
            except Exception:
                dlogger.log(
                    f"[{datetime.now().isoformat()}] MQTT status: check failed",
                    level="warning",
                )
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

    # 이미지 안전성 검사: PIL 검증, 치수/픽셀수 제한, 의심 서명 스캔
    try:
        # 환경 변수로 제한값을 조정 가능
        try:
            max_pixels = int(os.environ.get("IMAGE_MAX_PIXELS", "30000000"))
        except Exception:
            max_pixels = 30000000
        Image.MAX_IMAGE_PIXELS = max_pixels

        # 먼저 빠르게 검증(verify)하여 손상 여부 확인
        img = Image.open(io.BytesIO(data))
        img.verify()

        # 실제 크기를 얻기 위해 다시 열기
        img = Image.open(io.BytesIO(data))
        width, height = img.size

        try:
            max_w = int(os.environ.get("IMAGE_MAX_WIDTH", "10000"))
            max_h = int(os.environ.get("IMAGE_MAX_HEIGHT", "10000"))
            min_w = int(os.environ.get("IMAGE_MIN_WIDTH", "1"))
            min_h = int(os.environ.get("IMAGE_MIN_HEIGHT", "1"))
        except Exception:
            max_w, max_h, min_w, min_h = 10000, 10000, 1, 1

        if width <= 0 or height <= 0:
            return {
                "valid": False,
                "size": len(data),
                "error": "이미지 치수가 유효하지 않습니다",
            }

        if width < min_w or height < min_h:
            return {
                "valid": False,
                "size": len(data),
                "error": f"이미지 치수가 너무 작습니다 ({width}x{height})",
            }

        if width > max_w or height > max_h:
            return {
                "valid": False,
                "size": len(data),
                "error": f"이미지 치수가 너무 큽니다 ({width}x{height})",
            }

        # 간단한 서명 스캔: 헤더/앞부분에서 스크립트/실행 파일/압축 아카이브 흔적 검출
        head = data[:4096].lower()
        suspicious_signatures = [
            b"<?php",
            b"<script",
            b"javascript:",
            b"pk\x03\x04",
            b"mz",
            b"#!/bin/sh",
            b"<!doctype html",
        ]
        for sig in suspicious_signatures:
            if sig in head:
                return {
                    "valid": False,
                    "size": len(data),
                    "error": f"의심 서명 발견: {sig.decode('latin1', 'ignore')}",
                }

        return {
            "valid": True,
            "format": fmt,
            "extension": ext,
            "mime_type": mime,
            "size": len(data),
            "width": width,
            "height": height,
        }
    except Image.DecompressionBombError as e:
        return {
            "valid": False,
            "size": len(data),
            "error": f"이미지 디컴프레스 폭탄 의심: {str(e)}",
        }
    except Exception as e:
        return {
            "valid": False,
            "size": len(data),
            "format": fmt,
            "error": f"이미지 검증 실패: {str(e)}",
        }


def aggregate_batch_response(responses: list, include_images: bool = False) -> dict:
    """
    각 이미지별 응답(dict 리스트, `process_image` 반환)을 받아
    HTTP `/detect` 및 MQTT 발행에 사용되는 최상위 집계 응답 구조를 생성합니다.

    반환 dict의 주요 키: result, scratch_count, broken_count,
    separated_count, result_image(리스트), car_regions(리스트), reason, timestamp
    """
    total_scratch_count = 0
    total_broken_count = 0
    total_separated_count = 0
    total_car_regions = []
    result_images = []
    detected_any = False
    reasons = []

    for resp in responses:
        det = resp.get("detection", {})
        total_scratch_count += det.get("scratch_count", 0)
        total_broken_count += det.get("broken_count", 0)
        total_separated_count += det.get("separated_count", 0)
        total_car_regions.extend(det.get("car_regions", []))
        rimg = det.get("result_image")
        if rimg:
            if not isinstance(rimg, str):
                rimg = str(rimg)
            if not rimg.startswith("data:image/jpeg;base64,"):
                rimg = f"data:image/jpeg;base64,{rimg.lstrip()}"
            result_images.append(rimg)
        else:
            result_images.append("")
        if det.get("result") != "pass":
            detected_any = True
        if "reason" in det:
            reasons.append(det.get("reason"))

    aggregated = {
        "result": "defect" if detected_any else "ok",
        "detection": {
            "scratch_count": total_scratch_count,
            "broken_count": total_broken_count,
            "separated_count": total_separated_count,
            "result_image": result_images,
            "car_regions": total_car_regions,
            "reason": "; ".join(reasons) if reasons else None,
        },
        "timestamp": datetime.now().isoformat(),
    }
    # Optionally include per-image detailed responses normalized for publish
    if include_images:
        sanitized = []
        # whitelist of top-level keys to keep per-image
        keep_top = ("id", "result", "timestamp")
        # whitelist of detection fields considered safe
        keep_det = {
            "result",
            "scratch_count",
            "broken_count",
            "separated_count",
            "result_image",
            "car_regions",
            "reason",
            "success",
            "scratch_detected",
            "broken_detected",
            "separated_detected",
            "anomaly_detected",
        }

        for r in responses:
            new_r = {}
            for k in keep_top:
                if k in r:
                    new_r[k] = r[k]

            det = r.get("detection", {}) or {}
            new_det = {}
            for k in keep_det:
                if k in det:
                    v = det.get(k)
                    if k == "result_image":
                        if v:
                            if not isinstance(v, str):
                                v = str(v)
                            if _FORCE_DATA_URI_PREFIX:
                                if not v.startswith("data:image/jpeg;base64,"):
                                    v = f"data:image/jpeg;base64,{v.lstrip()}"
                        else:
                            v = ""
                    new_det[k] = v

            new_r["detection"] = new_det
            sanitized.append(new_r)

        aggregated["images"] = sanitized

    return aggregated


# Initialize Scratch Detection Pipeline (configurable backends)
dlogger.log("🚀 Scratch Detection Pipeline 초기화 중...", level="info")
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
        sam_model=os.environ.get("SAM_MODEL_PATH", "models/sam/FastSAM-s.pt"),
        sam_prompt=os.environ.get("SAM_PROMPT", "car"),
        anomaly_threshold=float(os.environ.get("ANOMALY_THRESHOLD", 33.08)),
        patchcore_ckpt=os.environ.get(
            "PATCHCORE_CHECKPOINT", os.path.join("models", "patch_core")
        ),
    )
    dlogger.log("✅ Scratch Detection Pipeline 준비 완료!", level="info")
except Exception as e:
    dlogger.log(f"❌ Scratch Detection Pipeline 초기화 실패: {e}", level="error")
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
    for pkg, name in (
        ("flask", "flask"),
        ("cv2", "opencv"),
        ("ultralytics", "ultralytics"),
    ):
        try:
            __import__(pkg)
            deps[name] = "installed"
        except Exception:
            deps[name] = "not installed"

    # basic system info
    uptime_sec = int((datetime.now() - _START_TIME).total_seconds())
    disk = shutil.disk_usage(os.getcwd())

    # MQTT status
    mqtt_connected = False
    try:
        mqtt_connected = bool(is_client_connected(_MQTT_CLIENT))
    except Exception:
        mqtt_connected = False

    # Pipeline / detector / anomaly readiness
    pipeline_ok = _SCRATCH_PIPELINE is not None
    detector_info = {}
    anomaly_info = {}
    if pipeline_ok:
        det = getattr(_SCRATCH_PIPELINE, "detector", None)
        if det is not None:
            detector_info["class"] = det.__class__.__name__
            detector_info["model_path"] = getattr(det, "model_path", None)
            detector_info["model_exists"] = bool(
                detector_info.get("model_path")
                and os.path.exists(str(detector_info.get("model_path")))
            )
            detector_info["conf"] = getattr(det, "conf", None)

        an = getattr(_SCRATCH_PIPELINE, "anomaly", None)
        if an is not None:
            anomaly_info["class"] = an.__class__.__name__
            # best-effort detection of readiness
            anomaly_info["ready"] = True
        else:
            anomaly_info["ready"] = False

    status_overall = (
        "healthy"
        if (
            all(v == "installed" for v in deps.values())
            and pipeline_ok
            and mqtt_connected
        )
        else "degraded"
    )

    payload = {
        "status": status_overall,
        "timestamp": datetime.now().isoformat(),
        "uptime_sec": uptime_sec,
        "disk_free_bytes": disk.free,
        "dependencies": deps,
        "mqtt": {
            "connected": mqtt_connected,
            "broker": _MQTT_BROKER,
            "port": _MQTT_PORT,
        },
        "pipeline": {
            "initialized": pipeline_ok,
            "detector": detector_info,
            "anomaly": anomaly_info,
        },
        "env": {
            "detection_backend": os.environ.get("DETECTION_BACKEND"),
            "anomaly_backend": os.environ.get("ANOMALY_BACKEND"),
            "sam_model": os.environ.get("SAM_MODEL_PATH"),
        },
    }

    return jsonify(payload)


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
    dlogger.log(
        f"📸 이미지 수신: {filename} ({img_info['width']}x{img_info['height']}, format={img_info.get('format')}, size={img_info['size']} bytes)",
        level="info",
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
            # 결과 이미지 저장 경로 (디버그용) — use per-process upload session dir
            debug_dir = _UPLOAD_DIR
            # 파일명은 세션 내에서 1,2,3... 형식으로 충돌 회피
            with _UPLOAD_COUNTER_LOCK:
                idx = next(_UPLOAD_COUNTER)
            base_name = os.path.splitext(filename)[0]
            debug_img_path = os.path.join(
                debug_dir, f"{idx:04d}_{base_name}{img_info['extension']}"
            )
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
            # pipeline에 위임된 선별 로직 사용: toy_car_class 선택 (이미지 경로 제공)
            toy_car_class, present_cls = _SCRATCH_PIPELINE.select_toy_car_class(
                car_regions, image_path=Path(tmp_path)
            )
            toy_car_exists = toy_car_class is not None
            dlogger.log(
                f"[DEBUG] present_cls={present_cls} -> toy_car_class={toy_car_class}",
                level="debug",
            )
            if not toy_car_exists:
                scratch_result = {
                    "success": False,
                    "result_image": (
                        f"data:image/jpeg;base64,{img_base64}"
                        if _FORCE_DATA_URI_PREFIX
                        else img_base64
                    ),
                    "car_regions": car_regions,
                    "result": "pass",
                    "reason": f"no car (cls in {sorted(list(present_cls))} -> none of 1,4,3,6)",
                }
                dlogger.log(
                    "[DEBUG] pipeline summary: no car (cls=1,3,4,6) detected, pass",
                    level="debug",
                )
            else:
                # 선택된 toy_car_class에 대해서만 anomaly 실행
                augmented = _SCRATCH_PIPELINE.predict_anomalies_for(
                    Path(tmp_path), car_regions, targets={toy_car_class}
                )
                # 집계: scratch(5), broken(placeholder), separated(6)
                scratch_count = sum(1 for r in augmented if r.get("class_id") == 5)
                broken_count = 0
                separated_count = sum(1 for r in augmented if r.get("class_id") == 6)
                # anomaly_detected는 선택된 toy_car_class 영역에서의 이상 유무
                anomaly_detected = any(
                    (
                        r.get("class_id") == toy_car_class
                        and r.get("anomaly")
                        and r["anomaly"].get("is_anomaly")
                    )
                    for r in augmented
                )
                scratch_result = {
                    "success": True,
                    "result_image": (
                        f"data:image/jpeg;base64,{img_base64}"
                        if _FORCE_DATA_URI_PREFIX
                        else img_base64
                    ),
                    "scratch_detected": bool(scratch_count),
                    "broken_detected": bool(broken_count),
                    "separated_detected": bool(separated_count),
                    "anomaly_detected": bool(anomaly_detected),
                    "scratch_count": scratch_count,
                    "broken_count": broken_count,
                    "separated_count": separated_count,
                    "car_regions": augmented,
                    "result": ("defect" if anomaly_detected else "ok"),
                }
                dlogger.log(
                    f"[DEBUG] pipeline summary: toy_car_class={toy_car_class} anomaly_detected={anomaly_detected}",
                    level="debug",
                )
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        except Exception as e:
            scratch_result = {"error": "detection_exception", "detail": str(e)}
    else:
        scratch_result = {"skipped": True, "reason": "scratch_pipeline_not_configured"}

    # Derive top-level result: prefer the pipeline's per-image `result` when present.
    # Map internal 'pass' -> top-level 'ok'. Fallback to 'defect' if any defect flags present.
    overall_result = "ok"
    try:
        if isinstance(scratch_result, dict):
            det_res = scratch_result.get("result")
            if det_res:
                # Preserve pipeline's explicit result values ('pass', 'ok', 'defect')
                overall_result = det_res
            else:
                if (
                    scratch_result.get("scratch_detected")
                    or scratch_result.get("broken_detected")
                    or scratch_result.get("separated_detected")
                    or scratch_result.get("anomaly_detected")
                ):
                    overall_result = "defect"
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
    # JSON 방식 지원: images: [{image: ...}, ...]
    images = []
    if request.is_json:
        req_json = request.get_json()
        dlogger.log(
            f"[DEBUG] /detect JSON payload: {json.dumps(req_json, ensure_ascii=False)}",
            level="debug",
        )
        images = req_json.get("images", [])
        # images가 없으면 에러
        if not images:
            return jsonify({"error": "no images in JSON payload"}), 400

        def get_image_bytes(imgobj):
            b64 = imgobj.get("image", "")
            if b64.startswith("data:") and "base64," in b64:
                b64 = b64.split("base64,", 1)[1]
            try:
                return base64.b64decode(b64)
            except Exception:
                return b""

        image_datas = [
            (get_image_bytes(img), f"json_image_{i}.png")
            for i, img in enumerate(images)
        ]
    elif "image" in request.files:
        files = request.files.getlist("image")
        image_datas = [
            (file.read(), file.filename or datetime.now().isoformat())
            for file in files
            if file.filename != ""
        ]
        if not image_datas:
            return jsonify({"error": "no image file provided"}), 400
    else:
        return jsonify({"error": "no image data provided"}), 400

    result_images = []
    total_scratch_count = 0
    total_broken_count = 0
    total_separated_count = 0
    total_car_regions = []
    detected_any = False
    reasons = []

    # Collect per-image responses for batching behavior
    responses = []
    non_pass_responses = []
    for data, fname in image_datas:
        if not data:
            continue
        resp = process_image(data, filename=fname, mimetype=None)
        responses.append(resp)

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
            non_pass_responses.append(resp)
        if "reason" in det:
            reasons.append(det["reason"])

    # MQTT publish: if input was a list, publish batched list (aligned to input order)
    # only when NOT all images are 'pass'. For single-image input, publish single
    # response object unless it is 'pass'.
    if len(non_pass_responses) == 0:
        dlogger.log(
            "HTTP /detect: publish skipped — all images result == 'pass'", level="info"
        )
    else:
        try:
            # build a publish payload that includes per-image details
            publish_payload = aggregate_batch_response(responses, include_images=True)

            if _MQTT_CLIENT is not None:
                publish_with_client(
                    _MQTT_CLIENT, publish_payload, topic=_OUT_TOPIC, qos=_OUT_QOS
                )
            else:
                publish_mqtt(publish_payload)
        except Exception:
            try:
                publish_mqtt(publish_payload)
            except Exception:
                dlogger.log(
                    "Failed to publish HTTP-detect batched result", level="error"
                )
    # 최종 응답 dict 구성: MQTT에 발행한 것과 동일한 구조로 반환
    # (per-image 상세 포함 여부는 include_images=True로 통일)
    aggregated = aggregate_batch_response(responses, include_images=True)
    return jsonify(aggregated)


# Start MQTT listener now that `process_image` and helpers are defined.
try:
    _MQTT_CLIENT = start_paho_listener(
        process_image_cb=process_image,
        validate_image_format_cb=validate_image_format,
        aggregate_fn=aggregate_batch_response,
        upload_dir=_UPLOAD_DIR,
        upload_counter=_UPLOAD_COUNTER,
        upload_counter_lock=_UPLOAD_COUNTER_LOCK,
        logger=dlogger,
        executor=_EXECUTOR,
        broker=_MQTT_BROKER,
        port=_MQTT_PORT,
        use_tls=_MQTT_TLS,
        in_topic=_IN_TOPIC,
        out_topic=_OUT_TOPIC,
        out_qos=_OUT_QOS,
        send_ack=_MQTT_SEND_ACK,
        ack_topic=_MQTT_ACK_TOPIC,
    )
except Exception as e:
    _MQTT_CLIENT = None
    dlogger.log(f"MQTT client init failed: {e}", level="error")


if __name__ == "__main__":
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, use_reloader=False, host=host, port=port)
