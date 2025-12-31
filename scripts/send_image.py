"""
Send an image file to the local server and/or MQTT topic.

Usage:
  python scripts/send_image.py path/to/image.png --http --mqtt

Options:
  --http    POST multipart/form-data to http://localhost:5000/detect
  --mqtt    Publish raw bytes to MQTT topic `camera01/control` on localhost:1883

If `paho-mqtt` or `requests` are not installed, the script will fall back
to `urllib` for HTTP and skip MQTT if paho is not available.
"""
import sys
import os
import argparse
import sys
import os
# Ensure project root is on sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from _daily_logger import DailyLogger

dlogger = DailyLogger()


def post_http(paths):
    url = "http://localhost:5000/detect"
    # Try requests first
    try:
        import requests
        files = []
        for path in paths:
            with open(path, "rb") as f:
                files.append(("image", (os.path.basename(path), f.read(), "application/octet-stream")))
        resp = requests.post(url, files=files, timeout=20)
        dlogger.log(f"HTTP status: {resp.status_code}", level="info")
        try:
            dlogger.log(str(resp.json()), level="info")
        except Exception:
            dlogger.log(resp.text[:1000], level="info")
        return
    except Exception:
        pass

    # urllib fallback: send only the first image (no multipart)
    try:
        from urllib import request as urllib_request
        with open(paths[0], "rb") as f:
            data = f.read()
        req = urllib_request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        with urllib_request.urlopen(req, timeout=10) as r:
            body = r.read()
            dlogger.log(f"HTTP status: {r.status}", level="info")
            dlogger.log(body[:1000], level="info")
    except Exception as e:
        dlogger.log(f"HTTP request failed: {e}", level="error")


def publish_mqtt(paths, topic="camera01/control", host="localhost", port=1883):
    try:
        import paho.mqtt.client as mqtt
    except Exception as e:
        dlogger.log(f"paho-mqtt not installed; skipping MQTT publish: {e}", level="info")
        return

    client = mqtt.Client()
    try:
        client.connect(host, port, 60)
    except Exception as e:
        dlogger.log(f"MQTT connect failed: {e}", level="info")
        return

    for path in paths:
        with open(path, "rb") as f:
            data = f.read()
        client.publish(topic, data)
        dlogger.log(f"Published {os.path.basename(path)} to MQTT topic {topic}", level="info")
    client.disconnect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="+", help="path(s) to image file(s)")
    p.add_argument("--http", action="store_true", help="POST to /detect")
    p.add_argument("--mqtt", action="store_true", help="Publish to MQTT topic")
    args = p.parse_args()

    for img in args.image:
        if not os.path.exists(img):
            dlogger.log(f"Image not found: {img}", level="error")
            sys.exit(1)

    if not args.http and not args.mqtt:
        dlogger.log("Specify at least one of --http or --mqtt", level="error")
        sys.exit(1)

    if args.http:
        dlogger.log(f"Posting {len(args.image)} image(s) to http://localhost:5000/detect ...", level="info")
        post_http(args.image)

    if args.mqtt:
        dlogger.log(f"Publishing {len(args.image)} image(s) bytes to MQTT topic camera01/control ...", level="info")
        publish_mqtt(args.image)


if __name__ == "__main__":
    main()
