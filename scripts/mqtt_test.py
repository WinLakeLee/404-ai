import base64
import json
import paho.mqtt.client as mqtt
import sys
import os
# Ensure project root is on sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from _daily_logger import DailyLogger

dlogger = DailyLogger()
dlogger = DailyLogger()

# 이미지 파일 경로 리스트
image_paths = ["img/broken.jpg", "img/scratch.jpg"]

# 각 이미지를 base64로 인코딩
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

images = [{"image": encode_image(p)} for p in image_paths]
payload = json.dumps({"images": images})
dlogger.log(f"[DEBUG] MQTT payload: {payload}", level="debug")


# MQTT 브로커 정보
broker = "localhost"
port = 1883
topic = "camera01/control"
result_topic = "camera01/result"

def on_message(client, userdata, msg):
    dlogger.log(f"[RESPONSE] Topic: {msg.topic}, Payload: {msg.payload.decode('utf-8', errors='ignore')}", level="info")

client = mqtt.Client()
client.on_message = on_message
client.connect(broker, port, 60)
client.subscribe(result_topic)
client.loop_start()
client.publish(topic, payload)
import time
time.sleep(2)  # 응답 대기
client.loop_stop()
client.disconnect()