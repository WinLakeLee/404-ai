import os
import json
import uuid
import logging
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# Single global executor for background MQTT publishes
_MQTT_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _do_publish(payload: dict) -> bool:
    try:
        import paho.mqtt.client as mqtt
    except Exception:
        return False

    broker = os.environ.get('MQTT_BROKER', 'localhost')
    port = int(os.environ.get('MQTT_PORT', 1883))
    topic = os.environ.get('MQTT_TOPIC', 'camera01/result')
    qos = int(os.environ.get('MQTT_QOS', 1))
    username = os.environ.get('MQTT_USERNAME')
    password = os.environ.get('MQTT_PASSWORD')
    use_tls = os.environ.get('MQTT_TLS', '0').lower() in ('1', 'true', 'yes')

    try:
        client = mqtt.Client()
        if username and password:
            client.username_pw_set(username, password)
        if use_tls:
            try:
                client.tls_set()
            except Exception:
                pass
        client.connect(broker, port, 60)
        client.loop_start()
        payload_str = json.dumps(payload, ensure_ascii=False)
        client.publish(topic, payload_str, qos=qos)
        client.loop_stop()
        client.disconnect()
        return True
    except Exception:
        return False


def publish_mqtt(payload: dict, async_send: bool = True) -> bool:
    """Publish payload to MQTT using paho-mqtt (legacy helper).

    - Adds `id` (UUID4) and `timestamp` (ISO8601 UTC) to payload if missing.
    - If `async_send` is True, schedule publish on background thread and
      return True immediately (best-effort). If False, publish synchronously
      and return True/False depending on success.
    """
    if not isinstance(payload, dict):
        try:
            payload = dict(payload)
        except Exception:
            payload = {'value': str(payload)}

    if 'id' not in payload:
        payload['id'] = str(uuid.uuid4())
    if 'timestamp' not in payload:
        payload['timestamp'] = datetime.now(timezone.utc).isoformat()

    if async_send:
        try:
            _MQTT_EXECUTOR.submit(_do_publish, payload)
            return True
        except Exception:
            # fallback to synchronous attempt
            return _do_publish(payload)
    else:
        return _do_publish(payload)


def init_flask_mqtt(app, mqtt_instance=None):
    """Initialize Flask-MQTT callbacks on a Flask `app`.

    - `app`: Flask application instance with optional MQTT config.
    - `mqtt_instance`: existing `flask_mqtt.Mqtt` instance or None.

    Returns the `Mqtt` instance with `on_connect`/`on_message` handlers attached.
    If initialization fails (broker unreachable or missing libs), returns a
    NoopMqtt fallback with `publish` and `subscribe` methods that only log.
    """
    try:
        from flask_mqtt import Mqtt
    except Exception:
        raise RuntimeError('flask-mqtt is required for Flask MQTT integration')

    try:
        mqtt = mqtt_instance or Mqtt(app)
    except Exception:
        logging.exception('Failed to initialize flask-mqtt; using NoopMqtt fallback')

        class NoopMqtt:
            def __init__(self):
                self.client = None

            def publish(self, topic, payload, qos=0):
                logging.info('NoopMqtt.publish called: topic=%s payload=%s qos=%s', topic, payload, qos)

            def subscribe(self, topic, qos=0):
                logging.info('NoopMqtt.subscribe called: topic=%s qos=%s', topic, qos)

        return NoopMqtt()

    # If mqtt is a real flask_mqtt.Mqtt instance, attach callbacks if available.
    try:
        on_connect_decorator = getattr(mqtt, 'on_connect', None)
    except Exception:
        on_connect_decorator = None

    if callable(on_connect_decorator):
        @mqtt.on_connect()
        def _on_connect(client, userdata, flags, rc):
            try:
                if rc == 0:
                    logging.info('MQTT connected successfully (rc=0)')
                else:
                    logging.warning('MQTT connect returned non-zero rc=%s', rc)

                topic = app.config.get('MQTT_SUBSCRIBE_TOPIC', os.environ.get('MQTT_SUBSCRIBE_TOPIC', '404ai/commands'))
                qos = int(app.config.get('MQTT_QOS', os.environ.get('MQTT_QOS', 1)))
                mqtt.subscribe(topic, qos)
                logging.info('Subscribed to topic %s (qos=%s)', topic, qos)
            except Exception:
                logging.exception('Exception in flask-mqtt on_connect')

    try:
        on_message_decorator = getattr(mqtt, 'on_message', None)
    except Exception:
        on_message_decorator = None

    if callable(on_message_decorator):
        @mqtt.on_message()
        def _on_message(client, userdata, message):
            try:
                payload_raw = message.payload.decode('utf-8', errors='replace')
                try:
                    payload = json.loads(payload_raw)
                except Exception:
                    payload = payload_raw
                logging.info('MQTT message received: topic=%s payload=%s', message.topic, payload)
            except Exception:
                logging.exception('Exception in flask-mqtt on_message')

    return mqtt


def publish_via_flask(mqtt_client, payload: dict, topic: str = None, qos: int = None) -> bool:
    """Publish `payload` using a `flask_mqtt.Mqtt` client instance.

    This is a convenience wrapper that ensures `id` and `timestamp` are present.
    """
    if mqtt_client is None:
        raise RuntimeError('mqtt_client (flask_mqtt.Mqtt) is required')

    if not isinstance(payload, dict):
        try:
            payload = dict(payload)
        except Exception:
            payload = {'value': str(payload)}

    if 'id' not in payload:
        payload['id'] = str(uuid.uuid4())
    if 'timestamp' not in payload:
        payload['timestamp'] = datetime.now(timezone.utc).isoformat()

    topic = topic or os.environ.get('MQTT_TOPIC', '404ai/detections')
    qos = int(qos or os.environ.get('MQTT_QOS', 1))

    try:
        payload_str = json.dumps(payload, ensure_ascii=False)
        mqtt_client.publish(topic, payload_str, qos=qos)
        return True
    except Exception:
        logging.exception('Failed to publish via flask-mqtt')
        return False


def create_paho_client(on_message_cb=None,
                       broker: str = None,
                       port: int = None,
                       username: str = None,
                       password: str = None,
                       use_tls: bool = False,
                       client_id: str = None,
                       keepalive: int = 60,
                       subscribe_topic: str = None,
                       qos: int = 1,
                       start_loop: bool = True):
    """Create and return a configured paho.mqtt.client.Client instance.

    - `on_message_cb(client, userdata, message)` will be registered if provided.
    - If `subscribe_topic` is provided, the client will subscribe on connect.
    - If `start_loop` is True, `loop_start()` is called before returning.
    """
    try:
        import paho.mqtt.client as mqtt
    except Exception:
        raise RuntimeError('paho-mqtt is required to create paho client')

    broker = broker or os.environ.get('MQTT_BROKER', 'localhost')
    port = int(port or os.environ.get('MQTT_PORT', 1883))

    # Prefer using the newer paho callback API when available to avoid deprecation warnings.
    try:
        import inspect

        sig = inspect.signature(mqtt.Client)
        if 'callback_api_version' in sig.parameters:
            client = mqtt.Client(client_id, callback_api_version=2) if client_id else mqtt.Client(callback_api_version=2)
        else:
            client = mqtt.Client(client_id) if client_id else mqtt.Client()
    except Exception:
        client = mqtt.Client(client_id) if client_id else mqtt.Client()
    if username and password:
        client.username_pw_set(username, password)
    if use_tls:
        try:
            client.tls_set()
        except Exception:
            logging.exception('Failed to configure TLS for paho client')

    def _on_connect(client_, userdata, flags, rc):
        try:
            if rc == 0:
                logging.info('paho MQTT connected (rc=0)')
            else:
                logging.warning('paho MQTT connect rc=%s', rc)
            if subscribe_topic:
                try:
                    client_.subscribe(subscribe_topic, qos=qos)
                    logging.info('paho MQTT subscribed to %s', subscribe_topic)
                except Exception:
                    logging.exception('paho subscribe failed')
        except Exception:
            logging.exception('Exception in paho on_connect')

    client.on_connect = _on_connect

    if on_message_cb is not None:
        client.on_message = on_message_cb

    try:
        client.connect(broker, port, keepalive)
    except Exception:
        logging.exception('Failed to connect paho client to broker')
        # still return client (user may start loop later or use fallback)

    if start_loop:
        try:
            client.loop_start()
        except Exception:
            logging.exception('Failed to start paho loop')

    return client


def publish_with_client(mqtt_client, payload: dict, topic: str = None, qos: int = None) -> bool:
    """Publish using an existing paho `mqtt_client` instance."""
    if mqtt_client is None:
        raise RuntimeError('mqtt_client is required')

    if not isinstance(payload, dict):
        try:
            payload = dict(payload)
        except Exception:
            payload = {'value': str(payload)}

    if 'id' not in payload:
        payload['id'] = str(uuid.uuid4())
    if 'timestamp' not in payload:
        payload['timestamp'] = datetime.now(timezone.utc).isoformat()

    topic = topic or os.environ.get('MQTT_TOPIC', '404ai/detections')
    qos = int(qos or os.environ.get('MQTT_QOS', 1))

    try:
        payload_str = json.dumps(payload, ensure_ascii=False)
        mqtt_client.publish(topic, payload_str, qos=qos)
        return True
    except Exception:
        logging.exception('Failed to publish via provided paho client')
        return False


def is_client_connected(mqtt_client) -> bool:
    """Return True if the provided paho `mqtt_client` appears connected.

    This tries several non-destructive checks in order:
    - call `is_connected()` if available
    - check for underlying socket `_sock` or `socket`
    - check internal `_state` against paho's `mqtt_cs_connected`
    - as a last resort, attempt a non-blocking `publish` and inspect the rc
    """
    if mqtt_client is None:
        return False
    try:
        is_conn = getattr(mqtt_client, 'is_connected', None)
        if callable(is_conn):
            try:
                return bool(is_conn())
            except Exception:
                pass

        # common internal socket attribute
        sock = getattr(mqtt_client, '_sock', None) or getattr(mqtt_client, 'socket', None)
        if sock:
            return True

        state = getattr(mqtt_client, '_state', None)
        if state is not None:
            try:
                import paho.mqtt.client as _mqtt
                return state == _mqtt.mqtt_cs_connected
            except Exception:
                # fallback: connected state is usually 1
                return state == 1

        # Last resort: attempt a quick publish and inspect return code.
        try:
            info = mqtt_client.publish('__health_check_404ai', 'ping', qos=0)
            rc = getattr(info, 'rc', None)
            if rc is None:
                # older paho may return tuple-like result
                try:
                    rc = int(info[0])
                except Exception:
                    rc = None
            return rc == 0 if rc is not None else True
        except Exception:
            return False
    except Exception:
        return False


if __name__ == '__main__':
    # Basic demo: start Flask-MQTT for a few seconds to check callbacks.
    logging.basicConfig(level=logging.INFO)
    print('Starting MQTT subscriber demo (will run ~5s)')
    mqtt_client = None
    try:
        try:
            from flask import Flask
        except Exception:
            raise RuntimeError('Flask is required for demo')

        app = Flask(__name__)
        # Allow env/config overrides via Flask config
        for k in ('MQTT_BROKER', 'MQTT_PORT', 'MQTT_USERNAME', 'MQTT_PASSWORD', 'MQTT_QOS', 'MQTT_SUBSCRIBE_TOPIC'):
            if k in os.environ:
                app.config[k] = os.environ[k]

        mqtt_client = init_flask_mqtt(app)
        # allow some time for connect/subscribe events (noop fallback won't block)
        time.sleep(5)
    except Exception:
        logging.exception('Demo subscriber failed')
    finally:
        try:
            if mqtt_client is not None:
                client = getattr(mqtt_client, 'client', None)
                if client is not None:
                    try:
                        client.loop_stop()
                        client.disconnect()
                    except Exception:
                        pass
        except Exception:
            pass
    print('Demo finished')
