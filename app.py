"""
Flask Application for Deep Learning Model Inference
"""
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
from PIL import Image
import logging

from patchcore import PatchCoreModel

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MODEL_FOLDER'] = os.getenv('MODEL_FOLDER', 'models')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB default
app.config['ALLOWED_EXTENSIONS'] = set(os.getenv('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,gif').split(','))

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MQTT client
mqtt_client = None
mqtt_enabled = os.getenv('MQTT_ENABLED', 'false').lower() == 'true'

if mqtt_enabled:
    mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
    mqtt_port = int(os.getenv('MQTT_PORT', 1883))
    mqtt_topic = os.getenv('MQTT_TOPIC', 'ai/detection')
    
    mqtt_client = mqtt.Client()
    try:
        mqtt_client.connect(mqtt_broker, mqtt_port, 60)
        mqtt_client.loop_start()
        logger.info(f"MQTT client connected to {mqtt_broker}:{mqtt_port}")
    except Exception as e:
        logger.error(f"Failed to connect to MQTT broker: {e}")
        mqtt_client = None

# Initialize model
model = None
try:
    model_path = os.path.join(app.config['MODEL_FOLDER'], os.getenv('MODEL_FILE', 'patchcore_model.pkl'))
    if os.path.exists(model_path):
        model = PatchCoreModel(model_path)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model file not found at {model_path}")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'mqtt_connected': mqtt_client is not None
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and inference"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load image
            image = Image.open(filepath)
            
            # Run inference if model is loaded
            if model:
                result = model.predict(image)
                
                # Publish to MQTT if enabled
                if mqtt_client and mqtt_enabled:
                    mqtt_client.publish(mqtt_topic, str(result))
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'result': result
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Model not loaded'
                }), 503
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    app.run(debug=debug_mode, host=host, port=port)
