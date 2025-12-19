# 404-ai

A Flask-based web application for deep learning inference using PatchCore model for anomaly detection.

## Features

- Flask web server with REST API
- PatchCore deep learning model for anomaly detection
- Image upload and processing
- MQTT integration for IoT connectivity
- Responsive web interface
- Environment-based configuration

## Project Structure

```
404-ai/
├── app.py                  # Main Flask application
├── patchcore.py           # PatchCore deep learning model
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── models/               # Directory for PyTorch model files (.pkl)
├── templates/            # HTML templates
│   └── index.html
├── static/               # Static files
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
└── uploads/              # Uploaded images (created at runtime)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 12.6 (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/WinLakeLee/404-ai.git
cd 404-ai
```

2. Install dependencies:

For CPU-only:
```bash
pip3 install -r requirements.txt
```

For GPU with CUDA 12.6:
```bash
pip3 install flask ultralytics paho-mqtt pillow python-dotenv
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

3. Create environment file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Place your trained PatchCore model:
   - Save your PyTorch model as a pickle file (`.pkl`)
   - Place it in the `models/` directory
   - Update `MODEL_FILE` in `.env` if needed

## Usage

### Running the Application

```bash
python3 app.py
```

The application will start on `http://0.0.0.0:5000` by default.

### API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check endpoint
- `POST /upload` - Upload image for inference
- `GET /uploads/<filename>` - Access uploaded files

### Environment Variables

See `.env.example` for all available configuration options:

- **Flask Configuration**: Host, port, debug mode
- **Upload Settings**: Folder, file size limits, allowed extensions
- **Model Settings**: Model path and parameters
- **MQTT Settings**: Broker connection and topic configuration
- **Deep Learning**: Device selection (cuda/cpu), batch size, thresholds

## Model Format

The PatchCore model should be saved as a pickle file containing:
- A PyTorch model with `forward()` or `predict()` method
- Optional configuration dictionary

Example model structure:
```python
model_data = {
    'model': trained_model,
    'config': {
        'threshold': 0.5,
        'other_params': ...
    }
}
```

## Development

### Creating a Sample Model

For testing purposes, you can create a sample model:

```python
python3 -c "from patchcore import create_sample_model, PatchCoreModel; \
model = create_sample_model(); \
wrapper = PatchCoreModel(); \
wrapper.model = model; \
wrapper.is_loaded = True; \
wrapper.save_model('models/patchcore_model.pkl')"
```

### MQTT Integration

Enable MQTT in your `.env` file:
```
MQTT_ENABLED=true
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_TOPIC=ai/detection
```

The application will publish inference results to the configured MQTT topic.

## Security Notes

- Change `SECRET_KEY` in production
- Use HTTPS in production
- Implement proper authentication for API endpoints
- Validate and sanitize all user inputs
- Keep dependencies updated

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]