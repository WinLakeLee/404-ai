"""
PatchCore Deep Learning Model for Anomaly Detection
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PatchCoreModel:
    """
    PatchCore model for anomaly detection using deep learning.
    This class handles loading pre-trained models and making predictions on images.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the PatchCore model.
        
        Args:
            model_path (str): Path to the saved model pickle file
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.is_loaded = False
        
        # Setup image preprocessing transforms
        self.setup_transforms()
        
        # Load model if path is provided
        if model_path:
            self.load_model(model_path)
    
    def setup_transforms(self):
        """Setup image preprocessing transformations"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_model(self, model_path):
        """
        Load a pre-trained model from a pickle file.
        
        Args:
            model_path (str): Path to the model pickle file
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different model formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model', None)
                self.config = model_data.get('config', {})
            else:
                self.model = model_data
                self.config = {}
            
            if self.model is not None:
                if hasattr(self.model, 'to'):
                    self.model = self.model.to(self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                self.is_loaded = True
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.error(f"Invalid model format in {model_path}")
                
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess an image for model input.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image or path to an image file")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image):
        """
        Make a prediction on an input image.
        
        Args:
            image (PIL.Image or str): Input image or path to image
            
        Returns:
            dict: Prediction results containing anomaly score and classification
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please load a model first.")
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Make prediction
            with torch.no_grad():
                if hasattr(self.model, 'predict'):
                    # Custom predict method
                    output = self.model.predict(image_tensor)
                elif callable(self.model):
                    # Model is callable
                    output = self.model(image_tensor)
                else:
                    raise ValueError("Model does not have a predict method or is not callable")
            
            # Process output
            result = self.process_output(output)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def process_output(self, output):
        """
        Process model output into a human-readable format.
        
        Args:
            output: Model output (can be tensor, dict, or other format)
            
        Returns:
            dict: Processed results
        """
        result = {
            'status': 'success',
            'anomaly_detected': False,
            'confidence': 0.0,
            'anomaly_score': 0.0
        }
        
        try:
            # Handle different output formats
            if isinstance(output, torch.Tensor):
                if output.dim() > 1:
                    output = output.squeeze()
                
                # Convert to numpy
                output_np = output.cpu().numpy()
                
                # Calculate anomaly score (assuming higher values indicate anomalies)
                if output_np.size == 1:
                    anomaly_score = float(output_np)
                else:
                    anomaly_score = float(np.max(output_np))
                
                result['anomaly_score'] = anomaly_score
                result['confidence'] = min(abs(anomaly_score), 1.0)
                
                # Threshold for anomaly detection (can be configured)
                threshold = self.config.get('threshold', 0.5)
                result['anomaly_detected'] = anomaly_score > threshold
                
            elif isinstance(output, dict):
                # If output is already a dictionary, use it directly
                result.update(output)
            else:
                result['raw_output'] = str(output)
                
        except Exception as e:
            logger.error(f"Error processing output: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def save_model(self, save_path):
        """
        Save the current model to a pickle file.
        
        Args:
            save_path (str): Path where to save the model
        """
        if not self.is_loaded:
            raise RuntimeError("No model to save")
        
        try:
            model_data = {
                'model': self.model,
                'config': self.config
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved successfully to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise


def create_sample_model():
    """
    Create a sample PatchCore model for testing purposes.
    This is a placeholder that should be replaced with actual model training.
    """
    class SimplePatchCoreNet(nn.Module):
        def __init__(self):
            super(SimplePatchCoreNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(128, 1)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return torch.sigmoid(x)
    
    model = SimplePatchCoreNet()
    return model


if __name__ == '__main__':
    # Example usage
    print("PatchCore Model Module")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create and save a sample model
    sample_model = create_sample_model()
    model_wrapper = PatchCoreModel()
    model_wrapper.model = sample_model
    model_wrapper.is_loaded = True
    
    print("\nSample model created successfully")
    print(f"Device: {model_wrapper.device}")
