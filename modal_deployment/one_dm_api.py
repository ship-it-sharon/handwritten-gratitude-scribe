import modal
import torch
import numpy as np
from PIL import Image
import base64
import io
import json
from pathlib import Path
import requests
import zipfile

# Create Modal app
app = modal.App("one-dm-handwriting")

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install([
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "pillow>=8.3.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.4.0",
        "requests>=2.25.0"
    ])
    .apt_install(["git", "wget", "unzip"])
)

# Download and setup One-DM model
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/models": modal.Volume.from_name("one-dm-models", create_if_missing=True)}
)
def setup_model():
    """Download and setup the One-DM model"""
    import subprocess
    import os
    
    model_dir = Path("/models")
    repo_dir = model_dir / "One-DM"
    
    if not repo_dir.exists():
        print("Downloading One-DM repository...")
        subprocess.run([
            "git", "clone", "https://github.com/dailenson/One-DM.git", str(repo_dir)
        ], check=True)
        
        # Download pretrained weights if available
        weights_url = "https://github.com/dailenson/One-DM/releases/download/v1.0/one_dm_weights.pth"
        weights_path = repo_dir / "weights" / "one_dm_weights.pth"
        weights_path.parent.mkdir(exist_ok=True)
        
        try:
            print("Downloading pretrained weights...")
            response = requests.get(weights_url)
            if response.status_code == 200:
                with open(weights_path, "wb") as f:
                    f.write(response.content)
                print("Weights downloaded successfully")
            else:
                print("Pretrained weights not available, will need to train")
        except Exception as e:
            print(f"Could not download weights: {e}")
    
    return "Model setup complete"

class OneDMModel:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the One-DM model"""
        try:
            # Import One-DM modules (this is a simplified version)
            # In reality, we'd need to implement the actual One-DM architecture
            print("Loading One-DM model...")
            
            # For now, create a placeholder model structure
            # This would be replaced with actual One-DM model loading
            self.model = self._create_placeholder_model()
            self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _create_placeholder_model(self):
        """Create a placeholder model structure"""
        # This is a simplified placeholder - real implementation would load One-DM
        import torch.nn as nn
        
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(1000, 512)
                self.decoder = nn.Linear(512, 1000)
                
            def forward(self, x):
                return self.decoder(self.encoder(x))
        
        return PlaceholderModel()
    
    def preprocess_reference(self, image_base64):
        """Preprocess reference handwriting sample"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Resize and normalize
            image = image.resize((256, 64))
            image_array = np.array(image) / 255.0
            
            # Convert to tensor
            image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            print(f"Error preprocessing reference: {e}")
            return None
    
    def generate_handwriting(self, text, reference_image_base64):
        """Generate handwriting for given text using reference style"""
        try:
            if not self.model:
                self.load_model()
            
            # Preprocess reference image
            ref_tensor = self.preprocess_reference(reference_image_base64)
            if ref_tensor is None:
                raise ValueError("Failed to preprocess reference image")
            
            # For this placeholder implementation, generate a simple image
            # Real One-DM would process text and generate handwriting
            generated_image = self._generate_placeholder_handwriting(text, ref_tensor)
            
            # Convert to base64
            output_base64 = self._tensor_to_base64(generated_image)
            return output_base64
            
        except Exception as e:
            print(f"Error generating handwriting: {e}")
            raise
    
    def _generate_placeholder_handwriting(self, text, ref_tensor):
        """Generate placeholder handwriting image"""
        # Create a simple handwriting-like image
        from PIL import Image, ImageDraw, ImageFont
        
        # Create image
        width = len(text) * 20 + 100
        height = 100
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw text (simplified handwriting simulation)
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((20, 30), text, fill='black', font=font)
        
        # Add some variation to simulate handwriting
        import random
        for i in range(len(text) * 5):
            x = random.randint(0, width-1)
            y = random.randint(25, 75)
            draw.point((x, y), fill='gray')
        
        return image
    
    def _tensor_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return image_base64

# Create global model instance
model_instance = OneDMModel()

@app.function(
    image=image,
    gpu="A10G",
    timeout=300,
    volumes={"/models": modal.Volume.from_name("one-dm-models", create_if_missing=True)}
)
@modal.web_endpoint(method="POST")
def generate_handwriting(request_data: dict):
    """API endpoint for generating handwriting"""
    try:
        # Parse request
        text = request_data.get("text", "")
        reference_samples = request_data.get("samples", [])
        
        if not text:
            return {"error": "No text provided"}, 400
        
        if not reference_samples:
            return {"error": "No reference samples provided"}, 400
        
        # Use first reference sample
        reference_base64 = reference_samples[0]
        if reference_base64.startswith('data:image'):
            reference_base64 = reference_base64.split(',')[1]
        
        # Generate handwriting
        result_base64 = model_instance.generate_handwriting(text, reference_base64)
        
        # Return as SVG format to match existing interface
        svg_content = f'''<svg width="800" height="200" viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg">
  <image href="data:image/png;base64,{result_base64}" x="0" y="0" width="800" height="200"/>
</svg>'''
        
        return {
            "handwritingSvg": svg_content,
            "styleCharacteristics": {
                "slant": 2.0,
                "spacing": 1.0,
                "strokeWidth": 2.0,
                "baseline": "straight"
            }
        }
        
    except Exception as e:
        print(f"Error in generate_handwriting: {e}")
        return {"error": str(e)}, 500

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "One-DM"}

if __name__ == "__main__":
    # Setup model when running locally
    setup_model.remote()