import modal

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
    
    from pathlib import Path
    import requests
    
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

# Remove class definition - move everything inside functions

@app.function(
    image=image,
    gpu="A10G",
    timeout=300,
    volumes={"/models": modal.Volume.from_name("one-dm-models", create_if_missing=True)}
)
@modal.fastapi_endpoint(method="POST")
def generate_handwriting(request_data: dict):
    """API endpoint for generating handwriting"""
    # Import all dependencies inside the function
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import base64
    import io
    import random
    
    try:
        # Parse request
        text = request_data.get("text", "")
        reference_samples = request_data.get("samples", [])
        
        if not text:
            return {"error": "No text provided"}
        
        print(f"Generating handwriting for text: '{text}' with {len(reference_samples)} samples")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Process reference image if provided
        ref_tensor = None
        if reference_samples and len(reference_samples) > 0:
            try:
                reference_base64 = reference_samples[0]
                if reference_base64.startswith('data:image'):
                    reference_base64 = reference_base64.split(',')[1]
                
                # Decode base64 image
                image_data = base64.b64decode(reference_base64)
                ref_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Resize and normalize
                ref_image = ref_image.resize((256, 64))
                image_array = np.array(ref_image) / 255.0
                
                # Convert to tensor
                ref_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
                ref_tensor = ref_tensor.to(device)
                
                print("Reference image processed successfully")
            except Exception as e:
                print(f"Error processing reference image: {e}")
        
        # Generate placeholder handwriting image
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
        for i in range(len(text) * 5):
            x = random.randint(0, width-1)
            y = random.randint(25, 75)
            draw.point((x, y), fill='gray')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
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
        return {"error": str(e)}

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "One-DM"}

if __name__ == "__main__":
    # Setup model when running locally
    setup_model.remote()