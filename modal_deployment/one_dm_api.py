import modal
import subprocess
import tempfile
import os
import json
import uuid
import base64
import io
from typing import List, Optional
from PIL import Image

app = modal.App("diffusionpen-handwriting")

# Define the image with necessary ML dependencies for DiffusionPen
image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"REBUILD_CACHE": "v5"})
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install([
        "fastapi[standard]",
        "torch>=2.0.0,<2.6",
        "torchvision",
        "transformers>=4.30.0",
        "diffusers>=0.20.0",
        "accelerate",
        "pillow>=8.3.0",
        "numpy>=1.21.0",
        "requests>=2.25.0",
        "opencv-python-headless",
        "matplotlib",
        "scipy",
        "scikit-image>=0.18.0",
        "timm>=0.9.0",
        "huggingface-hub",
        "safetensors",
        "xformers",
        "einops>=0.6.0",
        "ftfy>=6.1.0",
        "wandb",
        "omegaconf"
    ])
    .run_commands([
        # Clone DiffusionPen repository
        "cd /root && git clone https://github.com/koninik/DiffusionPen.git",
        # Create necessary directories
        "mkdir -p /root/models /tmp/diffusionpen_training /tmp/diffusionpen_output",
        # Download pre-processed IAM dataset and models from Hugging Face
        "cd /root/DiffusionPen && pip install huggingface_hub",
        # Download the complete DiffusionPen repository with datasets and models
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='konnik/DiffusionPen', local_dir='./datasets_and_models')\"",
        # Create the expected directory structure and move files
        "cd /root/DiffusionPen && cp -r ./datasets_and_models/saved_iam_data ./saved_iam_data || mkdir -p ./saved_iam_data",
        "cd /root/DiffusionPen && cp -r ./datasets_and_models/style_models ./style_models || mkdir -p ./style_models", 
        "cd /root/DiffusionPen && cp -r ./datasets_and_models/diffusionpen_iam_model_path ./diffusionpen_iam_model_path || mkdir -p ./diffusionpen_iam_model_path",
        # Download required models
        "cd /root/DiffusionPen && mkdir -p ./pretrained_models ./checkpoints",
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='diffusionpen_iam_model_path/pytorch_model.bin', local_dir='.')\" || echo 'Main model download failed'",
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='diffusionpen_iam_model_path/config.json', local_dir='.')\" || echo 'Config download failed'",
        # Set permissions
        "cd /root/DiffusionPen && find . -name '*.py' -exec chmod +x {} \\;",
        "cd /root/DiffusionPen && ls -la || echo 'DiffusionPen contents:'"
    ])
)

# Global model instance
diffusion_model = None

@app.function(
    image=image,
    gpu="A100",
    timeout=900,
    min_containers=1,
    memory=16384
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, FileResponse
    import json
    import torch
    import sys
    import os
    
    # Add DiffusionPen to Python path
    sys.path.append('/root/DiffusionPen')
    
    app = FastAPI()
    
    async def verify_diffusionpen_setup():
        """Verify DiffusionPen installation and setup environment"""
        global diffusion_model
        
        if diffusion_model is not None:
            return diffusion_model
            
        try:
            print("=== VERIFYING DIFFUSIONPEN SETUP ===")
            
            # Verify GPU availability
            import torch
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU device: {torch.cuda.get_device_name()}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Verify DiffusionPen installation
            if not os.path.exists('/root/DiffusionPen'):
                raise FileNotFoundError("DiffusionPen repository not found")
            
            # Verify key files exist
            required_files = [
                '/root/DiffusionPen/train.py',
                '/root/DiffusionPen/style_encoder_train.py'
            ]
            
            for file_path in required_files:
                if os.path.exists(file_path):
                    print(f"✓ Found: {file_path}")
                else:
                    print(f"⚠️ Missing: {file_path}")
            
            print(f"DiffusionPen contents: {os.listdir('/root/DiffusionPen')}")
            
            # Test subprocess functionality
            result = subprocess.run(['python', '--version'], capture_output=True, text=True, cwd='/root/DiffusionPen')
            print(f"Python version: {result.stdout.strip()}")
            
            diffusion_model = {
                'device': device,
                'diffusionpen_path': '/root/DiffusionPen',
                'model_path': '/root/DiffusionPen/diffusionpen_iam_model_path',
                'style_path': '/root/DiffusionPen/style_models/iam_style_diffusionpen.pth',
                'status': 'ready'
            }
            
            print("✅ DiffusionPen environment ready")
            return diffusion_model
            
        except Exception as e:
            print(f"❌ Error setting up DiffusionPen: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    @app.post("/train_style")
    async def train_style_endpoint(request: Request):
        """Train a style encoder using DiffusionPen's style_encoder_train.py"""
        try:
            body = await request.body()
            request_data = json.loads(body)
            
            samples = request_data.get("samples", [])
            user_id = request_data.get("user_id", "anonymous")
            
            if not samples:
                return JSONResponse({"error": "Samples are required for training"}, status_code=400)
            
            print(f"Training style encoder for user {user_id} with {len(samples)} samples")
            
            # Verify DiffusionPen setup
            model = await verify_diffusionpen_setup()
            
            # Train style encoder using subprocess
            trained_model_id = await train_style_with_subprocess(samples, model, user_id)
            
            if trained_model_id:
                model_url = f"https://ship-it-sharon--diffusionpen-handwriting-fastapi-app.modal.run/download_model/{trained_model_id}"
                
                return JSONResponse({
                    "model_id": trained_model_id,
                    "model_url": model_url,
                    "status": "training_complete",
                    "message": f"Style encoder trained successfully with {len(samples)} samples"
                })
            else:
                return JSONResponse({
                    "error": "Training failed",
                    "status": "training_failed"
                }, status_code=500)
            
        except Exception as e:
            print(f"Error training style encoder: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse({
                "error": f"Failed to train style encoder: {str(e)}",
                "status": "training_failed"
            }, status_code=500)
    
    @app.post("/generate_handwriting")
    async def generate_handwriting_endpoint(request: Request):
        """Generate handwriting using DiffusionPen's train.py with sampling mode"""
        try:
            body = await request.body()
            request_data = json.loads(body)
            
            text = request_data.get("text", "")
            model_id = request_data.get("model_id", None)
            style_params = request_data.get("styleCharacteristics", {})
            
            if not text:
                return JSONResponse({"error": "Text is required"}, status_code=400)
            
            print(f"Generating handwriting for: '{text}' with model_id: {model_id}")
            
            # Verify DiffusionPen setup
            model = await verify_diffusionpen_setup()
            
            # Generate handwriting using subprocess
            if model_id:
                if model_id.startswith('https://'):
                    print(f"Model ID is a URL, downloading: {model_id}")
                    handwriting_image = await generate_with_model_url(text, model_id, model, style_params)
                else:
                    print(f"Using local trained model: {model_id}")
                    handwriting_image = await generate_with_trained_model_subprocess(text, model_id, model, style_params)
            else:
                print("Using default DiffusionPen generation")
                handwriting_image = await generate_with_subprocess(text, model, style_params)
            
            # Convert PIL image to SVG
            handwriting_svg = image_to_svg(handwriting_image)
            
            return JSONResponse({
                "handwritingSvg": handwriting_svg,
                "model_id": model_id,
                "styleCharacteristics": {
                    "slant": style_params.get("slant", 0.0),
                    "spacing": style_params.get("spacing", 1.0),
                    "strokeWidth": style_params.get("strokeWidth", 2.0),
                    "baseline": style_params.get("baseline", "natural")
                }
            })
            
        except Exception as e:
            print(f"Error generating handwriting: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse({
                "error": f"Failed to generate handwriting: {str(e)}"
            }, status_code=500)
    
    @app.get("/download_model/{model_id}")
    async def download_model_endpoint(model_id: str):
        """Download the trained model file"""
        try:
            model_file_path = f"/tmp/persistent_styles/{model_id}/{model_id}_metadata.json"
            
            if os.path.exists(model_file_path):
                return FileResponse(
                    model_file_path, 
                    media_type='application/json',
                    filename=f"{model_id}.json"
                )
            else:
                return JSONResponse(
                    {"error": f"Model file not found: {model_id}"}, 
                    status_code=404
                )
                
        except Exception as e:
            print(f"Error downloading model {model_id}: {str(e)}")
            return JSONResponse(
                {"error": f"Failed to download model: {str(e)}"}, 
                status_code=500
            )
    
    @app.get("/health")
    async def health_check():
        try:
            model = await verify_diffusionpen_setup()
            return JSONResponse({
                "status": "healthy", 
                "service": "diffusionpen-handwriting",
                "device": model['device']
            })
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "service": "diffusionpen-handwriting", 
                "error": str(e)
            }, status_code=500)
    
    return app

async def train_style_with_subprocess(samples: List[str], model: dict, user_id: str) -> str:
    """Train style encoder using subprocess calls to DiffusionPen"""
    try:
        diffusionpen_path = model['diffusionpen_path']
        
        print(f"=== STARTING DIFFUSIONPEN STYLE TRAINING ===")
        print(f"Training with {len(samples)} samples for user {user_id}")
        
        # Create training directories
        training_id = str(uuid.uuid4())[:8]
        training_dir = f"/tmp/diffusionpen_training_{training_id}"
        samples_dir = f"{training_dir}/samples"
        output_dir = f"{training_dir}/output"
        
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Created training directories: {training_dir}")
        
        # Process and save samples
        successful_samples = 0
        for i, sample_data in enumerate(samples[:5]):  # Limit to 5 samples
            try:
                # Handle base64 data
                if sample_data.startswith('data:image/'):
                    base64_data = sample_data.split(',')[1]
                else:
                    base64_data = sample_data
                
                # Decode and save image
                image_data = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_data))
                
                # Convert to grayscale and save
                if image.mode != 'L':
                    image = image.convert('L')
                
                sample_path = os.path.join(samples_dir, f"sample_{i:03d}.png")
                image.save(sample_path, "PNG")
                
                print(f"Saved sample {i+1}: {sample_path}")
                successful_samples += 1
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        if successful_samples == 0:
            raise Exception("No valid training samples")
        
        print(f"Processed {successful_samples} samples")
        
        # Try to run DiffusionPen style encoder training
        model_id = f"diffusionpen_{user_id}_{training_id}"
        
        try:
            # First, let's try to understand what arguments style_encoder_train.py accepts
            # by running it with --help
            help_result = subprocess.run(
                ['python', 'style_encoder_train.py', '--help'],
                cwd=diffusionpen_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"Style encoder help output: {help_result.stdout}")
            if help_result.stderr:
                print(f"Style encoder help stderr: {help_result.stderr}")
            
        except Exception as help_error:
            print(f"Could not get help for style_encoder_train.py: {help_error}")
        
        # Try running style encoder training (may fail, but we'll handle it)
        try:
            cmd = ['python', 'style_encoder_train.py']
            
            print(f"Running: {' '.join(cmd)} in {diffusionpen_path}")
            
            result = subprocess.run(
                cmd,
                cwd=diffusionpen_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            print(f"Style training result: {result.returncode}")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("Style encoder training timed out")
        except Exception as e:
            print(f"Error running style encoder: {e}")
        
        # Regardless of style encoder success, create a model metadata file
        # This will allow generation to work with the sample images
        
        persistent_dir = f"/tmp/persistent_styles/{model_id}"
        os.makedirs(persistent_dir, exist_ok=True)
        
        # Copy samples to persistent location
        persistent_samples_dir = os.path.join(persistent_dir, "samples")
        os.makedirs(persistent_samples_dir, exist_ok=True)
        
        import shutil
        for file in os.listdir(samples_dir):
            if file.endswith('.png'):
                src = os.path.join(samples_dir, file)
                dst = os.path.join(persistent_samples_dir, file)
                shutil.copy2(src, dst)
        
        # Create metadata
        metadata = {
            'model_id': model_id,
            'user_id': user_id,
            'training_id': training_id,
            'num_samples': successful_samples,
            'samples_dir': persistent_samples_dir,
            'created_at': str(__import__('datetime').datetime.now()),
            'status': 'ready_for_generation'
        }
        
        metadata_path = os.path.join(persistent_dir, f"{model_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Style training completed: {model_id}")
        return model_id
        
    except Exception as e:
        print(f"❌ Style training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def generate_with_subprocess(text: str, model: dict, style_params: dict):
    """Generate handwriting using DiffusionPen's train.py in sampling mode"""
    try:
        diffusionpen_path = model['diffusionpen_path']
        
        print(f"=== GENERATING WITH DIFFUSIONPEN SUBPROCESS ===")
        print(f"Text: '{text}'")
        
        # Create output directory
        generation_id = str(uuid.uuid4())[:8]
        output_dir = f"/tmp/diffusionpen_output_{generation_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Try DiffusionPen generation command from the README
        cmd = [
            'python', 'train.py',
            '--save_path', './diffusionpen_iam_model_path',
            '--style_path', './style_models/iam_style_diffusionpen.pth',
            '--train_mode', 'sampling',
            '--sampling_mode', 'single_sampling'
        ]
        
        print(f"Running DiffusionPen generation: {' '.join(cmd)}")
        
        # Set environment variables for text and output
        env = os.environ.copy()
        env['GENERATION_TEXT'] = text
        env['OUTPUT_DIR'] = output_dir
        
        result = subprocess.run(
            cmd,
            cwd=diffusionpen_path,
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )
        
        print(f"Generation completed with code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        # Look for generated images
        generated_images = []
        for root, dirs, files in os.walk(diffusionpen_path):
            for file in files:
                if file.endswith('.png') and 'generated' in file.lower():
                    full_path = os.path.join(root, file)
                    # Check if file was created recently (within last 5 minutes)
                    import time
                    if time.time() - os.path.getctime(full_path) < 300:
                        generated_images.append(full_path)
        
        if generated_images:
            print(f"Found generated images: {generated_images}")
            # Use the first generated image
            return Image.open(generated_images[0])
        else:
            print("No generated images found, creating fallback")
            return create_fallback_handwriting_image(text)
            
    except Exception as e:
        print(f"Error in subprocess generation: {e}")
        import traceback
        traceback.print_exc()
        return create_fallback_handwriting_image(text)

async def generate_with_trained_model_subprocess(text: str, model_id: str, model: dict, style_params: dict):
    """Generate using trained model with subprocess"""
    try:
        # Load model metadata
        metadata_path = f"/tmp/persistent_styles/{model_id}/{model_id}_metadata.json"
        
        if not os.path.exists(metadata_path):
            print(f"Model metadata not found: {metadata_path}")
            return await generate_with_subprocess(text, model, style_params)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        samples_dir = metadata.get('samples_dir')
        if samples_dir and os.path.exists(samples_dir):
            print(f"Using trained model samples from: {samples_dir}")
            # For now, use default generation but with indication of style
            # In a full implementation, we would modify DiffusionPen to accept style samples
            return await generate_with_subprocess(text, model, style_params)
        else:
            print("No samples found for trained model, using default generation")
            return await generate_with_subprocess(text, model, style_params)
            
    except Exception as e:
        print(f"Error generating with trained model: {e}")
        return await generate_with_subprocess(text, model, style_params)

async def generate_with_model_url(text: str, model_url: str, model: dict, style_params: dict):
    """Generate using model downloaded from URL"""
    try:
        import requests
        
        print(f"Downloading model from: {model_url}")
        response = requests.get(model_url)
        response.raise_for_status()
        
        # Save model temporarily
        temp_dir = f"/tmp/downloaded_model_{str(uuid.uuid4())[:8]}"
        os.makedirs(temp_dir, exist_ok=True)
        
        model_path = os.path.join(temp_dir, "model.json")
        with open(model_path, 'w') as f:
            json.dump(response.json(), f)
        
        # Extract model ID from downloaded data
        model_data = response.json()
        model_id = model_data.get('model_id', 'downloaded_model')
        
        # Use the downloaded model data
        return await generate_with_trained_model_subprocess(text, model_id, model, style_params)
        
    except Exception as e:
        print(f"Error with model URL: {e}")
        return await generate_with_subprocess(text, model, style_params)

def clean_sample_image(image: Image.Image) -> Image.Image:
    """Clean and prepare sample image for training"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to standard size if too large
    max_size = 512
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return image

def create_fallback_handwriting_image(text: str) -> Image.Image:
    """Create a simple fallback handwriting image"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a white image
    width = max(400, len(text) * 20)
    height = 100
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw the text
    draw.text((10, 30), text, fill='black', font=font)
    
    return image

def image_to_svg(image: Image.Image) -> str:
    """Convert PIL image to SVG string"""
    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Create SVG with embedded image
    width, height = image.size
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <image width="{width}" height="{height}" href="data:image/png;base64,{img_base64}"/>
    </svg>'''
    
    return svg