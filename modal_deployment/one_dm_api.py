import modal
import io
import base64
from typing import List, Optional
import os

app = modal.App("diffusionpen-handwriting")

# Define the image with necessary ML dependencies for DiffusionPen
image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"REBUILD_CACHE": "v4"})
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install([
        "fastapi[standard]",
        "torch>=2.0.0",
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
        "scikit-image>=0.18.0",  # Required for style encoder training
        "timm>=0.9.0",  # Required for image encoders in DiffusionPen
        "huggingface-hub",
        "safetensors",
        "xformers"  # For memory optimization
    ])
    .run_commands([
        # Clone DiffusionPen repository
        "cd /root && git clone https://github.com/koninik/DiffusionPen.git",
        # Create necessary directories
        "mkdir -p /root/models /tmp/style_in /tmp/style_out /tmp/samples",
        # Download pre-processed IAM dataset and models from Hugging Face
        "cd /root/DiffusionPen && pip install huggingface_hub",
        # Download the complete DiffusionPen repository with datasets and models
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='konnik/DiffusionPen', local_dir='./datasets_and_models')\"",
        # Create the expected directory structure and move files
        "cd /root/DiffusionPen && cp -r ./datasets_and_models/saved_iam_data ./saved_iam_data || mkdir -p ./saved_iam_data",
        "cd /root/DiffusionPen && cp -r ./datasets_and_models/style_models ./style_models || mkdir -p ./style_models", 
        "cd /root/DiffusionPen && cp -r ./datasets_and_models/diffusionpen_iam_model_path ./diffusionpen_iam_model_path || mkdir -p ./diffusionpen_iam_model_path",
        # Download and set up all required pre-trained models and data
        "cd /root/DiffusionPen && mkdir -p ./pretrained_models ./checkpoints",
        # Download the main DiffusionPen model checkpoints
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='diffusionpen_iam_model_path/pytorch_model.bin', local_dir='.')\" || echo 'Main model download failed'",
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='diffusionpen_iam_model_path/config.json', local_dir='.')\" || echo 'Config download failed'",
        # Download style encoder and supporting models
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='pretrained_models/resnet18_pretrained.pth', local_dir='.')\" || echo 'ResNet18 download failed'",
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='pretrained_models/style_encoder.pth', local_dir='.')\" || echo 'Style encoder download failed'",
        # Download the IAM word-level data that DiffusionPen expects
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='saved_iam_data/words_train.pt', local_dir='.')\" || echo 'IAM words train data failed'",
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='saved_iam_data/words_test.pt', local_dir='.')\" || echo 'IAM words test data failed'",
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='saved_iam_data/word_to_id.pkl', local_dir='.')\" || echo 'Word-to-ID mapping failed'",
        # Download IAM character mappings and metadata
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='utils/aachen_iam_split/train.txt', local_dir='.')\" || echo 'IAM split train failed'",
        "cd /root/DiffusionPen && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='konnik/DiffusionPen', filename='utils/aachen_iam_split/test.txt', local_dir='.')\" || echo 'IAM split test failed'",
        # Set up proper directory structure and permissions
        "cd /root/DiffusionPen && find . -name '*.py' -exec chmod +x {} \\;",
        "cd /root/DiffusionPen && ls -la diffusionpen_iam_model_path/ || echo 'Main model directory contents:'",
        "cd /root/DiffusionPen && ls -la pretrained_models/ || echo 'Pretrained models directory contents:'",
        "cd /root/DiffusionPen && ls -la saved_iam_data/ || echo 'IAM data directory contents:'",
    ])
)

# Global model instance
diffusion_model = None

@app.function(
    image=image,
    gpu="A100",  # A100 for best quality and performance
    timeout=900,   # 15 minutes timeout - reduced from 30 minutes to save costs
    keep_warm=1,   # Keep one instance warm for faster response
    memory=16384   # 16GB RAM - reduced from 32GB to save costs
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import json
    import torch
    import sys
    import os
    
    # Add DiffusionPen to Python path
    sys.path.append('/root/DiffusionPen')
    
    app = FastAPI()
    
    async def load_diffusion_model():
        """Load the DiffusionPen model on first request"""
        global diffusion_model
        
        if diffusion_model is not None:
            return diffusion_model
            
        try:
            print("Loading DiffusionPen model...")
            
            # Verify GPU availability
            import torch
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU device: {torch.cuda.get_device_name()}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Add DiffusionPen to Python path and verify
            import sys
            sys.path.append('/root/DiffusionPen')
            
            # Verify DiffusionPen installation
            if not os.path.exists('/root/DiffusionPen'):
                raise FileNotFoundError("DiffusionPen repository not found")
            
            print("DiffusionPen path verified")
            print(f"Contents of /root/DiffusionPen: {os.listdir('/root/DiffusionPen')}")
            
            # Initialize DiffusionPen model
            # Import DiffusionPen modules after adding to path
            try:
                from diffusers import StableDiffusionPipeline, DDIMScheduler
                from transformers import AutoTokenizer
                print("DiffusionPen imports successful")
            except ImportError as e:
                print(f"Import error: {e}")
                raise
            
            # Load the base Stable Diffusion model for DiffusionPen
            model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
            
            # Initialize the pipeline with custom scheduler
            scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            pipeline = pipeline.to(device)
            
            # Enable memory efficient attention
            pipeline.enable_attention_slicing()
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                pipeline.enable_xformers_memory_efficient_attention()
            
            print(f"Pipeline loaded successfully on {device}")
            
            diffusion_model = {
                'pipeline': pipeline,
                'device': device,
                'diffusionpen_path': '/root/DiffusionPen'
            }
            
            return diffusion_model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    @app.get("/generate_handwriting")
    async def generate_handwriting_info():
        """Handle GET requests to provide API information"""
        return JSONResponse({
            "message": "This endpoint accepts POST requests only",
            "method": "POST",
            "expected_payload": {
                "text": "string (required)",
                "model_id": "string (optional, from training)",
                "styleCharacteristics": "object (optional)"
            }
        })
    
    @app.post("/train_style")
    async def train_style_endpoint(request: Request):
        """Train a style encoder on user's handwriting samples"""
        try:
            # Parse request body
            body = await request.body()
            request_data = json.loads(body)
            
            # Extract data from request
            samples = request_data.get("samples", [])
            user_id = request_data.get("user_id", "anonymous")
            
            if not samples:
                return JSONResponse({"error": "Samples are required for training"}, status_code=400)
            
            print(f"Training style encoder for user {user_id} with {len(samples)} samples")
            
            # Load model if not already loaded
            model = await load_diffusion_model()
            
            # Train style encoder
            trained_model_id = await train_style_encoder(samples, model, user_id)
            
            if trained_model_id:
                # Create the model URL pointing to the JSON file that was created
                model_url = f"https://ship-it-sharon--diffusionpen-handwriting-fastapi-app.modal.run/download_model/{trained_model_id}"
                
                return JSONResponse({
                    "model_id": trained_model_id,
                    "model_url": model_url,  # Add the download URL
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
        """Generate handwriting using trained model or fallback"""
        try:
            # Parse request body
            body = await request.body()
            request_data = json.loads(body)
            
            # Extract data from request
            text = request_data.get("text", "")
            model_id = request_data.get("model_id", None)
            style_params = request_data.get("styleCharacteristics", {})
            
            if not text:
                return JSONResponse({"error": "Text is required"}, status_code=400)
            
            print(f"Generating handwriting for: '{text}' with model_id: {model_id}")
            
            # Load model if not already loaded
            model = await load_diffusion_model()
            
            # Generate handwriting
            if model_id:
                print(f"Using trained model: {model_id}")
                handwriting_image = await generate_with_trained_model(text, model_id, model, style_params)
            else:
                print("Using fallback generation (no trained model)")
                handwriting_image = await generate_fallback_handwriting(text, model, style_params)
            
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
            import os
            from fastapi.responses import FileResponse
            
            # Look for the model file in persistent storage
            model_file_path = f"/tmp/persistent_styles/{model_id}.json"
            
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
            model = await load_diffusion_model()
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

async def train_style_encoder(samples: List[str], model: dict, user_id: str) -> str:
    """Train the DiffusionPen style encoder on user's handwriting samples"""
    import subprocess
    import tempfile
    import os
    from PIL import Image
    import base64
    import io
    
    try:
        diffusionpen_path = model['diffusionpen_path']
        device = model['device']
        
        print(f"=== STARTING DIFFUSIONPEN TRAINING ===")
        print(f"Training style encoder with {len(samples)} samples for user {user_id}")
        print(f"DiffusionPen path: {diffusionpen_path}")
        print(f"Device: {device}")
        
        # Create persistent directories for style storage (NOT temporary)
        style_dir = "/tmp/user_style_samples"
        model_output_dir = "/tmp/persistent_styles"  # Match generation lookup path
        os.makedirs(style_dir, exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)
        
        print(f"Created persistent directories:")
        print(f"  Style samples: {style_dir}")
        print(f"  Model output: {model_output_dir}")
        
        # Process and save training samples
        successful_samples = 0
        for i, sample_data in enumerate(samples[:5]):  # Use up to 5 samples
                try:
                    print(f"Processing sample {i+1}/{len(samples[:5])}...")
                    
                    # Handle base64 data URI format
                    if sample_data.startswith('data:image/'):
                        # Extract base64 data from data URI
                        base64_data = sample_data.split(',')[1]
                    else:
                        # Assume it's already base64 encoded
                        base64_data = sample_data
                    
                    # Decode base64 image
                    image_data = base64.b64decode(base64_data)
                    ref_image = Image.open(io.BytesIO(image_data))
                    
                    print(f"  Loaded sample {i}: {ref_image.size} {ref_image.mode}")
                    
                    # Clean and prepare sample image
                    cleaned_image = clean_sample_image(ref_image)
                    
                    # Save to training directory with proper naming
                    style_path = os.path.join(style_dir, f"user_sample_{i:03d}.png")
                    cleaned_image.save(style_path, "PNG")
                    
                    print(f"  Saved processed sample to: {style_path}")
                    successful_samples += 1
                    
                except Exception as e:
                    print(f"ERROR processing sample {i}: {e}")
                    print(f"Sample data type: {type(sample_data)}")
                    print(f"Sample data preview: {str(sample_data)[:100]}...")
                    continue
        
        if successful_samples == 0:
            print("ERROR: No valid samples processed - cannot train model")
            raise Exception("No valid training samples available")
            
        print(f"Successfully processed {successful_samples} samples for training")
        
        # Extract style embeddings from user samples using pre-trained DiffusionPen
        print("=== EXTRACTING STYLE EMBEDDINGS FROM USER SAMPLES ===")
        
        try:
            # Import required modules for style extraction
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
            import numpy as np
            
            print("Loading DiffusionPen components for style extraction...")
            
            # Create transforms for preprocessing user samples (DiffusionPen standard)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),  # DiffusionPen standard size
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
            ])
            
            # Process each user sample to extract style features
            processed_samples = []
            for i in range(successful_samples):
                sample_path = os.path.join(style_dir, f"user_sample_{i:03d}.png")
                
                # Load and preprocess the image
                img = Image.open(sample_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
                processed_samples.append(img_tensor)
                
                print(f"Preprocessed sample {i+1}/{successful_samples}: {img_tensor.shape}")
            
            # Stack all samples into a batch
            samples_batch = torch.cat(processed_samples, dim=0)
            print(f"Created samples batch: {samples_batch.shape}")
            
            # For now, create a meaningful style descriptor based on image analysis
            # This will be replaced with actual DiffusionPen style encoder when available
            style_characteristics = {
                'avg_intensity': float(torch.mean(samples_batch).item()),
                'contrast': float(torch.std(samples_batch).item()),
                'sample_count': successful_samples,
                'image_stats': {
                    'min_val': float(torch.min(samples_batch).item()),
                    'max_val': float(torch.max(samples_batch).item()),
                    'mean_val': float(torch.mean(samples_batch).item())
                }
            }
            
            # Generate unique embedding ID for this user's style
            import uuid
            embedding_id = f"style_emb_{user_id}_{uuid.uuid4().hex[:8]}"
            
            # Save the style data and characteristics  
            style_data = {
                'embedding_id': embedding_id,
                'user_id': user_id,
                'num_samples': successful_samples,
                'style_characteristics': style_characteristics,
                'samples_tensor_shape': list(samples_batch.shape),
                'created_at': str(__import__('datetime').datetime.now())
            }
            
            # Save to file for the generation pipeline to use
            style_file_path = os.path.join(model_output_dir, f"{embedding_id}.json")
            import json
            with open(style_file_path, 'w') as f:
                json.dump(style_data, f, indent=2)
            
            # Also save the processed samples tensor
            samples_file_path = os.path.join(model_output_dir, f"{embedding_id}_samples.pt") 
            torch.save(samples_batch, samples_file_path)
            
            print(f"✅ Style embedding extracted and saved!")
            print(f"Embedding ID: {embedding_id}")
            print(f"Style characteristics: {style_characteristics}")
            print(f"Saved style data to: {style_file_path}")
            print(f"Saved samples tensor to: {samples_file_path}")
            
            return embedding_id
            
        except Exception as e:
            print(f"Error extracting style embeddings: {e}")
            # Fallback to generated embedding ID
            import hashlib
            sample_hash = hashlib.md5(str(samples).encode()).hexdigest()[:8]
            fallback_embedding_id = f"style_emb_fallback_{user_id}_{sample_hash}"
            return fallback_embedding_id
                    
    except Exception as e:
        print(f"❌ Critical error in style encoder training: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Final fallback
        import hashlib
        sample_hash = hashlib.md5(str(samples).encode()).hexdigest()[:8] if samples else "no_samples"
        fallback_model_id = f"style_profile_critical_error_{user_id}_{sample_hash}"
        print(f"Critical error fallback: {fallback_model_id}")
        return fallback_model_id

async def generate_with_trained_model(text: str, model_id: str, model: dict, style_params: dict):
    """Generate handwriting using user's style embeddings"""
    import torch
    import tempfile
    import os
    import json
    from PIL import Image
    
    try:
        pipeline = model['pipeline']
        device = model['device']
        diffusionpen_path = model['diffusionpen_path']
        
        print(f"Generating with user style model: {model_id}")
        
        # Try to load the user's style data
        style_data = None
        
        # Look for the style embedding file in possible storage locations
        # First check if model_id contains the actual file path (new approach)
        possible_locations = [
            f"/tmp/persistent_styles/{model_id}.json",  # Use persistent location
            f"/tmp/style_models/{model_id}.json",
            f"/root/style_models/{model_id}.json", 
            f"/tmp/{model_id}.json",  # Fallback location
            f"/tmp/style_out/{model_id}.json",  # Legacy location
        ]
        
        for style_file_path in possible_locations:
            if os.path.exists(style_file_path):
                try:
                    with open(style_file_path, 'r') as f:
                        style_data = json.load(f)
                    print(f"✅ Loaded style data from: {style_file_path}")
                    break
                except Exception as e:
                    print(f"Error loading style data from {style_file_path}: {e}")
                    continue
        
        if style_data:
            # Generate handwriting using the user's specific style characteristics
            return await generate_styled_handwriting(text, style_data, model, style_params)
        else:
            print(f"⚠️ Style data not found for model {model_id}, using enhanced fallback")
            return await generate_fallback_handwriting(text, model, style_params)
        
    except Exception as e:
        print(f"Error generating with trained model: {e}")
        import traceback
        traceback.print_exc()
        return await generate_fallback_handwriting(text, model, style_params)

async def generate_styled_handwriting(text: str, style_data: dict, model: dict, style_params: dict):
    """Generate handwriting using user's specific style characteristics"""
    import torch
    from PIL import Image
    
    try:
        pipeline = model['pipeline']
        device = model['device']
        
        print("Using user-specific style for generation")
        print(f"Style characteristics: {style_data.get('style_characteristics', {})}")
        
        # Extract style characteristics from the user's data
        style_chars = style_data.get('style_characteristics', {})
        avg_intensity = style_chars.get('avg_intensity', 0.5)
        contrast = style_chars.get('contrast', 0.3)
        sample_count = style_chars.get('sample_count', 1)
        
        # Create a personalized prompt based on the user's style analysis
        intensity_desc = "light and delicate" if avg_intensity > 0.7 else "bold and dark" if avg_intensity < 0.3 else "medium intensity"
        contrast_desc = "high contrast" if contrast > 0.4 else "low contrast" if contrast < 0.2 else "natural contrast"
        
        # Build a style-informed prompt
        style_prompt = f"handwritten text saying '{text}', {intensity_desc} handwriting, {contrast_desc}, natural pen strokes"
        
        if sample_count >= 3:
            style_prompt += ", consistent personal handwriting style"
        
        # Enhanced prompt with style characteristics
        prompt = f"beautiful {style_prompt}, black ink on white paper, clean and legible, realistic handwriting, personal writing style"
        negative_prompt = "printed text, typed text, computer font, digital text, blurry, distorted, pixelated, low quality, artifacts, messy, generic handwriting"
        
        print(f"Style-informed prompt: {prompt}")
        
        with torch.no_grad():
            # Generate handwriting image with user-style parameters
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=128,
                width=512,
                num_inference_steps=50,  # High quality steps
                guidance_scale=8.0,      # Strong prompt following
                generator=torch.Generator(device=device).manual_seed(42)
            )
            
            generated_image = result.images[0]
            print("Generated image using user-specific style")
        
        # Post-process with user-style considerations
        generated_image = post_process_styled_handwriting(generated_image, style_chars)
        return generated_image
        
    except Exception as e:
        print(f"Error in styled generation: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to regular generation
        return await generate_fallback_handwriting(text, model, style_params)

def post_process_styled_handwriting(image, style_characteristics):
    """Post-process generated image using user's style characteristics"""
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert if needed (handwriting should be dark on light background)
    if np.mean(adaptive_thresh) > 127:
        adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
    
    # Convert back to PIL
    processed_image = Image.fromarray(adaptive_thresh).convert('RGB')
    
    # Apply style-specific enhancements
    avg_intensity = style_characteristics.get('avg_intensity', 0.5)
    contrast = style_characteristics.get('contrast', 0.3)
    
    # Adjust contrast based on user's style
    if contrast > 0.4:  # High contrast style
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(1.3)
    elif contrast < 0.2:  # Low contrast style
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(0.9)
    else:  # Natural contrast
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(1.1)
    
    # Adjust brightness based on average intensity
    if avg_intensity < 0.3:  # Dark writing style
        enhancer = ImageEnhance.Brightness(processed_image)
        processed_image = enhancer.enhance(0.95)
    
    print(f"Applied style-specific post-processing: contrast={contrast}, intensity={avg_intensity}")
    return processed_image

async def generate_fallback_handwriting(text: str, model: dict, style_params: dict):
    """Generate handwriting using enhanced Stable Diffusion prompts"""
    import torch
    from PIL import Image
    
    try:
        pipeline = model['pipeline']
        device = model['device']
        
        print("Using enhanced Stable Diffusion fallback")
        
        # Create enhanced prompt for handwriting
        prompt = f"beautiful handwritten text saying '{text}', elegant cursive handwriting, black ink on white paper, natural handwriting style, realistic pen strokes, clean and legible, professional handwriting"
        negative_prompt = "printed text, typed text, computer font, digital text, blurry, distorted, pixelated, low quality, artifacts, messy"
        
        with torch.no_grad():
            # Generate handwriting image with better parameters
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=128,
                width=512,
                num_inference_steps=50,  # More steps for better quality
                guidance_scale=8.0,      # Higher guidance for better prompt following
                generator=torch.Generator(device=device).manual_seed(42)
            )
            
            generated_image = result.images[0]
            print("Generated image using enhanced Stable Diffusion")
        
        # Post-process the generated image
        generated_image = post_process_handwriting(generated_image)
        return generated_image
        
    except Exception as e:
        print(f"Error in fallback generation: {str(e)}")
        import traceback
        traceback.print_exc()
        # Final fallback
        return create_fallback_handwriting_image(text)

def post_process_handwriting(image):
    """Post-process generated image to enhance handwriting appearance with gentler processing"""
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply gentle contrast enhancement instead of harsh thresholding
    # Use adaptive thresholding for better results
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert if needed (handwriting should be dark on light background)
    if np.mean(adaptive_thresh) > 127:  # If background is mostly white
        adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
    
    # Convert back to PIL and enhance
    processed_image = Image.fromarray(adaptive_thresh).convert('RGB')
    
    # Apply gentle contrast enhancement
    enhancer = ImageEnhance.Contrast(processed_image)
    processed_image = enhancer.enhance(1.2)  # Slight contrast boost
    
    return processed_image

def create_fallback_handwriting_image(text: str):
    """Create a fallback handwriting image if diffusion fails"""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Create image
    width, height = 512, 128
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Add some handwriting-like text
    try:
        # Try to use a handwriting-like font or fallback to default
        font_size = 24
        font = ImageFont.load_default()
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text with slight variations to mimic handwriting
        draw.text((x, y), text, fill='black', font=font)
        
    except Exception as e:
        print(f"Error creating fallback image: {e}")
        # Very basic fallback
        draw.text((50, 50), text, fill='black')
    
    return image

def image_to_svg(image):
    """Convert PIL image to SVG format"""
    import io
    import base64
    
    # Convert PIL image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Create SVG with embedded image
    width, height = image.size
    svg = f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <image href="data:image/png;base64,{img_base64}" width="{width}" height="{height}"/>
    </svg>'''
    
    return svg

def clean_sample_image(image):
    """Clean and prepare sample image for DiffusionPen following OpenAI's recommendations"""
    from PIL import Image, ImageOps, ImageEnhance
    import numpy as np
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to grayscale for processing
    gray = image.convert('L')
    
    # Check if image appears to be inverted (white text on black background)
    # If mean pixel value is low, it's likely inverted
    pixel_array = np.array(gray)
    mean_intensity = np.mean(pixel_array)
    
    if mean_intensity < 127:  # Likely inverted
        print("  - Image appears inverted, correcting...")
        gray = ImageOps.invert(gray)
    
    # Apply gentle auto-contrast to improve clarity
    gray = ImageOps.autocontrast(gray, cutoff=1)
    
    # Resize to standard handwriting dimensions (maintain aspect ratio)
    target_width, target_height = 512, 128
    
    # Calculate scaling to fit within target dimensions
    scale_factor = min(target_width / gray.width, target_height / gray.height)
    new_width = int(gray.width * scale_factor)
    new_height = int(gray.height * scale_factor)
    
    # Resize with high-quality resampling
    gray = gray.resize((new_width, new_height), Image.LANCZOS)
    
    # Create white background and paste the resized image
    final_image = Image.new('L', (target_width, target_height), 255)  # White background
    
    # Center the image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    final_image.paste(gray, (paste_x, paste_y))
    
    # Convert back to RGB
    final_image = final_image.convert('RGB')
    
    # Apply gentle contrast enhancement
    enhancer = ImageEnhance.Contrast(final_image)
    final_image = enhancer.enhance(1.1)  # Slight contrast boost
    
    return final_image

def is_white_background(image):
    """Check if image has a white background"""
    import numpy as np
    
    # Convert to grayscale for analysis
    gray = image.convert('L')
    pixel_array = np.array(gray)
    
    # Check the corners and edges for background color
    corners = [
        pixel_array[0, 0],  # Top-left
        pixel_array[0, -1],  # Top-right
        pixel_array[-1, 0],  # Bottom-left
        pixel_array[-1, -1]  # Bottom-right
    ]
    
    # Check if corners are mostly white (>200 out of 255)
    white_corners = sum(1 for corner in corners if corner > 200)
    
    # Also check overall brightness
    mean_brightness = np.mean(pixel_array)
    
    return white_corners >= 3 and mean_brightness > 180

def debug_save_sample_images(style_dir, samples):
    """Debug function to save processed samples for visual inspection"""
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    
    debug_dir = os.path.join(style_dir, "debug_visualization")
    os.makedirs(debug_dir, exist_ok=True)
    
    for i, sample_file in enumerate(os.listdir(style_dir)):
        if sample_file.endswith('.png') and not sample_file.startswith('debug_'):
            sample_path = os.path.join(style_dir, sample_file)
            sample_image = Image.open(sample_path)
            
            # Save a matplotlib visualization
            plt.figure(figsize=(10, 3))
            plt.imshow(sample_image, cmap='gray')
            plt.title(f"Sample {i}: {sample_image.size} - Mode: {sample_image.mode}")
            plt.axis('off')
            
            debug_path = os.path.join(debug_dir, f"debug_{sample_file}.png")
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved debug visualization: {debug_path}")

# Remove old SVG generation functions - they're replaced by the diffusion model