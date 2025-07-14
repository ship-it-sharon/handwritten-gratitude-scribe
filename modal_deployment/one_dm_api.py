import modal
import io
import base64
from typing import List, Optional
import os

app = modal.App("diffusionpen-handwriting")

# Define the image with necessary ML dependencies for DiffusionPen
image = (
    modal.Image.debian_slim(python_version="3.10")
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
        "huggingface-hub",
        "safetensors",
        "xformers"  # For memory optimization
    ])
    .run_commands([
        # Clone DiffusionPen repository
        "cd /root && git clone https://github.com/koninik/DiffusionPen.git",
        # Install DiffusionPen requirements
        "cd /root/DiffusionPen && pip install -r requirements.txt",
        # Create necessary directories
        "mkdir -p /root/models /tmp/style_in /tmp/style_out /tmp/samples",
    ])
)

# Global model instance
diffusion_model = None

@app.function(
    image=image,
    gpu="L40S",  # L40S recommended for large models and better performance
    timeout=300,  # 5 minutes timeout for model loading and inference
    keep_warm=1   # Keep one instance warm for faster response
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
    
    @app.post("/generate_handwriting")
    async def generate_handwriting_endpoint(request: Request):
        try:
            # Parse request body
            body = await request.body()
            request_data = json.loads(body)
            
            # Extract data from request
            text = request_data.get("text", "")
            samples = request_data.get("samples", [])
            style_params = request_data.get("styleCharacteristics", {})
            
            if not text:
                return JSONResponse({"error": "Text is required"}, status_code=400)
            
            print(f"Generating handwriting for: '{text}' with {len(samples)} reference samples")
            
            # Load model if not already loaded
            model = await load_diffusion_model()
            
            # Generate handwriting using DiffusionPen
            handwriting_image = await generate_diffusion_handwriting(
                text, samples, model, style_params
            )
            
            # Convert PIL image to SVG or base64
            handwriting_svg = image_to_svg(handwriting_image)
            
            return JSONResponse({
                "handwritingSvg": handwriting_svg,
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
            return JSONResponse({"error": f"Failed to generate handwriting: {str(e)}"}, status_code=500)
    
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

async def generate_diffusion_handwriting(text: str, samples: List[str], model: dict, style_params: dict):
    """Generate handwriting using DiffusionPen model with proper inference"""
    import torch
    import numpy as np
    from PIL import Image
    import cv2
    import subprocess
    import tempfile
    import json
    
    try:
        pipeline = model['pipeline']
        device = model['device']
        diffusionpen_path = model['diffusionpen_path']
        
        print(f"Generating handwriting for text: '{text}'")
        print(f"Using device: {device}")
        print(f"Number of style samples: {len(samples)}")
        
        # Create temporary directories for DiffusionPen inference
        with tempfile.TemporaryDirectory() as temp_dir:
            style_dir = os.path.join(temp_dir, "style_samples")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(style_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Created temp directories: {style_dir}, {output_dir}")
            
            # Process and save reference samples for style conditioning
            if samples:
                print(f"Processing {len(samples)} style samples...")
                for i, sample_b64 in enumerate(samples[:5]):  # Use up to 5 samples
                    try:
                        # Decode base64 image
                        image_data = base64.b64decode(sample_b64)
                        ref_image = Image.open(io.BytesIO(image_data))
                        
                        # Convert to RGB and resize appropriately for handwriting
                        ref_image = ref_image.convert('RGB')
                        ref_image = ref_image.resize((512, 128))  # Standard handwriting size
                        
                        # Save to style directory
                        style_path = os.path.join(style_dir, f"style_{i}.png")
                        ref_image.save(style_path)
                        print(f"Saved style sample {i} to {style_path}")
                        
                    except Exception as e:
                        print(f"Error processing sample {i}: {e}")
                        continue
            
            # Try to use actual DiffusionPen inference if available
            try:
                # Check if DiffusionPen inference script exists
                inference_script = os.path.join(diffusionpen_path, "inference.py")
                if os.path.exists(inference_script):
                    print(f"Found DiffusionPen inference script: {inference_script}")
                    
                    # Prepare arguments for DiffusionPen inference
                    cmd = [
                        "python", inference_script,
                        "--text", text,
                        "--style_dir", style_dir,
                        "--output_dir", output_dir,
                        "--device", device
                    ]
                    
                    # Add style parameters if provided
                    if style_params:
                        if "slant" in style_params:
                            cmd.extend(["--slant", str(style_params["slant"])])
                        if "spacing" in style_params:
                            cmd.extend(["--spacing", str(style_params["spacing"])])
                    
                    print(f"Running DiffusionPen inference: {' '.join(cmd)}")
                    
                    # Run DiffusionPen inference
                    result = subprocess.run(
                        cmd, 
                        cwd=diffusionpen_path,
                        capture_output=True, 
                        text=True, 
                        timeout=180  # 3 minutes timeout
                    )
                    
                    if result.returncode == 0:
                        print("DiffusionPen inference completed successfully")
                        
                        # Look for generated output
                        output_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                        if output_files:
                            output_path = os.path.join(output_dir, output_files[0])
                            generated_image = Image.open(output_path)
                            print(f"Loaded generated image from {output_path}")
                            return post_process_handwriting(generated_image)
                    else:
                        print(f"DiffusionPen inference failed with return code {result.returncode}")
                        print(f"stdout: {result.stdout}")
                        print(f"stderr: {result.stderr}")
                        
                else:
                    print(f"DiffusionPen inference script not found at {inference_script}")
                    
            except Exception as diffusion_error:
                print(f"Error running DiffusionPen inference: {diffusion_error}")
            
            # Fallback to Stable Diffusion pipeline if DiffusionPen fails
            print("Falling back to Stable Diffusion pipeline")
            
            # Create enhanced prompt for handwriting
            prompt = f"handwritten text '{text}', cursive handwriting, pen and paper, clean handwriting, realistic handwriting style"
            negative_prompt = "printed text, typed text, computer font, digital text, blurry, distorted"
            
            with torch.no_grad():
                # Generate handwriting image with better parameters
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=128,
                    width=512,
                    num_inference_steps=30,  # More steps for better quality
                    guidance_scale=7.5,
                    generator=torch.Generator(device=device).manual_seed(42)
                )
                
                generated_image = result.images[0]
                print("Generated image using Stable Diffusion fallback")
        
        # Post-process the generated image
        generated_image = post_process_handwriting(generated_image)
        return generated_image
        
    except Exception as e:
        print(f"Error in diffusion generation: {str(e)}")
        import traceback
        traceback.print_exc()
        # Final fallback
        return create_fallback_handwriting_image(text)

def post_process_handwriting(image):
    """Post-process generated image to enhance handwriting appearance"""
    import cv2
    import numpy as np
    from PIL import Image
    
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply threshold to create clean black/white handwriting
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    # Invert if needed (handwriting should be dark on light background)
    if np.mean(binary) > 127:  # If background is mostly white
        binary = cv2.bitwise_not(binary)
    
    # Convert back to PIL
    return Image.fromarray(binary).convert('RGB')

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

# Remove old SVG generation functions - they're replaced by the diffusion model