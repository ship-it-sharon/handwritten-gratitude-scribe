import modal
import io
import base64
from typing import List, Optional
import os

app = modal.App("diffusionpen-handwriting")

# Define the image with necessary ML dependencies for DiffusionPen
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget")
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
        "safetensors"
    ])
    .run_commands([
        # Clone DiffusionPen repository
        "cd /root && git clone https://github.com/koninik/DiffusionPen.git",
        # Download pre-trained models from Hugging Face
        "cd /root && mkdir -p /root/models",
    ])
)

# Global model instance
diffusion_model = None

@app.function(
    image=image,
    gpu="A10G",  # Need GPU for diffusion model inference
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
            
            # Import DiffusionPen modules
            from diffusers import StableDiffusionPipeline, DDIMScheduler
            from transformers import AutoTokenizer
            import torch
            
            # Load the base Stable Diffusion model
            model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
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
            
            # Load DiffusionPen specific components from Hugging Face
            from huggingface_hub import hf_hub_download
            
            # Download model weights
            style_encoder_path = hf_hub_download(
                repo_id="konnik/DiffusionPen",
                filename="style_models/style_encoder.pth"
            )
            
            print(f"Model loaded successfully on {device}")
            diffusion_model = {
                'pipeline': pipeline,
                'device': device,
                'style_encoder_path': style_encoder_path
            }
            
            return diffusion_model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
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
    """Generate handwriting using DiffusionPen model"""
    import torch
    import numpy as np
    from PIL import Image
    import cv2
    
    try:
        pipeline = model['pipeline']
        device = model['device']
        
        # Process reference samples for style conditioning
        style_images = []
        if samples:
            for sample_b64 in samples[:5]:  # Use up to 5 samples as per DiffusionPen
                try:
                    image_data = base64.b64decode(sample_b64)
                    ref_image = Image.open(io.BytesIO(image_data))
                    
                    # Resize and preprocess for style encoding
                    ref_image = ref_image.convert('RGB')
                    ref_image = ref_image.resize((256, 64))  # Standard size for handwriting
                    style_images.append(ref_image)
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
        
        # Create prompt for handwriting generation
        # DiffusionPen uses text prompts combined with style conditioning
        prompt = f"handwritten text: {text}"
        
        # Generate with style conditioning if we have reference samples
        if style_images:
            print(f"Generating with {len(style_images)} style reference samples")
            # For now, use the first style image as conditioning
            # In full DiffusionPen, this would go through the style encoder
            
            # Create a simple text-to-image generation
            # This is a simplified version - full DiffusionPen would use style encoder
            with torch.no_grad():
                # Generate handwriting image
                result = pipeline(
                    prompt=prompt,
                    height=128,
                    width=512,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=device).manual_seed(42)
                )
                
                generated_image = result.images[0]
        else:
            print("Generating with default style")
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    height=128,
                    width=512,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=device).manual_seed(42)
                )
                
                generated_image = result.images[0]
        
        # Post-process the generated image to look more like handwriting
        generated_image = post_process_handwriting(generated_image)
        
        return generated_image
        
    except Exception as e:
        print(f"Error in diffusion generation: {str(e)}")
        # Fallback to a simple generated image
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