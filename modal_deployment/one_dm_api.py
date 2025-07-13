#!/usr/bin/env python3
"""
One-DM handwriting generation API on Modal
"""

import modal

app = modal.App("one-dm-handwriting")

# Define the Modal image with all necessary dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=1.9.0",
    "torchvision>=0.10.0", 
    "pillow>=8.3.0",
    "numpy>=1.21.0",
    "opencv-python>=4.5.0",
    "scikit-image>=0.18.0",
    "matplotlib>=3.4.0",
    "requests>=2.25.0"
]).apt_install([
    "git",
    "wget",
    "libgl1-mesa-glx",
    "libglib2.0-0"
])

@app.function(
    image=image,
    gpu="any",
    timeout=300,
    container_idle_timeout=240
)
@modal.asgi_app()
def fastapi_app():
    """Create FastAPI app"""
    # Import FastAPI inside the function
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import json
    
    app = FastAPI()
    
    @app.post("/generate_handwriting")
    async def generate_handwriting_endpoint(request: Request):
        """Generate handwriting from text and reference samples"""
        # Import dependencies only when needed
        import base64
        import io
        from PIL import Image, ImageDraw, ImageFont
        
        try:
            # Parse request body
            body = await request.body()
            request_data = json.loads(body)
            
            # Extract data from request
            text = request_data.get("text", "")
            samples = request_data.get("samples", [])
            
            if not text:
                return JSONResponse({"error": "Text is required"}, status_code=400)
            
            print(f"Received request for text: '{text}' with {len(samples)} samples")
            
            # Generate placeholder handwriting using PIL only
            width, height = 800, 200
            image = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(image)
            
            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Draw the text
            draw.text((50, 80), text, fill='black', font=font)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            handwriting_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Create SVG with the generated handwriting
            svg_content = f'''<svg width="800" height="200" viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg">
                <rect width="100%" height="100%" fill="white"/>
                <image href="data:image/png;base64,{handwriting_base64}" x="0" y="0" width="800" height="200" preserveAspectRatio="xMidYMid meet"/>
            </svg>'''
            
            return JSONResponse({
                "handwritingSvg": svg_content,
                "styleCharacteristics": {
                    "slant": 0.1,
                    "spacing": 1.0,
                    "strokeWidth": 2.0,
                    "baseline": "straight"
                }
            })
            
        except Exception as e:
            print(f"Error generating handwriting: {str(e)}")
            return JSONResponse({"error": f"Failed to generate handwriting: {str(e)}"}, status_code=500)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return JSONResponse({"status": "healthy", "service": "one-dm-handwriting"})
    
    return app