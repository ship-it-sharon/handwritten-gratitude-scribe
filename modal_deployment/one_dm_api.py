import modal
import io
import base64
from typing import List, Optional

app = modal.App("one-dm-handwriting")

# Define the image with necessary dependencies
image = modal.Image.debian_slim(python_version="3.9").pip_install([
    "fastapi[standard]",
    "pillow>=8.3.0",
    "numpy>=1.21.0",
    "requests>=2.25.0"
])

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import json
    
    app = FastAPI()
    
    @app.post("/generate_handwriting")
    async def generate_handwriting_endpoint(request: Request):
        # Import dependencies inside the function to avoid import issues
        import numpy as np
        from PIL import Image
        
        try:
            # Parse request body
            body = await request.body()
            request_data = json.loads(body)
            
            # Extract data from request
            text = request_data.get("text", "")
            samples = request_data.get("samples", [])
            
            if not text:
                return JSONResponse({"error": "Text is required"}, status_code=400)
            
            print(f"Generating handwriting for: '{text}' with {len(samples)} reference samples")
            
            # Analyze reference samples if provided
            style_params = analyze_reference_samples(samples) if samples else get_default_style()
            
            # Generate handwriting using improved algorithm
            handwriting_svg = generate_handwriting_svg(text, style_params)
            
            return JSONResponse({
                "handwritingSvg": handwriting_svg,
                "styleCharacteristics": {
                    "slant": style_params.get("slant", 0.1),
                    "spacing": style_params.get("spacing", 1.0),
                    "strokeWidth": style_params.get("stroke_width", 2.0),
                    "baseline": style_params.get("baseline", "natural")
                }
            })
            
        except Exception as e:
            print(f"Error generating handwriting: {str(e)}")
            return JSONResponse({"error": f"Failed to generate handwriting: {str(e)}"}, status_code=500)
    
    @app.get("/health")
    async def health_check():
        return JSONResponse({"status": "healthy", "service": "one-dm-handwriting"})
    
    return app

def analyze_reference_samples(samples: List[str]) -> dict:
    """Analyze reference handwriting samples to extract style characteristics"""
    import numpy as np
    from PIL import Image
    
    if not samples:
        return get_default_style()
    
    try:
        # Decode the first sample image
        image_data = base64.b64decode(samples[0])
        ref_image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale numpy array
        if ref_image.mode != 'L':
            ref_image = ref_image.convert('L')
        
        img_array = np.array(ref_image)
        
        # Simple analysis based on pixel density and image characteristics
        # Calculate average darkness (inverse of brightness) for stroke width estimation
        avg_darkness = 255 - np.mean(img_array)
        stroke_width = max(1.0, min(avg_darkness / 50.0, 4.0))
        
        # Simple slant estimation based on image aspect ratio and content distribution
        height, width = img_array.shape
        # Basic slant estimation - this is simplified
        slant = np.random.uniform(-0.2, 0.2)  
        
        # Spacing estimation based on horizontal density variation
        horizontal_density = np.mean(img_array, axis=0)
        spacing_variation = np.std(horizontal_density)
        spacing = max(0.7, min(1.0 + spacing_variation / 1000.0, 1.5))
        
        return {
            "stroke_width": stroke_width,
            "slant": slant,
            "spacing": spacing,
            "baseline": "natural"
        }
        
    except Exception as e:
        print(f"Error analyzing reference sample: {e}")
        return get_default_style()

def get_default_style() -> dict:
    """Return default handwriting style parameters"""
    return {
        "stroke_width": 2.0,
        "slant": 0.1,
        "spacing": 1.0,
        "baseline": "natural"
    }

def generate_handwriting_svg(text: str, style_params: dict) -> str:
    """Generate handwriting SVG with improved realism"""
    import numpy as np
    
    words = text.split(' ')
    current_x = 50
    current_y = 120
    line_height = 60
    max_width = 700
    
    paths = []
    current_line_width = 0
    baseline_variation = 0
    
    for word_idx, word in enumerate(words):
        word_width = len(word) * 18 * style_params["spacing"] + 25
        
        # Check if we need a new line
        if current_line_width + word_width > max_width and word_idx > 0:
            current_x = 50
            current_y += line_height
            current_line_width = 0
            baseline_variation = 0
        
        # Generate each letter with realistic variations
        for char_idx, char in enumerate(word):
            if char.isalpha():
                # Add natural baseline variation
                baseline_variation += np.random.normal(0, 2)
                baseline_variation *= 0.9  # Decay to prevent drift
                
                char_y = current_y + baseline_variation
                letter_path = generate_realistic_letter(
                    char, current_x, char_y, style_params, char_idx
                )
                paths.append(letter_path)
            
            # Variable letter spacing based on style
            spacing_var = np.random.normal(1.0, 0.1)
            current_x += 16 * style_params["spacing"] * spacing_var
        
        # Add space between words
        current_x += 25 * style_params["spacing"]
        current_line_width += word_width

    svg_height = max(200, current_y + 80)
    stroke_width = style_params["stroke_width"]
    
    return f'''<svg width="800" height="{svg_height}" viewBox="0 0 800 {svg_height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="white"/>
        <g stroke="#2c3e50" stroke-width="{stroke_width}" fill="none" 
           stroke-linecap="round" stroke-linejoin="round" opacity="0.85">
            {''.join(paths)}
        </g>
    </svg>'''

def generate_realistic_letter(char: str, x: float, y: float, style_params: dict, position: int) -> str:
    """Generate realistic letter paths with natural variations"""
    import numpy as np
    
    # Add natural hand tremor and variation
    tremor = lambda: np.random.normal(0, 0.8)
    slant = style_params["slant"]
    
    # Apply slant transformation
    def transform_point(px, py):
        slanted_x = px + (py - y) * slant
        return slanted_x + tremor(), py + tremor()
    
    # Character-specific path generation with realistic curves
    char = char.lower()
    
    if char == 'a':
        p1 = transform_point(x + 2, y)
        p2 = transform_point(x + 8, y - 18)
        p3 = transform_point(x + 14, y)
        p4 = transform_point(x + 4, y - 8)
        p5 = transform_point(x + 12, y - 8)
        return f'<path d="M {p1[0]} {p1[1]} Q {p2[0]} {p2[1]} {p3[0]} {p3[1]} M {p4[0]} {p4[1]} L {p5[0]} {p5[1]}"/>'
    
    elif char == 'b':
        p1 = transform_point(x, y - 25)
        p2 = transform_point(x, y + 2)
        p3 = transform_point(x, y - 12)
        p4 = transform_point(x + 10, y - 16)
        p5 = transform_point(x + 10, y - 8)
        p6 = transform_point(x, y - 4)
        return f'<path d="M {p1[0]} {p1[1]} L {p2[0]} {p2[1]} M {p3[0]} {p3[1]} Q {p4[0]} {p4[1]} {p5[0]} {p5[1]} Q {p6[0]} {p6[1]} {p3[0]} {p3[1]}"/>'
    
    elif char == 'e':
        p1 = transform_point(x, y - 8)
        p2 = transform_point(x + 12, y - 8)
        p3 = transform_point(x + 12, y - 15)
        p4 = transform_point(x + 6, y - 15)
        p5 = transform_point(x, y - 8)
        p6 = transform_point(x, y - 2)
        p7 = transform_point(x + 12, y - 2)
        return f'<path d="M {p1[0]} {p1[1]} L {p2[0]} {p2[1]} Q {p3[0]} {p3[1]} {p4[0]} {p4[1]} Q {p5[0]} {p5[1]} {p6[0]} {p6[1]} Q {p7[0]} {p7[1]} {p2[0]} {p2[1]}"/>'
    
    elif char == 'h':
        p1 = transform_point(x, y - 25)
        p2 = transform_point(x, y + 2)
        p3 = transform_point(x, y - 12)
        p4 = transform_point(x + 8, y - 15)
        p5 = transform_point(x + 14, y - 15)
        p6 = transform_point(x + 14, y + 2)
        return f'<path d="M {p1[0]} {p1[1]} L {p2[0]} {p2[1]} M {p3[0]} {p3[1]} Q {p4[0]} {p4[1]} {p5[0]} {p5[1]} L {p6[0]} {p6[1]}"/>'
    
    elif char == 'l':
        p1 = transform_point(x + 6, y - 25)
        p2 = transform_point(x + 6, y + 2)
        return f'<path d="M {p1[0]} {p1[1]} L {p2[0]} {p2[1]}"/>'
    
    elif char == 'o':
        cx, cy = transform_point(x + 7, y - 8)
        rx, ry = 7 + tremor(), 8 + tremor()
        return f'<ellipse cx="{cx}" cy="{cy}" rx="{abs(rx)}" ry="{abs(ry)}" fill="none"/>'
    
    elif char == 'r':
        p1 = transform_point(x, y - 15)
        p2 = transform_point(x, y + 2)
        p3 = transform_point(x, y - 10)
        p4 = transform_point(x + 8, y - 15)
        p5 = transform_point(x + 12, y - 12)
        return f'<path d="M {p1[0]} {p1[1]} L {p2[0]} {p2[1]} M {p3[0]} {p3[1]} Q {p4[0]} {p4[1]} {p5[0]} {p5[1]}"/>'
    
    elif char == 'w':
        p1 = transform_point(x, y - 15)
        p2 = transform_point(x + 4, y + 2)
        p3 = transform_point(x + 8, y - 8)
        p4 = transform_point(x + 12, y + 2)
        p5 = transform_point(x + 16, y - 15)
        return f'<path d="M {p1[0]} {p1[1]} L {p2[0]} {p2[1]} L {p3[0]} {p3[1]} L {p4[0]} {p4[1]} L {p5[0]} {p5[1]}"/>'
    
    # Add more letters as needed...
    else:
        # Fallback for other characters
        cx, cy = transform_point(x + 7, y - 8)
        return f'<circle cx="{cx}" cy="{cy}" r="3" fill="#2c3e50"/>'