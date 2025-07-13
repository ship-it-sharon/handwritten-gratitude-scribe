import modal

app = modal.App("handwriting-simple")

image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pillow>=8.3.0",
    "requests>=2.25.0"
])

@app.function(image=image, timeout=300)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import json
    
    app = FastAPI()
    
    @app.post("/generate_handwriting")
    async def generate_handwriting_endpoint(request: Request):
        import base64
        import io
        from PIL import Image, ImageDraw, ImageFont
        
        try:
            body = await request.body()
            request_data = json.loads(body)
            text = request_data.get("text", "")
            
            if not text:
                return JSONResponse({"error": "Text is required"}, status_code=400)
            
            # Simple text rendering
            width, height = 800, 200
            image = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((50, 80), text, fill='black', font=font)
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            handwriting_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            svg_content = f'''<svg width="800" height="200" viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg">
                <rect width="100%" height="100%" fill="white"/>
                <image href="data:image/png;base64,{handwriting_base64}" x="0" y="0" width="800" height="200"/>
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
            return JSONResponse({"error": f"Failed: {str(e)}"}, status_code=500)
    
    @app.get("/health")
    async def health_check():
        return JSONResponse({"status": "healthy"})
    
    return app