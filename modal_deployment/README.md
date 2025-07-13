# One-DM Handwriting Generation - Modal Deployment

This directory contains the Modal deployment for the One-DM handwriting generation model.

## Setup

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```bash
   modal token new
   ```

3. **Deploy the model**:
   ```bash
   python deploy.py
   ```
   
   Or directly:
   ```bash
   modal deploy one_dm_api.py
   ```

## API Endpoints

Once deployed, the service provides:

- **POST /generate_handwriting**: Generate handwriting from text and reference samples
- **GET /health_check**: Health check endpoint

### Generate Handwriting Request

```json
{
  "text": "Hello, this is a test message",
  "samples": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."]
}
```

### Response

```json
{
  "handwritingSvg": "<svg>...</svg>",
  "styleCharacteristics": {
    "slant": 2.0,
    "spacing": 1.0,
    "strokeWidth": 2.0,
    "baseline": "straight"
  }
}
```

## Current Implementation

⚠️ **Note**: This is currently a placeholder implementation. The actual One-DM model integration requires:

1. **Complete One-DM Implementation**: The current code uses a placeholder model. We need to implement the full One-DM architecture.

2. **Model Weights**: Download or train the actual One-DM model weights.

3. **Proper Preprocessing**: Implement One-DM's specific image preprocessing pipeline.

4. **Text-to-Handwriting Pipeline**: Implement the character-by-character handwriting generation.

## Next Steps

1. **Deploy this placeholder**: Get the infrastructure running
2. **Get Modal endpoint URL**: Update Supabase edge function to call Modal instead of OpenAI
3. **Implement real One-DM**: Replace placeholder with actual model
4. **Fine-tune**: Optimize for quality and performance

## Cost Considerations

- Modal charges for GPU usage (~$0.50-$2.00/hour for A10G)
- Cold starts: ~30-60 seconds for model loading
- Warm inference: ~2-5 seconds per request
- Consider keeping one instance warm for production use