import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import Replicate from "https://esm.sh/replicate@0.25.2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface HandwritingRequest {
  text: string;
  styleCharacteristics?: {
    slant?: number;
    spacing?: number;
    strokeWidth?: number;
    baseline?: string;
  };
  samples?: string[]; // Base64 encoded sample images
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { text, styleCharacteristics, samples }: HandwritingRequest = await req.json();
    
    if (!text) {
      return new Response(
        JSON.stringify({ error: 'Text is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const replicateApiKey = Deno.env.get('REPLICATE_API_KEY');
    if (!replicateApiKey) {
      return new Response(
        JSON.stringify({ error: 'Replicate API key not configured' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('Generating handwriting for text:', text);
    console.log('Style characteristics:', styleCharacteristics);

    const replicate = new Replicate({
      auth: replicateApiKey,
    });

    // Use a different approach - generate an image that looks like handwriting
    // Using a more accessible text-to-image model that can create handwriting
    const prompt = `Handwritten text on white paper: "${text}". Natural handwriting style, ${styleCharacteristics?.slant ? 'slanted' : 'upright'} writing, ${styleCharacteristics?.spacing === 'wide' ? 'wide' : 'normal'} letter spacing, ${styleCharacteristics?.strokeWidth === 'thick' ? 'thick' : 'thin'} pen strokes. Clean white background, black ink, realistic handwriting`;
    
    console.log('Generated prompt:', prompt);
    
    const output = await replicate.run(
      "black-forest-labs/flux-schnell",
      {
        input: {
          prompt: prompt,
          go_fast: true,
          megapixels: "1",
          num_outputs: 1,
          aspect_ratio: "16:9",
          output_format: "webp",
          output_quality: 80,
          num_inference_steps: 4
        }
      }
    );

    console.log('Replicate output:', output);

    // Convert the output to SVG format if it's an image
    let handwritingSvg;
    
    if (Array.isArray(output) && output.length > 0) {
      // If we get an image URL, we'll convert it to an SVG with an embedded image
      const imageUrl = output[0];
      
      handwritingSvg = `<svg width="800" height="200" viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        <rect width="100%" height="100%" fill="white"/>
        <image href="${imageUrl}" x="20" y="20" width="760" height="160" preserveAspectRatio="xMidYMid meet"/>
      </svg>`;
    } else {
      // Fallback to a simple SVG text representation
      const words = text.split(' ');
      const letterSpacing = (styleCharacteristics?.spacing || 1) * 40;
      const strokeWidth = styleCharacteristics?.strokeWidth || 2;
      
      let svgContent = `<svg width="800" height="200" viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="white"/>
        <style>
          .handwriting {
            font-family: 'Brush Script MT', cursive;
            font-size: 24px;
            fill: #1e3a8a;
            stroke: #1e3a8a;
            stroke-width: ${strokeWidth}px;
          }
        </style>`;
      
      let x = 40;
      let y = 100;
      
      words.forEach((word, wordIndex) => {
        if (x + word.length * letterSpacing > 750) {
          x = 40;
          y += 40;
        }
        
        // Add some randomness to make it look more natural
        const randomY = y + (Math.random() - 0.5) * 8;
        const randomSlant = (styleCharacteristics?.slant || 0) + (Math.random() - 0.5) * 2;
        
        svgContent += `<text x="${x}" y="${randomY}" class="handwriting" transform="skewX(${randomSlant})">${word}</text>`;
        x += (word.length * letterSpacing) + 20;
      });
      
      svgContent += '</svg>';
      handwritingSvg = svgContent;
    }

    console.log('Successfully generated handwriting');

    return new Response(
      JSON.stringify({ 
        handwritingSvg: handwritingSvg,
        characteristics: styleCharacteristics 
      }),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error('Error in generate-handwriting function:', error);
    
    // Fallback: Generate a simple but readable SVG
    const { text } = await req.json();
    const fallbackSvg = `<svg width="800" height="200" viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="white"/>
      <style>
        .fallback-text {
          font-family: 'Brush Script MT', 'Lucida Handwriting', cursive;
          font-size: 28px;
          fill: #1e3a8a;
        }
      </style>
      <text x="40" y="100" class="fallback-text">${text || 'Sample handwriting text'}</text>
    </svg>`;
    
    return new Response(
      JSON.stringify({ 
        handwritingSvg: fallbackSvg,
        characteristics: {} 
      }),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});