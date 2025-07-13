import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import "https://deno.land/x/xhr@0.1.0/mod.ts";

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

    const openaiApiKey = Deno.env.get('OPENAI_API_KEY');
    if (!openaiApiKey) {
      return new Response(
        JSON.stringify({ error: 'OpenAI API key not configured' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('Generating handwriting for text:', text);
    console.log('Style characteristics:', styleCharacteristics);

    // Create a detailed prompt for handwriting generation
    const handwritingStyle = styleCharacteristics?.slant > 0 ? 'slanted' : 'upright';
    const spacing = styleCharacteristics?.spacing > 1 ? 'wide' : 'normal';
    const thickness = styleCharacteristics?.strokeWidth > 2 ? 'thick' : 'thin';
    
    const prompt = `High-quality handwritten text on clean white paper: "${text}". ${handwritingStyle} handwriting style, ${spacing} letter spacing, ${thickness} pen strokes. Natural, realistic handwriting with slight imperfections. Black ink on white background. Sharp, clear image.`;
    
    console.log('Generated prompt:', prompt);
    
    // Use OpenAI's gpt-image-1 model for superior handwriting generation
    const response = await fetch('https://api.openai.com/v1/images/generations', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openaiApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-image-1',
        prompt: prompt,
        n: 1,
        size: '1536x1024',
        quality: 'high',
        output_format: 'png',
        background: 'opaque'
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`OpenAI API error: ${response.status} ${response.statusText}`, errorText);
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    console.log('OpenAI response received:', JSON.stringify(data, null, 2));
    
    // The response format might be different - let's check what we get
    let base64Image;
    if (data.data && data.data[0]) {
      // Standard response format
      base64Image = data.data[0].b64_json || data.data[0].url;
    } else if (data.b64_json) {
      // Direct base64 format
      base64Image = data.b64_json;
    } else {
      console.error('Unexpected OpenAI response format:', data);
      throw new Error('Unexpected response format from OpenAI');
    }

    if (!base64Image) {
      throw new Error('No image data received from OpenAI');
    }

    // Create an SVG with the embedded base64 image
    const handwritingSvg = `<svg width="800" height="400" viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="white"/>
      <image href="data:image/png;base64,${base64Image}" x="0" y="0" width="800" height="400" preserveAspectRatio="xMidYMid meet"/>
    </svg>`;

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
    
    // Return error response instead of fallback
    return new Response(
      JSON.stringify({ 
        error: 'Failed to generate handwriting',
        details: error.message 
      }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});