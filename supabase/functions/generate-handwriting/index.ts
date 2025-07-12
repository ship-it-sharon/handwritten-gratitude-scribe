import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

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
    const prompt = `Generate an SVG representation of handwritten text that mimics human handwriting characteristics.

Text to write: "${text}"

Style characteristics:
- Slant: ${styleCharacteristics?.slant || 'slight right slant'}
- Letter spacing: ${styleCharacteristics?.spacing || 'normal'}
- Stroke width: ${styleCharacteristics?.strokeWidth || 'medium'}
- Baseline: ${styleCharacteristics?.baseline || 'slightly irregular'}

Requirements:
- Create realistic handwriting with natural variations
- Use SVG format with proper stroke paths
- Include subtle imperfections and character variations
- Make it look like natural cursive or print handwriting
- Use a consistent ink-blue color (#1e3a8a)
- Size should fit within 800x200 viewBox
- Add slight tremor and pressure variations

Return only the SVG code without any explanatory text.`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openaiApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: 'You are an expert in generating SVG representations of handwritten text. You create realistic, natural-looking handwriting with proper stroke paths and human-like variations.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 2000,
        temperature: 0.7
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error('OpenAI API error:', errorData);
      return new Response(
        JSON.stringify({ error: 'Failed to generate handwriting' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const data = await response.json();
    const generatedSvg = data.choices[0]?.message?.content;

    if (!generatedSvg) {
      return new Response(
        JSON.stringify({ error: 'No handwriting generated' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('Successfully generated handwriting SVG');

    return new Response(
      JSON.stringify({ 
        handwritingSvg: generatedSvg,
        characteristics: styleCharacteristics 
      }),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error('Error in generate-handwriting function:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});