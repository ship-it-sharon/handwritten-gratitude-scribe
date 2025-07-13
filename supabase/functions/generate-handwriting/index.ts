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

    const modalApiUrl = Deno.env.get('MODAL_API_URL');
    if (!modalApiUrl) {
      return new Response(
        JSON.stringify({ error: 'Modal API URL not configured. Please set MODAL_API_URL in Supabase secrets.' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('Generating handwriting for text:', text);
    console.log('Style characteristics:', styleCharacteristics);
    console.log('Number of samples:', samples?.length || 0);

    // Prepare request for Modal API
    const modalRequest = {
      text: text,
      samples: samples || []
    };

    console.log('Sending request to Modal API:', modalApiUrl);

    // Call Modal API to generate handwriting
    const response = await fetch(`${modalApiUrl}/generate_handwriting`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(modalRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Modal API error: ${response.status} ${response.statusText}`, errorText);
      throw new Error(`Modal API error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const data = await response.json();
    console.log('Modal API response received:', JSON.stringify(data, null, 2));
    
    // Extract the handwriting SVG from Modal response
    const handwritingSvg = data.handwritingSvg;
    if (!handwritingSvg) {
      throw new Error('No handwriting SVG received from Modal API');
    }

    console.log('Successfully generated handwriting');

    return new Response(
      JSON.stringify({ 
        handwritingSvg: handwritingSvg,
        characteristics: data.styleCharacteristics || styleCharacteristics 
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