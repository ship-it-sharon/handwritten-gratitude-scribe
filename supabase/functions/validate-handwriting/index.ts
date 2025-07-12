import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const openAIApiKey = Deno.env.get('OPENAI_API_KEY');

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { imageData, expectedText } = await req.json();

    if (!imageData || !expectedText) {
      return new Response(
        JSON.stringify({ error: 'Missing imageData or expectedText' }), 
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    console.log('Making OpenAI API request for validation...');
    
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openAIApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: `You are a handwriting validation assistant. Analyze the image and determine:
            1. Is this clearly handwritten text (not typed or printed)?
            2. Does the handwritten text match the expected text exactly?
            
            Return ONLY a valid JSON response with:
            - isValid: true if both conditions are met
            - isHandwriting: true if it's handwritten
            - textMatches: true if the text matches exactly
            - extractedText: what text you can read from the image
            - feedback: helpful message for the user
            
            Be strict about exact text matching - even small differences should result in textMatches: false.`
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: `Expected text: "${expectedText}"\n\nPlease validate this handwriting sample.`
              },
              {
                type: 'image_url',
                image_url: {
                  url: imageData
                }
              }
            ]
          }
        ],
        max_tokens: 500
      }),
    });

    console.log('OpenAI response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('OpenAI API error:', response.status, response.statusText, errorText);
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log('OpenAI response data:', JSON.stringify(data, null, 2));
    
    const content = data.choices[0].message.content;
    console.log('OpenAI content response:', content);
    
    try {
      // Parse the JSON response from OpenAI
      const validation = JSON.parse(content);
      
      return new Response(JSON.stringify(validation), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    } catch (parseError) {
      console.error('Failed to parse OpenAI response as JSON:', parseError);
      console.error('Raw OpenAI content:', content);
      
      // Try to extract information from non-JSON response
      const isHandwritingMention = content.toLowerCase().includes('handwrit');
      const isHandwritten = isHandwritingMention && !content.toLowerCase().includes('not handwrit') && !content.toLowerCase().includes('no handwrit');
      
      return new Response(JSON.stringify({
        isValid: false,
        isHandwriting: isHandwritten,
        textMatches: false,
        extractedText: '',
        feedback: isHandwritten 
          ? 'I can see this is handwriting, but I had trouble verifying if it matches the expected text. Please ensure your handwriting is clear and matches the prompt exactly.'
          : 'Please make sure you\'re uploading a clear photo of handwritten text that matches the prompt.'
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

  } catch (error) {
    console.error('Error in validate-handwriting function:', error);
    
    return new Response(JSON.stringify({
      isValid: false,
      isHandwriting: false,
      textMatches: false,
      extractedText: '',
      feedback: 'Technical error occurred. Please try again.',
      error: error.message
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});