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
            
            You must respond ONLY with valid JSON in this exact format:
            {
              "isValid": true/false,
              "isHandwriting": true/false,
              "textMatches": true/false,
              "extractedText": "what you read from the image",
              "feedback": "helpful message for the user"
            }
            
            Set isValid to true only if both isHandwriting and textMatches are true.
            For text matching, ignore case differences and minor punctuation variations.
            Focus on content accuracy rather than exact capitalization or punctuation.
            Do not include any text before or after the JSON.`
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: `Expected text: "${expectedText}"\n\nPlease validate this handwriting sample and respond with only JSON.`
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
        max_tokens: 300,
        temperature: 0
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
      
      // Additional client-side validation for text matching with normalization
      if (validation.extractedText && expectedText) {
        const normalizeText = (text: string) => text.toLowerCase().replace(/[^\w\s]/g, '').trim();
        const normalizedExpected = normalizeText(expectedText);
        const normalizedExtracted = normalizeText(validation.extractedText);
        
        // Override textMatches if normalized texts are equal
        if (normalizedExpected === normalizedExtracted) {
          validation.textMatches = true;
          validation.isValid = validation.isHandwriting && validation.textMatches;
        }
      }
      
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