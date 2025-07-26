import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface HandwritingRequest {
  text: string
  user_id?: string // User ID to fetch their trained model
  model_id?: string // Trained model ID for personalized generation (optional, for backward compatibility)
  styleCharacteristics?: {
    slant?: number
    spacing?: number
    strokeWidth?: number
    baseline?: string
    pressure?: number
  }
  samples?: string[] // base64 encoded images (fallback only)
}

serve(async (req) => {
  console.log('=== Edge function called ===')
  console.log('Method:', req.method)
  console.log('URL:', req.url)
  
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    console.log('Handling CORS preflight request')
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    console.log('Parsing request body...')
    const body = await req.json() as HandwritingRequest
    console.log('Request body parsed:', JSON.stringify(body, null, 2))
    
    if (!body.text) {
      return new Response(
        JSON.stringify({ error: 'Text is required' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    // Extract samples from request
    const samples = body.samples || [];
    
    console.log(`Generating handwriting for: "${body.text}" with ${samples.length} reference samples`)
    console.log('User ID provided:', body.user_id || 'none')
    console.log('Model ID provided:', body.model_id || 'none')

    // Initialize Supabase client  
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    // Fetch user's embedding URL from database if user_id is provided
    let modelUrl = null;
    if (body.user_id) {
      console.log('🔍 Fetching embedding URL for user:', body.user_id);
      
      const { data: modelData, error: modelError } = await supabase
        .from('user_style_models')
        .select('embedding_storage_url, training_status, model_id, created_at')
        .eq('user_id', body.user_id)
        .eq('training_status', 'completed')
        .order('created_at', { ascending: false })
        .limit(1)
        .maybeSingle();

        console.log('📋 Query results:', {
          data: modelData,
          error: modelError,
          user_id: body.user_id
        });

        if (modelError) {
          console.error('❌ Error fetching model data:', modelError);
        } else if (modelData?.embedding_storage_url) {
          modelUrl = modelData.embedding_storage_url;
          console.log('✅ Found Modal embedding storage path:', modelUrl);
          console.log('📊 Model details:', {
            model_id: modelData.model_id,
            created_at: modelData.created_at,
            training_status: modelData.training_status
          });
        } else {
          console.log('⚠️ No embedding storage URL found for model, will use samples fallback');
          console.log('🔍 Model data received:', modelData);
        }
    }
    
    // Follow ChatGPT's recommendation: pass user_id and model_url to Modal
    const maxRetries = 1
    const timeoutMs = 30000
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Attempting generation API call (attempt ${attempt}/${maxRetries})...`)
        
        const modalUrl = 'https://ship-it-sharon--diffusionpen-handwriting-fastapi-app.modal.run/'
        
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
        
        const requestPayload = {
          action: "generate",
          text: body.text,
          user_id: body.user_id,
          model_url: modelUrl, // Pass the embedding URL directly
          style_characteristics: body.styleCharacteristics || {},
          // Include samples as backup in case no trained model exists
          samples: samples.length > 0 ? samples.slice(0, 3) : []
        }
        
        console.log('Making POST request to Modal API...')
        console.log('Request payload:', JSON.stringify({
          ...requestPayload,
          model_url: requestPayload.model_url ? `${requestPayload.model_url.substring(0, 50)}...` : 'none',
          samples: requestPayload.samples ? `[${requestPayload.samples.length} samples]` : 'none'
        }))
        
        const modalResponse = await fetch(modalUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestPayload),
          signal: controller.signal
        })

        clearTimeout(timeoutId)

        console.log(`Modal API response status: ${modalResponse.status}`)
        
        if (modalResponse.ok) {
          const modalData = await modalResponse.json()
          console.log(`Modal API call successful on attempt ${attempt}`)
          console.log('Modal handwritingSvg length:', modalData.handwritingSvg?.length || 'undefined')
          
          const responseData = {
            handwritingSvg: modalData.handwritingSvg,
            styleCharacteristics: modalData.styleCharacteristics || {
              slant: 0.15,
              spacing: 1.1,
              strokeWidth: 2.2,
              baseline: "natural"
            },
            model_id: body.user_id // Return user_id instead of trainedModelId
          }
          
          return new Response(JSON.stringify(responseData), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          })
        } else {
          const errorText = await modalResponse.text()
          console.log(`Modal API returned error on attempt ${attempt}: ${modalResponse.status} - ${errorText}`)
          
          if (attempt === maxRetries) {
            // No fallback - return proper error
            return new Response(
              JSON.stringify({ error: `Modal API failed: ${modalResponse.status} - ${errorText}` }),
              { 
                status: 503, 
                headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
              }
            )
          }
        }
      } catch (modalError) {
        console.error(`Modal API attempt ${attempt} failed:`, modalError.message)
        
        if (attempt === maxRetries) {
          // No fallback - return error instead
          console.error('Modal API failed after all retries')
          return new Response(
            JSON.stringify({ error: `Failed to generate handwriting: Modal API unavailable after ${maxRetries} attempts. Last error: ${modalError.message}` }),
            { 
              status: 503, 
              headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
            }
          )
        }
      }
    }

    // If we get here, all retries failed
    return new Response(
      JSON.stringify({ error: 'Failed to generate handwriting: Modal API unavailable' }),
      { 
        status: 503, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )

  } catch (error) {
    console.error('Error in generate-handwriting function:', error)
    return new Response(
      JSON.stringify({ error: `Failed to generate handwriting: ${error.message}` }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )
  }
})