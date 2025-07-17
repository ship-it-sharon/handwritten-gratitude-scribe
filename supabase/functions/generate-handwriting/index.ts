import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface HandwritingRequest {
  text: string
  styleCharacteristics?: {
    slant?: number
    spacing?: number
    strokeWidth?: number
    baseline?: string
    pressure?: number
  }
  samples?: string[] // base64 encoded images
  userId?: string // User authentication
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

    // Extract samples and user info from request
    const samples = body.samples || [];
    const userId = body.userId;
    
    console.log(`Generating handwriting for: "${body.text}" with ${samples.length} reference samples`)
    console.log('User ID:', userId)
    console.log('Raw samples received:', JSON.stringify(body.samples?.slice(0, 2)?.map(s => s?.substring(0, 50)) || [], null, 2))
    console.log('Samples array check:', Array.isArray(body.samples), 'Length:', body.samples?.length || 0)

    // Initialize Supabase client
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    // Try to call Modal API with retries (Modal apps go idle and need warmup time)
    const maxRetries = 2  // Reduced retries to prevent total timeout
    const timeoutMs = 300000 // 5 minutes for high-quality model processing
    
    console.log('=== STARTING MODAL API ATTEMPTS ===')
    
    let modelId = null;
    
    // Step 1: Handle user-specific style training and model management
    if (userId && samples.length > 0) {
      console.log('=== CHECKING FOR EXISTING USER STYLE MODEL ===');
      
      // Check if user already has a trained model
      const { data: existingModel } = await supabase
        .from('user_style_models')
        .select('*')
        .eq('user_id', userId)
        .eq('training_status', 'completed')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();
      
      if (existingModel) {
        console.log("Found existing trained model:", existingModel.model_id);
        modelId = existingModel.model_id;
      } else {
        console.log("No existing model found, attempting immediate training...");
        
        // Try immediate training with shorter timeout
        const newModelId = `style_model_${userId}_${Date.now()}`;
        const trainingResult = await trainModelWithTimeout(samples, userId, 300000); // 5 minutes
        
        if (trainingResult.success) {
          console.log("Training completed successfully:", trainingResult.modelId);
          
          // Save the trained model
          await supabase
            .from('user_style_models')
            .insert({
              user_id: userId,
              model_id: newModelId,
              training_status: 'completed',
              sample_images: samples.slice(0, 5),
              style_model_path: trainingResult.modelId,
              training_started_at: new Date().toISOString(),
              training_completed_at: new Date().toISOString()
            });
          
          modelId = trainingResult.modelId;
        } else {
          console.log("Training failed or timed out, using fallback");
          // Continue to fallback generation below
        }
      }
    } else if (samples.length > 0) {
      // Anonymous user with samples - try immediate training
      console.log("=== ANONYMOUS USER TRAINING ===");
      console.log(`Training style encoder with ${samples.length} samples for anonymous user...`);
      
      const trainingResult = await trainModelWithTimeout(samples, null, 300000); // 5 minutes
      
      if (trainingResult.success) {
        console.log("Anonymous training completed:", trainingResult.modelId);
        modelId = trainingResult.modelId;
      } else {
        console.log("Anonymous training failed, using fallback");
        // Continue to fallback generation below
      }
    }
    
    // Step 2: Generate handwriting using trained model or fallback
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Attempting generation API call (attempt ${attempt}/${maxRetries})...`)
        
        const modalUrl = 'https://ship-it-sharon--diffusionpen-handwriting-fastapi-app.modal.run/generate_handwriting'
        console.log(`Modal URL: ${modalUrl}`)
        
        // Test basic connectivity first
        console.log('Testing basic connectivity...')
        try {
          const testResponse = await fetch(modalUrl, { 
            method: 'GET',
            signal: AbortSignal.timeout(5000)
          })
          console.log(`Basic connectivity test status: ${testResponse.status}`)
        } catch (connectError) {
          console.error('Connectivity test failed:', connectError.message)
        }
        
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
        
        console.log('Making POST request to Modal API...')
        const requestPayload = {
          text: body.text,
          model_id: modelId,
          styleCharacteristics: body.styleCharacteristics || {}
        }
        console.log('Request payload:', JSON.stringify(requestPayload, null, 2))
        
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
          console.log('Modal response data:', JSON.stringify(modalData, null, 2))
          console.log('Modal handwritingSvg length:', modalData.handwritingSvg?.length || 'undefined')
          
          // Check if Modal fell back to a generic style profile (indicates training failure)
          if (modalData.model_id && modalData.model_id.includes('style_profile_None_')) {
            console.log("⚠️ Modal training failed, fell back to generic style profile:", modalData.model_id);
            throw new Error("Modal training failed - fell back to generic style profile");
          }
          
          // Check if we got actual handwriting or a placeholder/error image
          if (modalData.handwritingSvg) {
            const svgContent = modalData.handwritingSvg
            const containsHandwriting = svgContent.includes('<path') || svgContent.includes('stroke') || svgContent.includes('text')
            const containsPen = svgContent.includes('pen') || svgContent.includes('marker')
            console.log('SVG contains handwriting paths:', containsHandwriting)
            console.log('SVG contains pen/marker:', containsPen)
            console.log('SVG preview (first 300 chars):', svgContent.substring(0, 300))
            
            if (containsPen && !containsHandwriting) {
              console.log('WARNING: Modal API returned pen/marker placeholder instead of handwriting')
            }
          }
          
          console.log('Modal styleCharacteristics:', modalData.styleCharacteristics)
          
          const responseData = {
            handwritingSvg: modalData.handwritingSvg,
            model_id: modelId,
            styleCharacteristics: modalData.styleCharacteristics || {
              slant: 0.15,
              spacing: 1.1,
              strokeWidth: 2.2,
              baseline: "natural"
            }
          }
          console.log('Final response data:', JSON.stringify(responseData, null, 2))
          
          return new Response(JSON.stringify(responseData), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          })
        } else {
          const errorText = await modalResponse.text()
          console.log(`Modal API returned error on attempt ${attempt}: ${modalResponse.status} - ${errorText}`)
          
          // If it's the last attempt, fall back to local generation
          if (attempt === maxRetries) {
            console.log('All Modal API attempts failed, falling back to local generation')
            break
          }
          
          // Wait before retrying (exponential backoff)
          await new Promise(resolve => setTimeout(resolve, 2000 * attempt))
        }
      } catch (modalError) {
        console.error(`Modal API attempt ${attempt} failed:`, {
          message: modalError.message,
          name: modalError.name,
          stack: modalError.stack,
          cause: modalError.cause
        })
        
        // Check if it's a timeout
        if (modalError.name === 'AbortError') {
          console.log('Modal API call timed out after 2 minutes')
        }
        
        // If it's the last attempt, fall back to local generation
        if (attempt === maxRetries) {
          console.log('All Modal API attempts failed, falling back to local generation')
          break
        }
        
        // Wait before retrying (exponential backoff)
        await new Promise(resolve => setTimeout(resolve, 2000 * attempt))
      }
    }

    // Fallback to local handwriting generation
    console.log("Using fallback handwriting generation")
    const handwritingSvg = generateHandwritingSVG(body.text, body.styleCharacteristics || {})
    
    return new Response(JSON.stringify({
      handwritingSvg,
      styleCharacteristics: {
        slant: body.styleCharacteristics?.slant || 0.1,
        spacing: body.styleCharacteristics?.spacing || 1.0,
        strokeWidth: body.styleCharacteristics?.strokeWidth || 2.0,
        baseline: body.styleCharacteristics?.baseline || "natural",
        pressure: body.styleCharacteristics?.pressure || 1.0
      }
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    })

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

// Training function with timeout for immediate results
async function trainModelWithTimeout(samples: string[], userId: string | null, timeoutMs: number): Promise<{success: boolean, modelId?: string}> {
  try {
    console.log(`=== STARTING IMMEDIATE TRAINING ===`);
    console.log(`Training with ${samples.length} samples, timeout: ${timeoutMs}ms`);
    
    const trainingUrl = 'https://ship-it-sharon--diffusionpen-handwriting-fastapi-app.modal.run/train_style'
    
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
    
    const trainingPayload = {
      samples: samples.slice(0, 5),
      user_id: userId
    }
    
    const trainingResponse = await fetch(trainingUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(trainingPayload),
      signal: controller.signal
    })

    clearTimeout(timeoutId)

    if (trainingResponse.ok) {
      const trainingData = await trainingResponse.json()
      const trainedModelId = trainingData.model_id
      
      console.log(`Immediate training completed successfully`)
      console.log(`Trained model ID: ${trainedModelId}`)
      
      return { success: true, modelId: trainedModelId }
    } else {
      const errorText = await trainingResponse.text()
      console.log(`Immediate training failed: ${trainingResponse.status} - ${errorText}`)
      return { success: false }
    }
  } catch (error) {
    console.log(`Immediate training failed with error: ${error}`)
    return { success: false }
  }
}

// Background training function to prevent timeouts
async function trainModelInBackground(supabase: any, samples: string[], userId: string, modelId: string) {
  console.log(`=== BACKGROUND TRAINING STARTED ===`);
  console.log(`Training style encoder with ${samples.length} samples for model ${modelId}...`);
  
  const maxRetries = 3;
  const timeoutMs = 600000; // 10 minutes for background training
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`Background training attempt ${attempt}/${maxRetries}...`)
      
      const trainingUrl = 'https://ship-it-sharon--diffusionpen-handwriting-fastapi-app.modal.run/train_style'
      
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
      
      const trainingPayload = {
        samples: samples.slice(0, 5),
        user_id: userId
      }
      
      const trainingResponse = await fetch(trainingUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingPayload),
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (trainingResponse.ok) {
        const trainingData = await trainingResponse.json()
        const trainedModelId = trainingData.model_id
        
        // Update training record as completed
        await supabase
          .from('user_style_models')
          .update({
            training_status: 'completed',
            style_model_path: trainedModelId,
            training_completed_at: new Date().toISOString()
          })
          .eq('model_id', modelId);
        
        console.log(`Background training completed successfully for model ${modelId}`)
        console.log(`Trained model ID: ${trainedModelId}`)
        return
      } else {
        const errorText = await trainingResponse.text()
        console.log(`Background training attempt ${attempt} failed: ${trainingResponse.status} - ${errorText}`)
      }
    } catch (error) {
      console.log(`Background training attempt ${attempt} failed with error: ${error}`)
    }
    
    // Exponential backoff
    if (attempt < maxRetries) {
      await new Promise(resolve => setTimeout(resolve, 5000 * attempt))
    }
  }
  
  // Mark as failed if all attempts failed
  await supabase
    .from('user_style_models')
    .update({
      training_status: 'failed'
    })
    .eq('model_id', modelId);
  
  console.log(`Background training failed for model ${modelId} after all attempts`)
}

function generateHandwritingSVG(text: string, styleParams: any): string {
  // Enhanced handwriting generation with more realistic letter forms
  const words = text.split(' ')
  let currentX = 50
  let currentY = 120
  const lineHeight = 60
  const maxWidth = 700
  
  let paths = []
  let currentLineWidth = 0
  let baselineVariation = 0
  
  // Style parameters with defaults
  const slant = styleParams.slant || 0.1
  const spacing = styleParams.spacing || 1.0
  const strokeWidth = styleParams.strokeWidth || 2.0
  
  for (let wordIdx = 0; wordIdx < words.length; wordIdx++) {
    const word = words[wordIdx]
    const wordWidth = word.length * 18 * spacing + 25
    
    // Check if we need a new line
    if (currentLineWidth + wordWidth > maxWidth && wordIdx > 0) {
      currentX = 50
      currentY += lineHeight
      currentLineWidth = 0
      baselineVariation = 0
    }
    
    // Generate each letter with realistic variations
    for (let charIdx = 0; charIdx < word.length; charIdx++) {
      const char = word[charIdx]
      if (/[a-zA-Z]/.test(char)) {
        // Add natural baseline variation
        baselineVariation += (Math.random() - 0.5) * 4
        baselineVariation *= 0.9 // Decay to prevent drift
        
        const charY = currentY + baselineVariation
        const letterPath = generateRealisticLetter(char, currentX, charY, { slant, spacing, strokeWidth }, charIdx)
        paths.push(letterPath)
      }
      
      // Variable letter spacing
      const spacingVar = 0.9 + Math.random() * 0.2
      currentX += 16 * spacing * spacingVar
    }
    
    // Add space between words
    currentX += 25 * spacing
    currentLineWidth += wordWidth
  }

  const svgHeight = Math.max(200, currentY + 80)
  
  return `<svg width="800" height="${svgHeight}" viewBox="0 0 800 ${svgHeight}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="white"/>
    <g stroke="#2c3e50" stroke-width="${strokeWidth}" fill="none" 
       stroke-linecap="round" stroke-linejoin="round" opacity="0.85">
      ${paths.join('')}
    </g>
  </svg>`
}

function generateRealisticLetter(char: string, x: number, y: number, style: any, position: number): string {
  // Add natural hand tremor
  const tremor = () => (Math.random() - 0.5) * 1.6
  const slant = style.slant
  
  // Apply slant transformation
  const transformPoint = (px: number, py: number) => {
    const slantedX = px + (py - y) * slant
    return [slantedX + tremor(), py + tremor()]
  }
  
  const lowerChar = char.toLowerCase()
  
  // Enhanced letter generation with more realistic curves
  switch (lowerChar) {
    case 'a': {
      const [p1x, p1y] = transformPoint(x + 2, y)
      const [p2x, p2y] = transformPoint(x + 8, y - 18)
      const [p3x, p3y] = transformPoint(x + 14, y)
      const [p4x, p4y] = transformPoint(x + 4, y - 8)
      const [p5x, p5y] = transformPoint(x + 12, y - 8)
      return `<path d="M ${p1x} ${p1y} Q ${p2x} ${p2y} ${p3x} ${p3y} M ${p4x} ${p4y} L ${p5x} ${p5y}"/>`
    }
    case 'b': {
      const [p1x, p1y] = transformPoint(x, y - 25)
      const [p2x, p2y] = transformPoint(x, y + 2)
      const [p3x, p3y] = transformPoint(x, y - 12)
      const [p4x, p4y] = transformPoint(x + 10, y - 16)
      const [p5x, p5y] = transformPoint(x + 10, y - 8)
      const [p6x, p6y] = transformPoint(x, y - 4)
      return `<path d="M ${p1x} ${p1y} L ${p2x} ${p2y} M ${p3x} ${p3y} Q ${p4x} ${p4y} ${p5x} ${p5y} Q ${p6x} ${p6y} ${p3x} ${p3y}"/>`
    }
    case 'e': {
      const [cx, cy] = transformPoint(x + 7, y - 8)
      const [p1x, p1y] = transformPoint(x, y - 8)
      const [p2x, p2y] = transformPoint(x + 12, y - 12)
      return `<circle cx="${cx}" cy="${cy}" r="7" fill="none"/><path d="M ${p1x} ${p1y} L ${p2x} ${p2y}"/>`
    }
    case 'h': {
      const [p1x, p1y] = transformPoint(x, y - 25)
      const [p2x, p2y] = transformPoint(x, y + 2)
      const [p3x, p3y] = transformPoint(x, y - 12)
      const [p4x, p4y] = transformPoint(x + 8, y - 15)
      const [p5x, p5y] = transformPoint(x + 14, y - 15)
      const [p6x, p6y] = transformPoint(x + 14, y + 2)
      return `<path d="M ${p1x} ${p1y} L ${p2x} ${p2y} M ${p3x} ${p3y} Q ${p4x} ${p4y} ${p5x} ${p5y} L ${p6x} ${p6y}"/>`
    }
    case 'l': {
      const [p1x, p1y] = transformPoint(x + 6, y - 25)
      const [p2x, p2y] = transformPoint(x + 6, y + 2)
      return `<path d="M ${p1x} ${p1y} L ${p2x} ${p2y}"/>`
    }
    case 'o': {
      const [cx, cy] = transformPoint(x + 7, y - 8)
      const rx = 7 + tremor()
      const ry = 8 + tremor()
      return `<ellipse cx="${cx}" cy="${cy}" rx="${Math.abs(rx)}" ry="${Math.abs(ry)}" fill="none"/>`
    }
    case 'r': {
      const [p1x, p1y] = transformPoint(x, y - 15)
      const [p2x, p2y] = transformPoint(x, y + 2)
      const [p3x, p3y] = transformPoint(x, y - 10)
      const [p4x, p4y] = transformPoint(x + 8, y - 15)
      const [p5x, p5y] = transformPoint(x + 12, y - 12)
      return `<path d="M ${p1x} ${p1y} L ${p2x} ${p2y} M ${p3x} ${p3y} Q ${p4x} ${p4y} ${p5x} ${p5y}"/>`
    }
    case 'w': {
      const [p1x, p1y] = transformPoint(x, y - 15)
      const [p2x, p2y] = transformPoint(x + 4, y + 2)
      const [p3x, p3y] = transformPoint(x + 8, y - 8)
      const [p4x, p4y] = transformPoint(x + 12, y + 2)
      const [p5x, p5y] = transformPoint(x + 16, y - 15)
      return `<path d="M ${p1x} ${p1y} L ${p2x} ${p2y} L ${p3x} ${p3y} L ${p4x} ${p4y} L ${p5x} ${p5y}"/>`
    }
    default: {
      // Fallback for other characters
      const [cx, cy] = transformPoint(x + 7, y - 8)
      return `<circle cx="${cx}" cy="${cy}" r="3" fill="#2c3e50"/>`
    }
  }
}