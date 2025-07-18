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

    // Initialize Supabase client
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    let trainedModelId = null;
    
    // Step 1: Handle model training and management
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
        trainedModelId = existingModel.style_model_path;
      } else {
        console.log("No existing model found, checking for training in progress...");
        
        // Check if training is already in progress
        const { data: trainingModel } = await supabase
          .from('user_style_models')
          .select('*')
          .eq('user_id', userId)
          .eq('training_status', 'training')
          .order('created_at', { ascending: false })
          .limit(1)
          .single();
          
        if (trainingModel) {
          console.log("Training already in progress for user");
          return new Response(JSON.stringify({
            status: 'training',
            message: 'Your handwriting style is being trained. This takes 10-15 minutes.',
            estimated_completion: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
            model_id: trainingModel.model_id
          }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }
        
        // Start new training
        const newModelId = `user_${userId}_${Date.now()}`;
        console.log("Starting training for new model:", newModelId);
        
        await supabase
          .from('user_style_models')
          .insert({
            user_id: userId,
            model_id: newModelId,
            training_status: 'training',
            sample_images: samples.slice(0, 5),
            training_started_at: new Date().toISOString()
          });
        
        // Start training process  
        const trainingResult = await startTrainingProcess(samples, userId, newModelId, supabase);
        
        if (trainingResult.success) {
          trainedModelId = trainingResult.modelId;
        } else {
          // Return training status for user feedback
          return new Response(JSON.stringify({
            status: 'training',
            message: 'Training your handwriting style. Please wait 10-15 minutes.',
            estimated_completion: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
            model_id: newModelId
          }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }
      }
    }
    
    // Step 2: Generate handwriting using trained model or fallback
    const maxRetries = 2
    const timeoutMs = 60000 // 1 minute for generation (training happens separately)
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Attempting generation API call (attempt ${attempt}/${maxRetries})...`)
        
        const modalUrl = 'https://ship-it-sharon--diffusionpen-handwriting-fastapi-app.modal.run/generate_handwriting'
        
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
        
        const requestPayload = {
          text: body.text,
          model_id: trainedModelId,
          samples: trainedModelId ? undefined : samples, // Only send samples if no trained model
          user_id: userId,
          styleCharacteristics: body.styleCharacteristics || {}
        }
        
        console.log('Making POST request to Modal API...')
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
          console.log('Modal handwritingSvg length:', modalData.handwritingSvg?.length || 'undefined')
          
          const responseData = {
            handwritingSvg: modalData.handwritingSvg,
            styleCharacteristics: modalData.styleCharacteristics || {
              slant: 0.15,
              spacing: 1.1,
              strokeWidth: 2.2,
              baseline: "natural"
            },
            model_id: trainedModelId
          }
          
          return new Response(JSON.stringify(responseData), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          })
        } else {
          const errorText = await modalResponse.text()
          console.log(`Modal API returned error on attempt ${attempt}: ${modalResponse.status} - ${errorText}`)
          
          if (attempt === maxRetries) {
            console.log('All Modal API attempts failed, falling back to local generation')
            break
          }
          
          await new Promise(resolve => setTimeout(resolve, 2000 * attempt))
        }
      } catch (modalError) {
        console.error(`Modal API attempt ${attempt} failed:`, modalError.message)
        
        if (attempt === maxRetries) {
          console.log('All Modal API attempts failed, falling back to local generation')
          break
        }
        
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

// Training function with proper status updates
async function startTrainingProcess(samples: string[], userId: string, modelId: string, supabase: any): Promise<{success: boolean, modelId?: string}> {
  try {
    console.log(`=== STARTING DIFFUSIONPEN TRAINING ===`);
    console.log(`Training model ${modelId} with ${samples.length} samples`);
    
    const trainingUrl = 'https://ship-it-sharon--diffusionpen-handwriting-fastapi-app.modal.run/train_style'
    
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 1800000) // 30 minutes
    
    const trainingPayload = {
      samples: samples.slice(0, 5),
      user_id: userId,
      model_id: modelId
    }
    
    console.log('Sending training request to Modal...')
    
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
      
      console.log(`Training completed successfully`)
      console.log(`Trained model ID: ${trainedModelId}`)
      
      // Update training record as completed
      await supabase
        .from('user_style_models')
        .update({
          training_status: 'completed',
          style_model_path: trainedModelId,
          training_completed_at: new Date().toISOString()
        })
        .eq('model_id', modelId);
      
      return { success: true, modelId: trainedModelId }
    } else {
      const errorText = await trainingResponse.text()
      console.log(`Training failed: ${trainingResponse.status} - ${errorText}`)
      
      await supabase
        .from('user_style_models')
        .update({ training_status: 'failed' })
        .eq('model_id', modelId);
        
      return { success: false }
    }
  } catch (error) {
    console.log(`Training failed with error: ${error}`)
    
    if (supabase) {
      await supabase
        .from('user_style_models')
        .update({ training_status: 'failed' })
        .eq('model_id', modelId);
    }
    
    return { success: false }
  }
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
    default: {
      // Fallback for other characters
      const [cx, cy] = transformPoint(x + 7, y - 8)
      return `<circle cx="${cx}" cy="${cy}" r="3" fill="#2c3e50"/>`
    }
  }
}