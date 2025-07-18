import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface TrainingRequest {
  samples: string[] // base64 encoded images
  user_id: string // User authentication
}

serve(async (req) => {
  console.log('=== Training edge function called ===')
  console.log('Method:', req.method)
  console.log('URL:', req.url)
  
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    console.log('Handling CORS preflight request')
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    console.log('Parsing training request body...')
    const body = await req.json() as TrainingRequest
    
    if (!body.samples || body.samples.length === 0) {
      return new Response(
        JSON.stringify({ error: 'Training samples are required' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    if (!body.user_id) {
      return new Response(
        JSON.stringify({ error: 'User ID is required for training' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      )
    }

    console.log(`Training request for user ${body.user_id} with ${body.samples.length} samples`)

    // Initialize Supabase client
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    // Check if user already has a completed model
    const { data: existingModel } = await supabase
      .from('user_style_models')
      .select('*')
      .eq('user_id', body.user_id)
      .eq('training_status', 'completed')
      .order('created_at', { ascending: false })
      .limit(1)
      .single();
    
    if (existingModel) {
      console.log("User already has a trained model:", existingModel.model_id);
      return new Response(JSON.stringify({
        status: 'already_trained',
        message: 'You already have a trained handwriting model.',
        model_id: existingModel.model_id
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Check if training is already in progress
    const { data: trainingModel } = await supabase
      .from('user_style_models')
      .select('*')
      .eq('user_id', body.user_id)
      .in('training_status', ['training', 'pending'])
      .order('created_at', { ascending: false })
      .limit(1)
      .single();
      
    if (trainingModel) {
      console.log("Training already in progress for user");
      return new Response(JSON.stringify({
        status: 'training_in_progress',
        message: 'Your handwriting style is already being trained. Please wait.',
        estimated_completion: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
        model_id: trainingModel.model_id
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Start new training
    const newModelId = `user_${body.user_id}_${Date.now()}`;
    console.log("Starting training for new model:", newModelId);
    
    // Insert training record
    const { error: insertError } = await supabase
      .from('user_style_models')
      .insert({
        user_id: body.user_id,
        model_id: newModelId,
        training_status: 'training',
        sample_images: body.samples.slice(0, 5),
        training_started_at: new Date().toISOString()
      });

    if (insertError) {
      console.error('Error inserting training record:', insertError);
      return new Response(JSON.stringify({
        error: 'Failed to start training process',
        details: insertError.message
      }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    // Start training process in background
    const trainingResult = await startTrainingProcess(body.samples, body.user_id, newModelId, supabase);
    
    if (trainingResult.success) {
      return new Response(JSON.stringify({
        status: 'training_complete',
        message: 'Your handwriting model has been trained successfully!',
        model_id: trainingResult.modelId
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    } else {
      return new Response(JSON.stringify({
        status: 'training_started',
        message: 'Training started. This will take 10-15 minutes.',
        estimated_completion: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
        model_id: newModelId
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

  } catch (error) {
    console.error('Error in train-handwriting function:', error)
    return new Response(
      JSON.stringify({ error: `Failed to process training request: ${error.message}` }),
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
    const timeoutId = setTimeout(() => controller.abort(), 900000) // 15 minutes instead of 30
    
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