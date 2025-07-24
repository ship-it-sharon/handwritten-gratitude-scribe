import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.7.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface TrainingRequest {
  samples: string[];
  user_id: string;
}

// Create fingerprint from samples for change detection
function createSampleFingerprint(samples: string[]): string {
  const sampleHashes = samples.map(sample => {
    // For base64 strings, use first and last 50 characters as fingerprint
    return sample.length < 100 ? sample : sample.substring(0, 50) + sample.substring(sample.length - 50);
  });
  
  // Sort to ensure consistent fingerprint regardless of order
  return sampleHashes.sort().join('|');
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  console.log('=== Train Handwriting Edge Function Called ===');
  console.log('Method:', req.method);
  console.log('URL:', req.url);

  try {
    console.log('Parsing request body...');
    const { samples, user_id }: TrainingRequest = await req.json();
    
    console.log('Request body parsed:', {
      samplesCount: samples?.length || 0,
      user_id: user_id?.substring(0, 8) + '...',
    });

    if (!samples || !Array.isArray(samples) || samples.length === 0) {
      console.error('No samples provided');
      return new Response(
        JSON.stringify({ error: 'No handwriting samples provided' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    if (!user_id) {
      console.error('No user_id provided');
      return new Response(
        JSON.stringify({ error: 'User ID is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Initialize Supabase client for database operations
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    console.log('Training handwriting model for user:', user_id);
    console.log('Using samples:', samples.length);
    
    // Create fingerprint from current samples
    const currentFingerprint = createSampleFingerprint(samples);

    // Check if user already has a model and if it's already trained or training
    const { data: existingModel, error: fetchError } = await supabase
      .from('user_style_models')
      .select('*')
      .eq('user_id', user_id)
      .maybeSingle();

    if (fetchError) {
      console.error('Error fetching existing model:', fetchError);
      return new Response(
        JSON.stringify({ error: 'Database error checking existing model' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('Existing model check:', {
      exists: !!existingModel,
      training_status: existingModel?.training_status,
      has_embedding_url: !!existingModel?.embedding_storage_url,
      model_id: existingModel?.model_id
    });

    // If model is already trained AND has embedding storage, return early
    if (existingModel?.training_status === 'completed' && existingModel.embedding_storage_url) {
      console.log('Model already trained with embeddings stored, skipping training');
      return new Response(
        JSON.stringify({ 
          message: 'Model already trained',
          model_id: existingModel.model_id,
          status: 'completed'
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // If model is currently training, return training status
    if (existingModel?.training_status === 'training') {
      console.log('Model already training, returning training status');
      return new Response(
        JSON.stringify({ 
          message: 'Model is currently training',
          model_id: existingModel.model_id,
          status: 'training'
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Force re-training if model exists but missing embedding storage
    if (existingModel?.training_status === 'completed' && !existingModel.embedding_storage_url) {
      console.log('Model completed but missing embedding storage, forcing re-training');
      // Continue to training process below
    }

    console.log('Proceeding with training - no valid existing model found');

    // Generate a model ID for this training session
    const model_id = `user_${user_id}_${Date.now()}`;
    console.log('Generated model ID:', model_id);

    // Update database to mark training as started
    const { error: updateError } = await supabase
      .from('user_style_models')
      .upsert({
        user_id: user_id,
        training_status: 'training',
        training_started_at: new Date().toISOString(),
        model_id: model_id,
        sample_fingerprint: currentFingerprint,
        sample_images: samples
      }, {
        onConflict: 'user_id'
      });

    if (updateError) {
      console.error('Error updating training status:', updateError);
      return new Response(
        JSON.stringify({ error: 'Failed to update training status' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('Training status updated, starting background training process...');

    // Start the actual training process in the background with proper completion handling
    const trainingPromise = startTrainingProcess(samples, user_id, model_id, supabase);
    
    // Use EdgeRuntime.waitUntil to ensure the training continues even after response is sent
    if (typeof EdgeRuntime !== 'undefined' && EdgeRuntime.waitUntil) {
      EdgeRuntime.waitUntil(trainingPromise);
    } else {
      // Fallback for environments without EdgeRuntime
      trainingPromise.catch(error => {
        console.error('Background training failed:', error);
      });
    }

    // Return immediate response indicating training has started
    return new Response(
      JSON.stringify({ 
        message: 'Training started successfully',
        model_id: model_id,
        status: 'training',
        estimated_time: '10-15 minutes'
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Error in train-handwriting function:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error', details: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});

async function startTrainingProcess(samples: string[], userId: string, modelId: string, supabase: any): Promise<{success: boolean, modelId?: string, embeddingUrl?: string}> {
  console.log('=== Starting Training Process ===');
  
  try {
    // Call Modal API for actual training
    console.log('Making request to Modal training API...');
    
    const modalApiUrl = 'https://ship-it-sharon--diffusionpen-handwriting-fastapi-app.modal.run';
    const trainEndpoint = `${modalApiUrl}/train_style`;
    console.log('Modal API URL:', trainEndpoint);
    
    const requestBody = {
      samples: samples.slice(0, 5), // Limit to 5 samples for training
      user_id: userId,
      model_id: modelId
    };
    
    console.log('Request payload:', {
      ...requestBody,
      samples: `${requestBody.samples.length} samples`
    });

    const response = await fetch(trainEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(300000) // 5 minute timeout for training (increased from 2 minutes)
    });

    console.log('Modal API response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Modal API error:', errorText);
      throw new Error(`Modal API failed with status ${response.status}: ${errorText}`);
    }

    const result = await response.json();
    console.log('Modal API training result:', result);

    // TEMPORARY: Handle Modal's current response format until Modal is updated
    // Once Modal provides model_url for download, we'll switch to Supabase storage
    let embeddingStorageUrl = null;
    
    console.log('📥 Modal training response received:', JSON.stringify(result, null, 2));
    
    // For now, construct the Modal path like before until Modal is updated to provide download URL
    if (result.model_id || result.embedding_id) {
      const modelRef = result.model_id || result.embedding_id || `user_${userId}_${Date.now()}`;
      embeddingStorageUrl = `/tmp/persistent_styles/style_emb_${modelRef.replace('user_', '')}.json`;
      console.log('✅ Constructed Modal embedding storage path:', embeddingStorageUrl);
    } else {
      console.log('⚠️ Could not construct embedding storage path - missing model_id/embedding_id');
      console.log('Available response fields:', Object.keys(result));
      // Don't fail - try to construct a path anyway
      embeddingStorageUrl = `/tmp/persistent_styles/style_emb_${userId}_${Date.now()}.json`;
      console.log('📝 Using fallback storage path:', embeddingStorageUrl);
    }

    // Update database with successful training completion and Modal storage path
    // TODO: Switch to Supabase storage URL once Modal provides download capability
    const { error: updateError } = await supabase
      .from('user_style_models')
      .update({
        training_status: 'completed',
        training_completed_at: new Date().toISOString(),
        style_model_path: result.model_path || `models/${modelId}`,
        embedding_storage_url: embeddingStorageUrl, // Temporarily store Modal path
        sample_fingerprint: createSampleFingerprint(samples)
      })
      .eq('user_id', userId);

    if (updateError) {
      console.error('Error updating training completion status:', updateError);
      throw updateError;
    }

    console.log('Training completed successfully for user:', userId);
    return { success: true, modelId, embeddingUrl: embeddingStorageUrl };

  } catch (error) {
    console.error('Training failed:', error);
    
    // Update database with failed status
    try {
      await supabase
        .from('user_style_models')
        .update({
          training_status: 'failed',
          training_completed_at: new Date().toISOString()
        })
        .eq('user_id', userId);
    } catch (dbError) {
      console.error('Failed to update failure status:', dbError);
    }

    return { success: false };
  }
}