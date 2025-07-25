import { supabase } from "@/integrations/supabase/client";

export interface HandwritingStyle {
  slant: number; // -10 to 10 (negative = left slant, positive = right slant)
  spacing: number; // 0.8 to 1.5 (letter spacing multiplier)
  strokeWidth: number; // 1 to 4 (stroke thickness)
  baseline: 'straight' | 'slightly_wavy' | 'irregular';
  pressure: number; // 0.5 to 1.5 (stroke pressure variation)
}

// Create a fingerprint for samples to detect changes
export const createSampleFingerprint = (samples: (string | HTMLCanvasElement)[]): string => {
  const sampleHashes = samples.map(sample => {
    if (typeof sample === 'string') {
      // For base64 strings, use first and last 50 characters as fingerprint
      return sample.length < 100 ? sample : sample.substring(0, 50) + sample.substring(sample.length - 50);
    } else if (sample instanceof HTMLCanvasElement) {
      // For canvas, convert to base64 and then fingerprint
      const base64 = sample.toDataURL('image/png');
      return base64.length < 100 ? base64 : base64.substring(0, 50) + base64.substring(base64.length - 50);
    }
    return '';
  });
  
  // Sort to ensure consistent fingerprint regardless of order
  return sampleHashes.sort().join('|');
};

// Check if training is needed based on samples
export const checkTrainingStatus = async (userId: string, samples: (string | HTMLCanvasElement)[]) => {
  const currentFingerprint = createSampleFingerprint(samples);
  console.log('🔍 Sample fingerprint:', currentFingerprint);
  
  // Check if we have existing embeddings
  const { data: modelData, error } = await supabase
    .from('user_style_models')
    .select('training_status, model_id, sample_fingerprint, embedding_storage_url')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(1)
    .maybeSingle();
    
  if (error) {
    console.error('❌ Error checking training status:', error);
    return { needsTraining: true, reason: 'error_checking_status' };
  }
  
  if (!modelData) {
    console.log('📝 No existing model found - training needed');
    return { needsTraining: true, reason: 'no_existing_model' };
  }
  
  console.log('🔍 Latest model:', {
    model_id: modelData.model_id,
    training_status: modelData.training_status,
    sample_fingerprint: modelData.sample_fingerprint,
    embedding_storage_url: modelData.embedding_storage_url ? 'present' : 'missing'
  });
  
  if (modelData.training_status === 'failed') {
    console.log('❌ Previous training failed - retraining needed');
    return { needsTraining: true, reason: 'previous_training_failed' };
  }
  
  if (modelData.training_status === 'pending' || modelData.training_status === 'training') {
    console.log('⏳ Training in progress');
    return { needsTraining: false, reason: 'training_in_progress', modelId: modelData.model_id };
  }
  
  if (modelData.training_status === 'completed') {
    // Check if we have both fingerprint AND embedding storage URL
    const hasFingerprint = !!modelData.sample_fingerprint;
    const hasEmbeddingStorage = !!modelData.embedding_storage_url;
    const samplesChanged = modelData.sample_fingerprint !== currentFingerprint;
    
    console.log('🔍 Model validation:', {
      hasFingerprint,
      hasEmbeddingStorage,
      samplesChanged,
      current_fingerprint: currentFingerprint,
      stored_fingerprint: modelData.sample_fingerprint
    });
    
    // Skip URL accessibility test - it's causing false negatives
    // The URL test is unreliable because of CORS and auth headers
    console.log('✅ Skipping URL accessibility test to prevent false negatives');
    
    if (!hasFingerprint || !hasEmbeddingStorage || samplesChanged) {
      const reason = !hasFingerprint ? 'no_fingerprint_stored' : 
                     !hasEmbeddingStorage ? 'no_embedding_storage' :
                     'samples_changed';
      console.log(`📝 Training needed: ${reason}`);
      return { needsTraining: true, reason };
    }
    
    console.log('✅ Model ready and valid');
    return { needsTraining: false, reason: 'model_ready', modelId: modelData.model_id };
  }
  
  console.log('🔄 Unknown status - defaulting to retrain');
  return { needsTraining: true, reason: 'unknown_status' };
};

// Poll training status until completion
export const waitForTrainingCompletion = async (
  userId: string, 
  modelId: string,
  onProgress?: (attempt: number, maxAttempts: number, timeRemaining: string) => void
): Promise<boolean> => {
  const maxAttempts = 80; // 80 attempts * 15 seconds = 20 minutes max (increased buffer)
  const pollInterval = 15000; // 15 seconds between polls
  
  console.log(`🔄 Starting to poll training status for model: ${modelId}`);
  
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      console.log(`📊 Polling attempt ${attempt}/${maxAttempts}`);
      
      const { data: modelData, error } = await supabase
        .from('user_style_models')
        .select('training_status, embedding_storage_url')
        .eq('user_id', userId)
        .eq('model_id', modelId)
        .maybeSingle();
      
      if (error) {
        console.error(`❌ Error polling training status (attempt ${attempt}):`, error);
        continue;
      }
      
      if (!modelData) {
        console.error(`❌ Model not found (attempt ${attempt})`);
        continue;
      }
      
      console.log(`📈 Training status (attempt ${attempt}):`, {
        training_status: modelData.training_status,
        has_embedding_url: !!modelData.embedding_storage_url
      });
      
      if (modelData.training_status === 'completed' && modelData.embedding_storage_url) {
        console.log('✅ Training completed successfully!');
        return true;
      }
      
      if (modelData.training_status === 'failed') {
        console.error('❌ Training failed');
        return false;
      }
      
      // If still training, provide progress update and wait before next poll
      if (attempt < maxAttempts) {
        const minutesElapsed = Math.round((attempt * pollInterval) / 60000);
        const minutesRemaining = Math.max(0, 15 - minutesElapsed); // Estimate 15 min total
        const timeRemaining = minutesRemaining > 0 ? `~${minutesRemaining} minutes remaining` : "Almost done...";
        
        // Call progress callback if provided
        onProgress?.(attempt, maxAttempts, timeRemaining);
        
        console.log(`⏳ Still training, waiting ${pollInterval/1000} seconds before next check...`);
        console.log(`📊 Progress: ${attempt}/${maxAttempts} attempts, ${timeRemaining}`);
        await new Promise(resolve => setTimeout(resolve, pollInterval));
      }
      
    } catch (error) {
      console.error(`❌ Error during polling attempt ${attempt}:`, error);
      if (attempt === maxAttempts) {
        return false;
      }
    }
  }
  
  console.error('❌ Training timed out after maximum attempts');
  return false;
};

export const analyzeHandwritingSamples = (samples: (string | HTMLCanvasElement)[]): HandwritingStyle => {
  // This is a simplified analysis - in a real implementation, this would use
  // computer vision to analyze the actual handwriting characteristics
  
  console.log('Analyzing handwriting samples:', samples.length);
  
  // For now, return some randomized but realistic characteristics
  // In production, this would analyze actual stroke patterns, angles, etc.
  
  const randomInRange = (min: number, max: number) => min + Math.random() * (max - min);
  
  return {
    slant: randomInRange(-2, 5), // Most people have slight right slant
    spacing: randomInRange(0.9, 1.2), // Normal spacing with some variation
    strokeWidth: randomInRange(1.5, 2.5), // Medium stroke width
    baseline: Math.random() > 0.7 ? 'slightly_wavy' : 'straight',
    pressure: randomInRange(0.8, 1.2), // Moderate pressure variation
  };
};

export const generateHandwritingStyle = async (
  text: string, 
  style: HandwritingStyle | null, 
  samples?: string[],
  userId?: string
): Promise<string | { status: string; [key: string]: any }> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');
    
    console.log('🎨 generateHandwritingStyle called with:', { 
      textLength: text.length, 
      samplesCount: samples?.length || 0,
      hasUserId: !!userId,
      hasStyle: !!style 
    });

    const requestBody: any = {
      text,
      samples: samples || [],
    };

    // If we have a user ID, let Modal API fetch the embedding directly
    if (userId) {
      console.log('🎯 Using user ID for generation (Modal will fetch embedding):', userId);
      requestBody.user_id = userId;
      
      // No need to fetch model_id - Modal will handle this internally
      console.log('✅ Sending user_id to Modal API for dynamic embedding fetch');
    } else if (style) {
      // Otherwise use style characteristics for initial preview
      console.log('🎨 Using style characteristics:', style);
      requestBody.styleCharacteristics = {
        slant: style.slant,
        spacing: style.spacing,
        strokeWidth: style.strokeWidth,
        baseline: style.baseline,
        pressure: style.pressure,
      };
    }

    console.log('📤 Sending request to generate-handwriting edge function:', { 
      ...requestBody, 
      samples: `${requestBody.samples.length} samples` 
    });
    
    const { data, error } = await supabase.functions.invoke('generate-handwriting', {
      body: requestBody,
    });

    console.log('📥 Generate handwriting response:', { data, error });

    if (error) {
      console.error('❌ Generation error:', error);
      throw new Error(`Failed to generate handwriting: ${error.message}`);
    }

    // Handle different response types from the new backend
    if (data) {
      if (typeof data === 'string') {
        // Direct SVG response
        console.log('✅ Received SVG response, length:', data.length);
        return data;
      } else if (data.handwritingSvg) {
        // SVG in response object
        console.log('✅ Received SVG in response object, length:', data.handwritingSvg.length);
        return data.handwritingSvg;
      } else if (data.status) {
        // Status response (e.g., embeddings still processing)
        console.log('📊 Received status response:', data);
        return data;
      } else {
        // Unknown response format
        console.warn('⚠️ Unknown response format:', data);
        return data;
      }
    }

    throw new Error('No response data received from generation service');
  } catch (error) {
    console.error('❌ Error generating handwriting:', error);
    throw error;
  }
};