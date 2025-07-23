export interface HandwritingStyle {
  slant: number; // -10 to 10 (negative = left slant, positive = right slant)
  spacing: number; // 0.8 to 1.5 (letter spacing multiplier)
  strokeWidth: number; // 1 to 4 (stroke thickness)
  baseline: 'straight' | 'slightly_wavy' | 'irregular';
  pressure: number; // 0.5 to 1.5 (stroke pressure variation)
}

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

    // If we have a user ID, try to use their extracted style embeddings
    if (userId) {
      console.log('🎯 Using user embeddings for generation:', userId);
      requestBody.user_id = userId;
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