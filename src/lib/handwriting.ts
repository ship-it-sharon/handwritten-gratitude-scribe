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
  modelId?: string
): Promise<string | { status: string; [key: string]: any }> => {
  try {
    const { supabase } = await import('@/integrations/supabase/client');
    
    const requestBody: any = {
      text,
      samples: samples || [],
    };

    // If we have a model ID, use it for trained model generation
    if (modelId) {
      requestBody.model_id = modelId;
    } else if (style) {
      // Otherwise use style characteristics for initial preview
      requestBody.styleCharacteristics = {
        slant: style.slant,
        spacing: style.spacing,
        strokeWidth: style.strokeWidth,
        baseline: style.baseline,
        pressure: style.pressure,
      };
    }
    
    const { data, error } = await supabase.functions.invoke('generate-handwriting', {
      body: requestBody,
    });

    console.log('Edge function response:', { data, error });
    console.log('Received handwritingSvg length:', data?.handwritingSvg?.length || 'undefined');

    if (error) {
      throw new Error(`Failed to generate handwriting: ${error.message}`);
    }

    // Return the full response for training status handling
    return data.handwritingSvg || data;
  } catch (error) {
    console.error('Error generating handwriting:', error);
    throw error;
  }
};