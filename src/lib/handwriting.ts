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
  style: HandwritingStyle, 
  samples?: string[]
): Promise<string> => {
  try {
    // First check if the API is healthy
    console.log('Checking API health...');
    const healthResponse = await fetch('https://ship-it-sharon--one-dm-handwriting-fastapi-app.modal.run/health');
    console.log('Health check response:', healthResponse.status);

    const response = await fetch('https://ship-it-sharon--one-dm-handwriting-fastapi-app.modal.run/generate_handwriting', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        samples: samples || [],
      }),
    });

    console.log('Generate handwriting response:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Response error:', errorText);
      throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
    }

    const data = await response.json();
    console.log('Response data:', data);
    
    if (data.error) {
      throw new Error(data.error);
    }

    return data.handwritingSvg;
  } catch (error) {
    console.error('Error generating handwriting:', error);
    throw error;
  }
};