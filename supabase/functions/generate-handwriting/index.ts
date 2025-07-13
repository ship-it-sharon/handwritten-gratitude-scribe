import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface HandwritingRequest {
  text: string;
  styleCharacteristics?: {
    slant?: number;
    spacing?: number;
    strokeWidth?: number;
    baseline?: string;
  };
  samples?: string[];
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { text, samples = [] }: HandwritingRequest = await req.json();
    
    if (!text || text.trim() === '') {
      return new Response(
        JSON.stringify({ error: 'Text is required' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    console.log(`Generating handwriting for: "${text}" with ${samples.length} samples`);

    // Simple handwriting-style SVG generation
    const words = text.split(' ');
    let currentX = 50;
    let currentY = 100;
    const lineHeight = 40;
    const maxWidth = 700;
    
    let paths = '';
    let currentLineWidth = 0;
    
    words.forEach((word, wordIndex) => {
      const wordWidth = word.length * 15 + 20; // Estimate word width
      
      // Check if we need a new line
      if (currentLineWidth + wordWidth > maxWidth && wordIndex > 0) {
        currentX = 50;
        currentY += lineHeight;
        currentLineWidth = 0;
      }
      
      // Generate path for each letter in the word
      for (let i = 0; i < word.length; i++) {
        const letter = word[i];
        const letterPath = generateLetterPath(letter, currentX, currentY);
        paths += letterPath;
        currentX += 15 + Math.random() * 5; // Variable letter spacing
      }
      
      currentX += 20; // Space between words
      currentLineWidth += wordWidth;
    });

    const svgHeight = Math.max(200, currentY + 50);
    
    const handwritingSvg = `<svg width="800" height="${svgHeight}" viewBox="0 0 800 ${svgHeight}" xmlns="http://www.w3.org/2000/svg">
      <rect width="100%" height="100%" fill="white"/>
      <g stroke="#2d3748" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
        ${paths}
      </g>
    </svg>`;

    return new Response(
      JSON.stringify({
        handwritingSvg,
        styleCharacteristics: {
          slant: 0.1,
          spacing: 1.0,
          strokeWidth: 2.0,
          baseline: "slightly_wavy"
        }
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error('Error in generate-handwriting function:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});

function generateLetterPath(letter: string, x: number, y: number): string {
  const variation = () => (Math.random() - 0.5) * 3; // Small random variations
  
  switch (letter.toLowerCase()) {
    case 'a':
      return `<path d="M ${x + variation()} ${y} Q ${x + 7 + variation()} ${y - 15 + variation()} ${x + 14 + variation()} ${y} M ${x + 4 + variation()} ${y - 7 + variation()} L ${x + 10 + variation()} ${y - 7 + variation()}"/>`;
    case 'b':
      return `<path d="M ${x + variation()} ${y - 20 + variation()} L ${x + variation()} ${y} M ${x + variation()} ${y - 10 + variation()} Q ${x + 8 + variation()} ${y - 15 + variation()} ${x + 8 + variation()} ${y - 10 + variation()} Q ${x + 8 + variation()} ${y - 5 + variation()} ${x + variation()} ${y + variation()}"/>`;
    case 'c':
      return `<path d="M ${x + 12 + variation()} ${y - 12 + variation()} Q ${x + variation()} ${y - 12 + variation()} ${x + variation()} ${y - 6 + variation()} Q ${x + variation()} ${y + variation()} ${x + 12 + variation()} ${y + variation()}"/>`;
    case 'd':
      return `<path d="M ${x + 12 + variation()} ${y - 20 + variation()} L ${x + 12 + variation()} ${y} M ${x + 12 + variation()} ${y - 12 + variation()} Q ${x + variation()} ${y - 12 + variation()} ${x + variation()} ${y - 6 + variation()} Q ${x + variation()} ${y + variation()} ${x + 12 + variation()} ${y + variation()}"/>`;
    case 'e':
      return `<path d="M ${x + variation()} ${y - 6 + variation()} L ${x + 12 + variation()} ${y - 6 + variation()} Q ${x + 12 + variation()} ${y - 12 + variation()} ${x + 6 + variation()} ${y - 12 + variation()} Q ${x + variation()} ${y - 12 + variation()} ${x + variation()} ${y - 6 + variation()} Q ${x + variation()} ${y + variation()} ${x + 12 + variation()} ${y + variation()}"/>`;
    case 'f':
      return `<path d="M ${x + 12 + variation()} ${y - 20 + variation()} Q ${x + 8 + variation()} ${y - 20 + variation()} ${x + 6 + variation()} ${y - 15 + variation()} L ${x + 6 + variation()} ${y + 5 + variation()} M ${x + 2 + variation()} ${y - 12 + variation()} L ${x + 10 + variation()} ${y - 12 + variation()}"/>`;
    case 'g':
      return `<path d="M ${x + 12 + variation()} ${y - 12 + variation()} Q ${x + variation()} ${y - 12 + variation()} ${x + variation()} ${y - 6 + variation()} Q ${x + variation()} ${y + variation()} ${x + 12 + variation()} ${y + variation()} L ${x + 12 + variation()} ${y + 8 + variation()} Q ${x + 12 + variation()} ${y + 12 + variation()} ${x + 6 + variation()} ${y + 12 + variation()}"/>`;
    case 'h':
      return `<path d="M ${x + variation()} ${y - 20 + variation()} L ${x + variation()} ${y} M ${x + variation()} ${y - 10 + variation()} Q ${x + 6 + variation()} ${y - 12 + variation()} ${x + 12 + variation()} ${y - 12 + variation()} L ${x + 12 + variation()} ${y}"/>`;
    case 'i':
      return `<path d="M ${x + 6 + variation()} ${y - 12 + variation()} L ${x + 6 + variation()} ${y} M ${x + 6 + variation()} ${y - 18 + variation()} L ${x + 6 + variation()} ${y - 16 + variation()}"/>`;
    case 'j':
      return `<path d="M ${x + 8 + variation()} ${y - 12 + variation()} L ${x + 8 + variation()} ${y + 8 + variation()} Q ${x + 8 + variation()} ${y + 12 + variation()} ${x + 2 + variation()} ${y + 12 + variation()} M ${x + 8 + variation()} ${y - 18 + variation()} L ${x + 8 + variation()} ${y - 16 + variation()}"/>`;
    case 'k':
      return `<path d="M ${x + variation()} ${y - 20 + variation()} L ${x + variation()} ${y} M ${x + variation()} ${y - 6 + variation()} L ${x + 12 + variation()} ${y - 12 + variation()} M ${x + 6 + variation()} ${y - 8 + variation()} L ${x + 12 + variation()} ${y}"/>`;
    case 'l':
      return `<path d="M ${x + 6 + variation()} ${y - 20 + variation()} L ${x + 6 + variation()} ${y}"/>`;
    case 'm':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + variation()} ${y} M ${x + variation()} ${y - 10 + variation()} Q ${x + 4 + variation()} ${y - 12 + variation()} ${x + 8 + variation()} ${y - 12 + variation()} L ${x + 8 + variation()} ${y} M ${x + 8 + variation()} ${y - 10 + variation()} Q ${x + 12 + variation()} ${y - 12 + variation()} ${x + 16 + variation()} ${y - 12 + variation()} L ${x + 16 + variation()} ${y}"/>`;
    case 'n':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + variation()} ${y} M ${x + variation()} ${y - 10 + variation()} Q ${x + 6 + variation()} ${y - 12 + variation()} ${x + 12 + variation()} ${y - 12 + variation()} L ${x + 12 + variation()} ${y}"/>`;
    case 'o':
      return `<path d="M ${x + 6 + variation()} ${y - 12 + variation()} Q ${x + variation()} ${y - 12 + variation()} ${x + variation()} ${y - 6 + variation()} Q ${x + variation()} ${y + variation()} ${x + 6 + variation()} ${y + variation()} Q ${x + 12 + variation()} ${y + variation()} ${x + 12 + variation()} ${y - 6 + variation()} Q ${x + 12 + variation()} ${y - 12 + variation()} ${x + 6 + variation()} ${y - 12 + variation()}"/>`;
    case 'p':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + variation()} ${y + 8 + variation()} M ${x + variation()} ${y - 12 + variation()} Q ${x + 8 + variation()} ${y - 12 + variation()} ${x + 8 + variation()} ${y - 6 + variation()} Q ${x + 8 + variation()} ${y + variation()} ${x + variation()} ${y + variation()}"/>`;
    case 'q':
      return `<path d="M ${x + 12 + variation()} ${y - 12 + variation()} Q ${x + variation()} ${y - 12 + variation()} ${x + variation()} ${y - 6 + variation()} Q ${x + variation()} ${y + variation()} ${x + 12 + variation()} ${y + variation()} L ${x + 12 + variation()} ${y + 8 + variation()}"/>`;
    case 'r':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + variation()} ${y} M ${x + variation()} ${y - 8 + variation()} Q ${x + 6 + variation()} ${y - 12 + variation()} ${x + 10 + variation()} ${y - 10 + variation()}"/>`;
    case 's':
      return `<path d="M ${x + 12 + variation()} ${y - 10 + variation()} Q ${x + 6 + variation()} ${y - 12 + variation()} ${x + variation()} ${y - 8 + variation()} Q ${x + 6 + variation()} ${y - 4 + variation()} ${x + 12 + variation()} ${y - 6 + variation()} Q ${x + 6 + variation()} ${y + variation()} ${x + variation()} ${y - 2 + variation()}"/>`;
    case 't':
      return `<path d="M ${x + 6 + variation()} ${y - 16 + variation()} L ${x + 6 + variation()} ${y - 2 + variation()} Q ${x + 6 + variation()} ${y + variation()} ${x + 10 + variation()} ${y + variation()} M ${x + 2 + variation()} ${y - 12 + variation()} L ${x + 10 + variation()} ${y - 12 + variation()}"/>`;
    case 'u':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + variation()} ${y - 3 + variation()} Q ${x + variation()} ${y + variation()} ${x + 6 + variation()} ${y + variation()} Q ${x + 12 + variation()} ${y + variation()} ${x + 12 + variation()} ${y - 3 + variation()} L ${x + 12 + variation()} ${y - 12 + variation()}"/>`;
    case 'v':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + 6 + variation()} ${y} L ${x + 12 + variation()} ${y - 12 + variation()}"/>`;
    case 'w':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + 3 + variation()} ${y} L ${x + 6 + variation()} ${y - 6 + variation()} L ${x + 9 + variation()} ${y} L ${x + 12 + variation()} ${y - 12 + variation()}"/>`;
    case 'x':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + 12 + variation()} ${y} M ${x + 12 + variation()} ${y - 12 + variation()} L ${x + variation()} ${y}"/>`;
    case 'y':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + 6 + variation()} ${y - 4 + variation()} L ${x + 12 + variation()} ${y - 12 + variation()} M ${x + 6 + variation()} ${y - 4 + variation()} L ${x + 2 + variation()} ${y + 8 + variation()}"/>`;
    case 'z':
      return `<path d="M ${x + variation()} ${y - 12 + variation()} L ${x + 12 + variation()} ${y - 12 + variation()} L ${x + variation()} ${y} L ${x + 12 + variation()} ${y}"/>`;
    case ' ':
      return '';
    default:
      return `<circle cx="${x + 6 + variation()}" cy="${y - 6 + variation()}" r="2"/>`;
  }
}