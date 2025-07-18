import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Wand2, Eye, Settings, Sparkles, RotateCcw } from "lucide-react";
import { analyzeHandwritingSamples, generateHandwritingStyle, type HandwritingStyle } from "@/lib/handwriting";
import { supabase } from "@/integrations/supabase/client";

interface HandwritingPreviewProps {
  text: string;
  samples: (string | HTMLCanvasElement)[];
  onStyleChange?: (style: HandwritingStyle) => void;
}

export const HandwritingPreview = ({ text, samples, onStyleChange }: HandwritingPreviewProps) => {
  const [handwritingStyle, setHandwritingStyle] = useState<HandwritingStyle | null>(null);
  const [generatedSvg, setGeneratedSvg] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<any>(null);
  const [showGenerateAnother, setShowGenerateAnother] = useState(false);

  // Get current user ID
  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setUserId(user?.id || null);
    });
  }, []);

  const analyzeStyle = async () => {
    setIsAnalyzing(true);
    try {
      const style = analyzeHandwritingSamples(samples);
      setHandwritingStyle(style);
      onStyleChange?.(style);
      
      // Auto-generate preview with analyzed style
      await generatePreview(style);
    } catch (error) {
      console.error('Error analyzing handwriting:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const generatePreview = async (style?: HandwritingStyle) => {
    if (!style && !handwritingStyle) {
      await analyzeStyle();
      return;
    }

    setIsGenerating(true);
    try {
      // Convert all samples to base64 strings
      const base64Samples: string[] = [];
      for (const sample of samples) {
        if (typeof sample === 'string') {
          base64Samples.push(sample);
        } else if (sample instanceof HTMLCanvasElement) {
          // Convert canvas to base64
          const base64 = sample.toDataURL('image/png');
          base64Samples.push(base64);
        }
      }
      
      console.log(`Converting ${samples.length} samples to base64, got ${base64Samples.length} strings`);
      
      const response = await generateHandwritingStyle(
        text, 
        style || handwritingStyle!, 
        base64Samples,
        userId || undefined // Pass user ID for authenticated users
      );
      
      // Handle different response types
      if (!response) {
        return;
      }
      
      console.log('Generate response:', response, 'Type:', typeof response);
      
      if (typeof response === 'object' && 'status' in response) {
        const statusResponse = response as any;
        if (statusResponse.status === 'training') {
          // Show training status instead of SVG
          setGeneratedSvg(''); // Clear any existing SVG
          setTrainingStatus(statusResponse);
        }
      } else if (typeof response === 'string') {
        // Regular SVG response
        console.log('Setting generatedSvg:', response.substring(0, 100) + '...');
        setGeneratedSvg(response);
        setTrainingStatus(null);
        setShowGenerateAnother(true);
      } else {
        console.warn('Unexpected response format:', response);
      }
    } catch (error) {
      console.error('Error generating handwriting preview:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Card className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h3 className="font-elegant text-lg text-ink">Handwriting Preview</h3>
          <p className="text-sm text-muted-foreground">
            AI-generated handwriting based on your samples
          </p>
        </div>
        
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={analyzeStyle}
            disabled={isAnalyzing || samples.length === 0}
          >
            <Settings className="w-4 h-4" />
            {isAnalyzing ? 'Analyzing...' : 'Analyze Style'}
          </Button>
          
          <Button
            variant="elegant"
            size="sm"
            onClick={() => generatePreview()}
            disabled={isGenerating || !text}
          >
            <Wand2 className="w-4 h-4" />
            {isGenerating ? 'Generating...' : 'Generate'}
          </Button>
        </div>
      </div>

      {handwritingStyle && (
        <div className="space-y-3">
          <h4 className="font-medium text-ink">Detected Style Characteristics</h4>
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              Slant: {handwritingStyle.slant > 0 ? 'Right' : handwritingStyle.slant < 0 ? 'Left' : 'Straight'} 
              ({handwritingStyle.slant.toFixed(1)}°)
            </Badge>
            <Badge variant="secondary">
              Spacing: {(handwritingStyle.spacing * 100).toFixed(0)}%
            </Badge>
            <Badge variant="secondary">
              Stroke: {handwritingStyle.strokeWidth.toFixed(1)}px
            </Badge>
            <Badge variant="secondary">
              Baseline: {handwritingStyle.baseline.replace('_', ' ')}
            </Badge>
          </div>
        </div>
      )}

      <Separator />

      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-muted-foreground">Live Preview</span>
        </div>
        
        <Card className="p-6 bg-paper min-h-40 flex items-center justify-center">
          {trainingStatus ? (
            // Training in progress state
            <div className="text-center space-y-4 max-w-md">
              <div className="relative">
                <div className="w-16 h-16 mx-auto mb-4 relative">
                  <div className="absolute inset-0 border-4 border-primary/20 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                </div>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium text-ink text-lg">Learning Your Handwriting Style</h4>
                <p className="text-muted-foreground">
                  Our AI is analyzing your samples to create a personalized handwriting model
                </p>
                <div className="text-sm text-primary font-medium">
                  Estimated time: {trainingStatus.estimated_time}
                </div>
                {trainingStatus.note && (
                  <p className="text-xs text-muted-foreground italic mt-3">
                    {trainingStatus.note}
                  </p>
                )}
              </div>
              
              <div className="pt-4">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => generatePreview()}
                  disabled={isGenerating}
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  {isGenerating ? 'Checking...' : 'Check Progress'}
                </Button>
              </div>
            </div>
          ) : generatedSvg ? (
            <div 
              className="max-w-full" 
              dangerouslySetInnerHTML={{ __html: generatedSvg }}
            />
          ) : text ? (
            <div className="text-center space-y-3">
              <Sparkles className="w-8 h-8 mx-auto text-muted-foreground" />
              <p className="text-muted-foreground">
                Click "Generate" to see your handwritten text
              </p>
            </div>
          ) : (
            <p className="text-muted-foreground">
              Enter text to preview handwritten version
            </p>
          )}
        </Card>
      </div>

      {generatedSvg && !trainingStatus && (
        <div className="space-y-4">
          <div className="flex gap-2 justify-center">
            <Button
              variant="outline"
              size="sm"
              onClick={() => generatePreview()}
              disabled={isGenerating}
              className="gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              {isGenerating ? 'Regenerating...' : 'Generate Another'}
            </Button>
          </div>
          
          <div className="text-center space-y-2">
            <p className="text-xs text-muted-foreground">
              ✨ AI-generated handwriting matching your unique style
            </p>
            <p className="text-xs text-muted-foreground">
              💡 Love how it looks? Continue to create your thank you note!<br/>
              🎯 Want it even more accurate? Collect a few more handwriting samples!
            </p>
          </div>
        </div>
      )}
    </Card>
  );
};