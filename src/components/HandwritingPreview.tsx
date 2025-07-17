import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Wand2, Eye, Settings, Sparkles } from "lucide-react";
import { analyzeHandwritingSamples, generateHandwritingStyle, type HandwritingStyle } from "@/lib/handwriting";

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
      
      const svg = await generateHandwritingStyle(
        text, 
        style || handwritingStyle!, 
        base64Samples
      );
      setGeneratedSvg(svg);
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
          {generatedSvg ? (
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

      {generatedSvg && (
        <div className="space-y-4">
          <div className="flex justify-center">
            <Button
              variant="outline"
              size="sm"
              onClick={() => generatePreview()}
              disabled={isGenerating}
              className="gap-2"
            >
              <Sparkles className="w-4 h-4" />
              {isGenerating ? 'Regenerating...' : 'Generate Again'}
            </Button>
          </div>
          
          <div className="text-center">
            <p className="text-xs text-muted-foreground">
              ✨ AI-generated handwriting matching your unique style
            </p>
          </div>
        </div>
      )}
    </Card>
  );
};