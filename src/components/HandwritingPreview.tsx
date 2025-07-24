import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Wand2, Eye, Sparkles, RotateCcw } from "lucide-react";
import { 
  analyzeHandwritingSamples, 
  generateHandwritingStyle, 
  checkTrainingStatus,
  type HandwritingStyle 
} from "@/lib/handwriting";
import { supabase } from "@/integrations/supabase/client";

interface HandwritingPreviewProps {
  text: string;
  samples: (string | HTMLCanvasElement)[];
  onStyleChange?: (style: HandwritingStyle) => void;
}

export const HandwritingPreview = ({ text, samples, onStyleChange }: HandwritingPreviewProps) => {
  const [handwritingStyle, setHandwritingStyle] = useState<HandwritingStyle | null>(null);
  const [generatedSvg, setGeneratedSvg] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);
  const [processingStage, setProcessingStage] = useState<string>("");
  const [estimatedTime, setEstimatedTime] = useState<string>("");

  // Get current user ID
  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setUserId(user?.id || null);
    });
  }, []);

  const generatePreview = async () => {
    if (!text || samples.length === 0) {
      console.warn('No text or samples provided for generation');
      return;
    }

    setIsGenerating(true);
    setGeneratedSvg("");
    setProcessingStage("Checking your handwriting samples...");

    try {
      // Convert all samples to base64 strings
      const base64Samples: string[] = [];
      for (const sample of samples) {
        if (typeof sample === 'string') {
          base64Samples.push(sample);
        } else if (sample instanceof HTMLCanvasElement) {
          const base64 = sample.toDataURL('image/png');
          base64Samples.push(base64);
        }
      }

      console.log(`🔄 Processing ${samples.length} samples for generation`);

      if (userId) {
        // Smart training logic: check if we need to train or retrain
        console.log('🧠 Checking if training is needed...');
        setProcessingStage("Analyzing your handwriting style...");
        
        const trainingCheck = await checkTrainingStatus(userId, samples);
        console.log('📊 Training check result:', trainingCheck);

        if (trainingCheck.needsTraining) {
          console.log(`🚀 Training needed: ${trainingCheck.reason}`);
          setProcessingStage("Creating your personalized handwriting model...");
          setEstimatedTime("This may take 2-5 minutes");

          // Call training endpoint
          const { data: trainingData, error: trainingError } = await supabase.functions.invoke('train-handwriting', {
            body: {
              samples: base64Samples.slice(0, 5),
              user_id: userId
            }
          });

          console.log('🚀 Training function response:', { trainingData, trainingError });

          if (trainingError) {
            console.error('❌ Training error:', trainingError);
            throw new Error(`Training failed: ${trainingError.message}`);
          }

          // If training just started, wait a bit then check status
          if (trainingData?.status === 'training') {
            setProcessingStage("Training in progress...");
            setEstimatedTime(trainingData.estimated_time || "2-5 minutes");
            
            // For now, continue to generation which will handle training status
            console.log('📈 Training started, proceeding to generation...');
          }
        } else {
          console.log(`✅ Training not needed: ${trainingCheck.reason}`);
          if (trainingCheck.modelId) {
            console.log(`🎯 Using existing model: ${trainingCheck.modelId}`);
          }
        }
      }

      // Analyze style for fallback/enhancement
      setProcessingStage("Finalizing style characteristics...");
      const style = analyzeHandwritingSamples(samples);
      console.log('📊 Analyzed style:', style);
      setHandwritingStyle(style);
      onStyleChange?.(style);

      // Generate the handwriting
      setProcessingStage("Generating your handwritten text...");
      console.log('🎨 Generating handwriting with analyzed style...');

      const response = await generateHandwritingStyle(
        text, 
        style, 
        base64Samples,
        userId || undefined
      );

      console.log('🎭 Generation response type:', typeof response);

      if (!response) {
        throw new Error('No response from generation service');
      }

      // Handle different response types
      if (typeof response === 'object' && 'status' in response) {
        const statusResponse = response as any;
        if (statusResponse.status === 'training') {
          // Model still training
          setProcessingStage("Your handwriting model is still being created...");
          setEstimatedTime(statusResponse.estimated_time || "2-5 minutes");
          setGeneratedSvg(''); // Keep empty to show training state
          return;
        }
      } else if (typeof response === 'string') {
        // Successful SVG generation
        console.log('✅ Received SVG response, length:', response.length);
        setGeneratedSvg(response);
        setProcessingStage("");
        setEstimatedTime("");
      } else {
        console.warn('⚠️ Unexpected response format:', response);
        throw new Error('Unexpected response format from generation service');
      }

    } catch (error) {
      console.error('❌ Error generating handwriting preview:', error);
      setProcessingStage("");
      setEstimatedTime("");
      setGeneratedSvg("");
      // You could add error state UI here
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
            variant="elegant"
            size="sm"
            onClick={generatePreview}
            disabled={isGenerating || !text || samples.length === 0}
          >
            <Wand2 className="w-4 h-4" />
            {isGenerating ? 'Generating...' : 'Generate Preview'}
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
          {isGenerating ? (
            // Processing state
            <div className="text-center space-y-4 max-w-md">
              <div className="relative">
                <div className="w-16 h-16 mx-auto mb-4 relative">
                  <div className="absolute inset-0 border-4 border-primary/20 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                </div>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium text-ink text-lg">
                  {processingStage || "Processing..."}
                </h4>
                {estimatedTime && (
                  <div className="text-sm text-primary font-medium">
                    Estimated time: {estimatedTime}
                  </div>
                )}
                {processingStage.includes("Creating") && (
                  <p className="text-muted-foreground text-sm">
                    Our AI is learning your unique handwriting style to create personalized results
                  </p>
                )}
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
                Click "Generate Preview" to see your handwritten text
              </p>
            </div>
          ) : (
            <p className="text-muted-foreground">
              Enter text to preview handwritten version
            </p>
          )}
        </Card>
      </div>

      {generatedSvg && !isGenerating && (
        <div className="space-y-4">
          <div className="flex gap-2 justify-center">
            <Button
              variant="outline"
              size="sm"
              onClick={generatePreview}
              disabled={isGenerating}
              className="gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Generate Another
            </Button>
          </div>
          
          <div className="text-center space-y-2">
            <p className="text-xs text-muted-foreground">
              ✨ AI-generated handwriting matching your unique style
            </p>
            <p className="text-xs text-muted-foreground">
              💡 Love how it looks? Continue to create your thank you note!<br/>
              🎯 Want it even more accurate? Add more handwriting samples!
            </p>
          </div>
        </div>
      )}
    </Card>
  );
};