import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Brain, Sparkles, CheckCircle, AlertCircle } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

interface TrainingProgressDisplayProps {
  userId: string;
  onTrainingComplete: () => void;
  onCancel: () => void;
}

export const TrainingProgressDisplay = ({ userId, onTrainingComplete, onCancel }: TrainingProgressDisplayProps) => {
  const [progress, setProgress] = useState(10);
  const [status, setStatus] = useState<'training' | 'completed' | 'failed'>('training');
  const [currentStep, setCurrentStep] = useState("Initializing training...");
  const { toast } = useToast();

  const trainingSteps = [
    "Analyzing your handwriting samples...",
    "Processing stroke patterns...",
    "Training neural network...",
    "Optimizing model parameters...",
    "Finalizing your personal handwriting style...",
    "Training complete!"
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    let stepInterval: NodeJS.Timeout;
    let checkInterval: NodeJS.Timeout;

    // Simulate progress and steps
    if (status === 'training') {
      // Progress simulation
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 95) return prev; // Don't go to 100% until we confirm completion
          return prev + Math.random() * 3; // Increase by 0-3% each interval
        });
      }, 2000);

      // Step progression
      let currentStepIndex = 0;
      stepInterval = setInterval(() => {
        setCurrentStep(trainingSteps[currentStepIndex]);
        currentStepIndex = (currentStepIndex + 1) % (trainingSteps.length - 1); // Don't show last step until done
      }, 120000); // Change step every 2 minutes

      // Check training status
      checkInterval = setInterval(async () => {
        try {
          const { data, error } = await supabase
            .from('user_style_models')
            .select('training_status, training_completed_at')
            .eq('user_id', userId)
            .single();

          if (data?.training_status === 'completed') {
            setStatus('completed');
            setProgress(100);
            setCurrentStep(trainingSteps[trainingSteps.length - 1]);
            clearInterval(interval);
            clearInterval(stepInterval);
            clearInterval(checkInterval);
            
            toast({
              title: "Training Complete!",
              description: "Your handwriting model is ready to use.",
            });
            
            setTimeout(() => {
              onTrainingComplete();
            }, 2000);
          } else if (data?.training_status === 'failed') {
            setStatus('failed');
            clearInterval(interval);
            clearInterval(stepInterval);
            clearInterval(checkInterval);
            
            toast({
              variant: "destructive",
              title: "Training Failed",
              description: "There was an error training your model. Please try again.",
            });
          }
        } catch (error) {
          console.error('Error checking training status:', error);
        }
      }, 30000); // Check every 30 seconds
    }

    return () => {
      if (interval) clearInterval(interval);
      if (stepInterval) clearInterval(stepInterval);
      if (checkInterval) clearInterval(checkInterval);
    };
  }, [status, userId, onTrainingComplete, toast]);

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl p-8 shadow-elegant">
        <div className="text-center space-y-6">
          <div className="space-y-2">
            <div className="flex justify-center">
              {status === 'training' && (
                <Brain className="w-16 h-16 text-primary animate-pulse" />
              )}
              {status === 'completed' && (
                <CheckCircle className="w-16 h-16 text-green-500" />
              )}
              {status === 'failed' && (
                <AlertCircle className="w-16 h-16 text-red-500" />
              )}
            </div>
            
            <h1 className="text-3xl font-elegant text-ink">
              {status === 'training' && "Training Your Handwriting Model"}
              {status === 'completed' && "Training Complete!"}
              {status === 'failed' && "Training Failed"}
            </h1>
            
            <p className="text-muted-foreground">
              {status === 'training' && "Please wait while we create your personalized handwriting model"}
              {status === 'completed' && "Your handwriting model is ready to use"}
              {status === 'failed' && "Something went wrong during training"}
            </p>
          </div>

          {status === 'training' && (
            <>
              <div className="space-y-4">
                <Progress value={progress} className="w-full h-3" />
                <p className="text-sm text-muted-foreground">
                  {Math.round(progress)}% complete
                </p>
              </div>

              <div className="flex items-center justify-center gap-2 text-primary">
                <Sparkles className="w-5 h-5 animate-spin" />
                <span className="text-lg">{currentStep}</span>
              </div>

              <div className="text-sm text-muted-foreground space-y-1">
                <p>Estimated time remaining: {Math.max(1, Math.round((100 - progress) / 100 * 12))} minutes</p>
                <p>This process typically takes 10-15 minutes</p>
              </div>
            </>
          )}

          {status === 'completed' && (
            <div className="space-y-4">
              <Progress value={100} className="w-full h-3" />
              <p className="text-lg text-green-600 font-medium">
                Ready to generate your first handwritten note!
              </p>
            </div>
          )}

          {status === 'failed' && (
            <div className="space-y-4">
              <p className="text-red-600">
                Training failed. Please try again or contact support if the problem persists.
              </p>
              <Button onClick={onCancel} variant="outline">
                Go Back
              </Button>
            </div>
          )}

          {status === 'training' && (
            <div className="pt-4">
              <Button onClick={onCancel} variant="outline">
                Cancel Training
              </Button>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};