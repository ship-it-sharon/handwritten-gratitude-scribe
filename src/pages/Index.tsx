import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PenTool, Heart, Send, Sparkles, ChevronRight, LogOut, RotateCcw } from "lucide-react";
import { HandwritingCapture } from "@/components/HandwritingCapture";
import { NoteGenerator } from "@/components/NoteGenerator";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const { user, loading, signOut } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [currentStep, setCurrentStep] = useState<'welcome' | 'capture' | 'preview-samples' | 'generate' | 'preview'>('welcome');
  const [handwritingSamples, setHandwritingSamples] = useState<(string | HTMLCanvasElement)[]>([]);
  const [userStyleModel, setUserStyleModel] = useState<any>(null);

  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
  }, [user, loading, navigate]);

  useEffect(() => {
    if (user) {
      loadUserStyleModel();
    }
  }, [user]);

  const loadUserStyleModel = async () => {
    if (!user) return;
    
    const { data, error } = await supabase
      .from('user_style_models')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle();

    if (data && !error) {
      setUserStyleModel(data);
      if (data.sample_images && Array.isArray(data.sample_images)) {
        // If user has samples, show them first before generating
        setHandwritingSamples(data.sample_images as string[]);
        setCurrentStep('preview-samples');
      }
    }
  };

  const saveUserSamples = async (samples: (string | HTMLCanvasElement)[]) => {
    if (!user) return;

    // Convert samples to base64 strings
    const sampleImages = samples.map(sample => {
      if (typeof sample === 'string') {
        return sample;
      } else {
        return sample.toDataURL();
      }
    });

    try {
      // Check if user already has a style model
      const { data: existingModel } = await supabase
        .from('user_style_models')
        .select('*')
        .eq('user_id', user.id)
        .maybeSingle();

      let data, error;

      if (existingModel) {
        // Update existing model with new samples
        ({ data, error } = await supabase
          .from('user_style_models')
          .update({
            sample_images: sampleImages,
            training_status: 'pending',
            training_started_at: null,
            training_completed_at: null
          })
          .eq('user_id', user.id)
          .select()
          .single());
      } else {
        // Create new model
        ({ data, error } = await supabase
          .from('user_style_models')
          .insert({
            user_id: user.id,
            model_id: `user_${user.id}_${Date.now()}`,
            sample_images: sampleImages,
            training_status: 'pending'
          })
          .select()
          .single());
      }

      if (error) {
        console.error('Database error:', error);
        toast({
          variant: "destructive",
          title: "Error saving samples",
          description: error.message,
        });
        return;
      }

      setUserStyleModel(data);
      
      // Start training process in background
      const { error: trainingError } = await supabase.functions.invoke('train-handwriting', {
        body: {
          samples: sampleImages.slice(0, 5), // Limit to 5 samples for training
          user_id: user.id
        }
      });

      if (trainingError) {
        console.error('Training failed to start:', trainingError);
        toast({
          variant: "destructive",
          title: "Training failed to start",
          description: trainingError.message,
        });
      } else {
        toast({
          title: "Training started!",
          description: "Your handwriting model is being trained. This will take 10-15 minutes.",
        });
      }
    } catch (error: any) {
      console.error('Error in saveUserSamples:', error);
      toast({
        variant: "destructive",
        title: "Error",
        description: error.message,
      });
    }
  };

  const handleLogout = async () => {
    await signOut();
    navigate("/auth");
  };

  const regenerateFromSavedSamples = () => {
    if (userStyleModel?.sample_images) {
      setHandwritingSamples(userStyleModel.sample_images);
      setCurrentStep('preview-samples');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-warm flex items-center justify-center">
        <div className="flex items-center gap-2 text-primary">
          <Sparkles className="w-6 h-6 animate-spin" />
          <span className="text-lg font-elegant">Loading...</span>
        </div>
      </div>
    );
  }

  if (!user) {
    return null; // Will redirect to auth
  }

  const renderStep = () => {
    switch (currentStep) {
      case 'welcome':
        return (
          <WelcomeScreen 
            onNext={() => setCurrentStep('capture')}
            user={user}
            onLogout={handleLogout}
            userStyleModel={userStyleModel}
            onRegenerateFromSaved={regenerateFromSavedSamples}
          />
        );
      case 'capture':
        return <HandwritingCapture 
          onNext={(samples) => {
            setHandwritingSamples(samples);
            saveUserSamples(samples);
            setCurrentStep('preview-samples');
          }} 
          user={user}
        />;
      case 'preview-samples':
        return <SamplePreviewScreen 
          samples={handwritingSamples} 
          onContinue={() => setCurrentStep('generate')} 
          onRetake={() => setCurrentStep('capture')} 
        />;
      case 'generate':
        return <NoteGenerator onNext={() => setCurrentStep('preview')} handwritingSamples={handwritingSamples} />;
      case 'preview':
        return <PreviewScreen onBack={() => setCurrentStep('generate')} />;
      default:
        return (
          <WelcomeScreen 
            onNext={() => setCurrentStep('capture')}
            user={user}
            onLogout={handleLogout}
            userStyleModel={userStyleModel}
            onRegenerateFromSaved={regenerateFromSavedSamples}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-warm">
      {renderStep()}
    </div>
  );
};

const SamplePreviewScreen = ({ 
  samples, 
  onContinue, 
  onRetake 
}: { 
  samples: (string | HTMLCanvasElement)[];
  onContinue: () => void;
  onRetake: () => void;
}) => {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-4xl p-8 shadow-elegant">
        <div className="space-y-6">
          <div className="text-center space-y-2">
            <h1 className="text-3xl font-elegant text-ink">Your Handwriting Samples</h1>
            <p className="text-muted-foreground">
              Review your samples before creating your note
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {samples.map((sample, index) => (
              <Card key={index} className="p-4 bg-paper">
                <div className="aspect-square w-full flex items-center justify-center">
                  {typeof sample === 'string' ? (
                    <img 
                      src={sample} 
                      alt={`Sample ${index + 1}`}
                      className="max-w-full max-h-full object-contain rounded-md"
                    />
                  ) : sample ? (
                    <div className="w-full h-full flex items-center justify-center">
                      <canvas 
                        ref={(canvas) => {
                          if (canvas && sample instanceof HTMLCanvasElement) {
                            const ctx = canvas.getContext('2d');
                            if (ctx) {
                              canvas.width = sample.width;
                              canvas.height = sample.height;
                              ctx.drawImage(sample, 0, 0);
                            }
                          }
                        }}
                        className="max-w-full max-h-full border rounded-md"
                      />
                    </div>
                  ) : (
                    <div className="text-muted-foreground text-sm">Empty sample</div>
                  )}
                </div>
                <p className="text-center text-sm text-muted-foreground mt-2">
                  Sample {index + 1}
                </p>
              </Card>
            ))}
          </div>
          
          <div className="flex gap-4 justify-center">
            <Button variant="outline" onClick={onRetake}>
              <RotateCcw className="w-4 h-4" />
              Retake Samples
            </Button>
            <Button variant="elegant" size="lg" onClick={onContinue}>
              Continue to Note Generation
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

const PreviewScreen = ({ onBack }: { onBack: () => void }) => {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl p-8 shadow-elegant">
        <div className="text-center space-y-6">
          <div className="space-y-2">
            <h1 className="text-3xl font-elegant text-ink">Preview Your Note</h1>
            <p className="text-muted-foreground">Review your handwritten thank you note before sending</p>
          </div>
          
          <div className="bg-paper p-8 rounded-lg border border-border shadow-soft">
            <div className="font-script text-xl text-ink leading-relaxed">
              Dear Sarah,
              <br /><br />
              Thank you so much for your thoughtful gift. Your kindness means the world to me, and I'm truly grateful for your friendship.
              <br /><br />
              With love and appreciation,
              <br />
              Alex
            </div>
          </div>
          
          <div className="flex gap-4 justify-center">
            <Button variant="outline" onClick={onBack}>
              <ChevronRight className="w-4 h-4 rotate-180" />
              Edit Note
            </Button>
            <Button variant="elegant" size="lg">
              <Send className="w-4 h-4" />
              Send Note
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Index;
