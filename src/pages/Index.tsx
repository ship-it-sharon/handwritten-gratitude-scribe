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
  const [currentStep, setCurrentStep] = useState<'welcome' | 'capture' | 'generate' | 'preview'>('welcome');
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
      if (data.sample_images) {
        // If user has samples, they can go directly to generate
        setCurrentStep('generate');
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
      setCurrentStep('generate');
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
            setCurrentStep('generate');
          }} 
          user={user}
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
