import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PenTool, Heart, Send, Sparkles, ChevronRight, LogOut, RotateCcw } from "lucide-react";
import { HandwritingCapture } from "@/components/HandwritingCapture";
import { HandwritingPreview } from "@/components/HandwritingPreview";
import { NoteGenerator } from "@/components/NoteGenerator";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { TrainingProgressDisplay } from "@/components/TrainingProgressDisplay";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const { user, loading, signOut } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [currentStep, setCurrentStep] = useState<'welcome' | 'capture' | 'preview-samples' | 'processing' | 'generate' | 'preview'>('welcome');
  const [handwritingSamples, setHandwritingSamples] = useState<(string | HTMLCanvasElement)[]>([]);
  const [userStyleModel, setUserStyleModel] = useState<any>(null);
  const [previewText, setPreviewText] = useState("Best wishes: sincerely yours (always & forever)");

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
      .in('training_status', ['completed', 'failed', 'pending', 'training'])
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle();

    if (data && !error) {
      setUserStyleModel(data);
      if (data.sample_images && Array.isArray(data.sample_images)) {
        // Load the samples but don't change the current step if user is already on preview
        setHandwritingSamples(data.sample_images as string[]);
        // Only redirect to preview-samples if user is on welcome screen
        if (currentStep === 'welcome') {
          setCurrentStep('preview-samples');
        }
      }
    }
  };

  const startEmbeddingExtraction = async () => {
    console.log('🔥 startEmbeddingExtraction called, user:', user?.id);
    if (!user) {
      console.log('❌ No user found, aborting embedding extraction');
      return;
    }

    try {
      console.log('📦 Getting latest samples from database...');
      // Get the latest samples from database
      const { data: modelData, error: modelError } = await supabase
        .from('user_style_models')
        .select('sample_images, training_status, embedding_storage_url')
        .eq('user_id', user.id)
        .eq('training_status', 'completed')
        .order('created_at', { ascending: false })
        .limit(1)
        .maybeSingle();

      console.log('📦 Database query result:', { modelData, modelError });

      if (modelError || !modelData?.sample_images) {
        console.error('No samples found for embedding extraction:', modelError);
        toast({
          variant: "destructive",
          title: "No samples found",
          description: "Please complete handwriting capture first.",
        });
        return;
      }

      // Extract valid samples from database (handle both array and object formats)
      let validSamples: string[] = [];
      if (Array.isArray(modelData.sample_images)) {
        validSamples = modelData.sample_images.filter((img: any) => img && typeof img === 'string') as string[];
      } else if (typeof modelData.sample_images === 'object' && modelData.sample_images !== null) {
        // If stored as object, get all values and filter for strings
        validSamples = Object.values(modelData.sample_images).filter((img: any) => img && typeof img === 'string') as string[];
      }
      
      if (validSamples.length === 0) {
        console.error('No valid samples found for embedding extraction');
        toast({
          variant: "destructive",
          title: "No valid samples",
          description: "Please capture some handwriting samples first.",
        });
        return;
      }

      // Start embedding extraction process
      console.log('🚀 Starting embedding extraction process with samples:', validSamples.length);
      const { data: extractionData, error: extractionError } = await supabase.functions.invoke('train-handwriting', {
        body: {
          samples: validSamples.slice(0, 5), // Limit to 5 samples for processing
          user_id: user.id
        }
      });

      console.log('🚀 Embedding extraction function response:', { extractionData, extractionError });

      if (extractionError) {
        console.error('Embedding extraction failed to start:', extractionError);
        toast({
          variant: "destructive",
          title: "Processing failed to start",
          description: extractionError.message,
        });
      } else {
        console.log('✅ Embedding extraction started successfully:', extractionData);
        toast({
          title: "Processing started!",
          description: "Your handwriting style is being analyzed. This will take a few minutes.",
        });
      }
    } catch (error: any) {
      console.error('Error starting embedding extraction:', error);
      toast({
        variant: "destructive",
        title: "Error",
        description: error.message,
      });
    }
  };

  const handleLogout = async () => {
    console.log('🔴 Logout clicked');
    try {
      console.log('🔴 Calling signOut()');
      await signOut();
      console.log('🔴 SignOut completed, navigating to auth');
      navigate("/auth");
    } catch (error) {
      console.error('🔴 Logout error:', error);
    }
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
            // Move to preview step first so user can confirm samples
            setCurrentStep('preview-samples');
          }} 
          user={user}
          existingSamples={handwritingSamples}
        />;
      case 'preview-samples':
        return <SamplePreviewScreen 
          samples={handwritingSamples} 
          onContinue={() => {
            // Move directly to preview step with HandwritingPreview component
            setCurrentStep('preview');
          }}
          onRetake={() => setCurrentStep('capture')} 
        />;
      case 'processing':
        return <TrainingProgressDisplay 
          userId={user.id}
          onTrainingComplete={() => setCurrentStep('generate')} 
          onCancel={() => setCurrentStep('preview-samples')} 
        />;
      case 'generate':
        return <NoteGenerator onNext={() => setCurrentStep('preview')} handwritingSamples={handwritingSamples} />;
      case 'preview':
        return (
          <div className="min-h-screen flex items-center justify-center p-4">
            <div className="w-full max-w-4xl space-y-6">
              <div className="text-center">
                <h1 className="text-3xl font-elegant text-ink mb-2">Generate Your Handwriting Preview</h1>
                <p className="text-muted-foreground">
                  Click "Generate Preview" to see your text in your personalized handwriting style
                </p>
              </div>
              
              <HandwritingPreview 
                text={previewText}
                samples={handwritingSamples}
                onStyleChange={(style) => {
                  console.log('Style updated:', style);
                }}
              />
              
              <div className="flex justify-center">
                <Button variant="outline" onClick={() => setCurrentStep('preview-samples')}>
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Back to Samples
                </Button>
              </div>
            </div>
          </div>
        );
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
            {samples.filter(sample => sample && sample !== null).map((sample, index) => (
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
              <Sparkles className="w-4 h-4" />
              Generate Preview
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
