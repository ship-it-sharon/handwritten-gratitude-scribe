import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PenTool, Heart, Send, Sparkles, ChevronRight } from "lucide-react";
import { HandwritingCapture } from "@/components/HandwritingCapture";
import { NoteGenerator } from "@/components/NoteGenerator";
import { WelcomeScreen } from "@/components/WelcomeScreen";

const Index = () => {
  const [currentStep, setCurrentStep] = useState<'welcome' | 'capture' | 'generate' | 'preview'>('welcome');

  const renderStep = () => {
    switch (currentStep) {
      case 'welcome':
        return <WelcomeScreen onNext={() => setCurrentStep('capture')} />;
      case 'capture':
        return <HandwritingCapture onNext={() => setCurrentStep('generate')} />;
      case 'generate':
        return <NoteGenerator onNext={() => setCurrentStep('preview')} />;
      case 'preview':
        return <PreviewScreen onBack={() => setCurrentStep('generate')} />;
      default:
        return <WelcomeScreen onNext={() => setCurrentStep('capture')} />;
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
