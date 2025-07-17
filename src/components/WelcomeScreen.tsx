import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PenTool, Heart, Send, Sparkles, ChevronRight, LogOut, RotateCcw } from "lucide-react";
import { User } from "@supabase/supabase-js";

interface WelcomeScreenProps {
  onNext: () => void;
  user?: User | null;
  onLogout?: () => void;
  userStyleModel?: any;
  onRegenerateFromSaved?: () => void;
}

export const WelcomeScreen = ({ onNext, user, onLogout, userStyleModel, onRegenerateFromSaved }: WelcomeScreenProps) => {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-4xl p-8 shadow-elegant bg-gradient-subtle">
        {/* User header */}
        {user && (
          <div className="flex justify-between items-center mb-6">
            <div className="text-sm text-muted-foreground">
              Welcome back, {user.email}
            </div>
            <Button variant="ghost" size="sm" onClick={onLogout}>
              <LogOut className="w-4 h-4 mr-2" />
              Sign Out
            </Button>
          </div>
        )}
        
        <div className="text-center space-y-8">
          {/* Hero Section */}
          <div className="space-y-4">
            <div className="flex justify-center mb-6">
              <div className="relative">
                <PenTool className="w-16 h-16 text-ink animate-float" />
                <Heart className="w-8 h-8 text-destructive absolute -top-2 -right-2 animate-pulse" />
              </div>
            </div>
            <h1 className="text-5xl font-elegant text-ink mb-4">
              Gratitude Scribe
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Transform your heartfelt gratitude into beautiful, personalized handwritten notes that touch hearts and create lasting memories.
            </p>
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-6 my-12">
            <FeatureCard
              icon={<PenTool className="w-8 h-8 text-ink" />}
              title="Capture Your Style"
              description="We learn your unique handwriting to create authentic-looking notes"
            />
            <FeatureCard
              icon={<Sparkles className="w-8 h-8 text-ink" />}
              title="AI-Powered Content"
              description="Generate heartfelt messages tailored to any occasion or recipient"
            />
            <FeatureCard
              icon={<Send className="w-8 h-8 text-ink" />}
              title="Real Mail Delivery"
              description="Your notes are printed and mailed directly to recipients"
            />
          </div>

          {/* CTA */}
          <div className="space-y-4">
            {userStyleModel?.sample_images ? (
              <div className="space-y-4">
                <Button 
                  variant="elegant" 
                  size="xl" 
                  onClick={onRegenerateFromSaved}
                  className="font-script text-xl px-12"
                >
                  <RotateCcw className="w-5 h-5 mr-2" />
                  Generate from Saved Samples
                </Button>
                <p className="text-sm text-muted-foreground">
                  Use your previously captured handwriting style
                </p>
                <div className="pt-2">
                  <Button 
                    variant="outline" 
                    size="lg" 
                    onClick={onNext}
                    className="font-script text-lg px-8"
                  >
                    Capture New Samples
                    <ChevronRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>
              </div>
            ) : (
              <>
                <Button 
                  variant="elegant" 
                  size="xl" 
                  onClick={onNext}
                  className="font-script text-xl px-12"
                >
                  Start Writing
                  <ChevronRight className="w-5 h-5 ml-2" />
                </Button>
                <p className="text-sm text-muted-foreground">
                  Begin by teaching us your handwriting style
                </p>
              </>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
};

const FeatureCard = ({ icon, title, description }: { 
  icon: React.ReactNode; 
  title: string; 
  description: string; 
}) => {
  return (
    <Card className="p-6 text-center space-y-3 shadow-soft hover:shadow-paper transition-all duration-300 bg-cream/50">
      <div className="flex justify-center">{icon}</div>
      <h3 className="font-elegant text-lg text-ink">{title}</h3>
      <p className="text-muted-foreground text-sm leading-relaxed">{description}</p>
    </Card>
  );
};