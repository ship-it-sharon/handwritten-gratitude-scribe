import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sparkles, ChevronRight, User, MapPin, Gift } from "lucide-react";
import { HandwritingPreview } from "./HandwritingPreview";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/hooks/useAuth";

interface NoteGeneratorProps {
  onNext: () => void;
  handwritingSamples?: (string | HTMLCanvasElement)[];
}

const occasions = [
  { value: "birthday", label: "Birthday Gift" },
  { value: "wedding", label: "Wedding Gift" },
  { value: "graduation", label: "Graduation Gift" },
  { value: "holiday", label: "Holiday Gift" },
  { value: "anniversary", label: "Anniversary Gift" },
  { value: "thank-you", label: "General Thank You" },
  { value: "support", label: "Support & Kindness" },
  { value: "custom", label: "Custom Occasion" }
];

export const NoteGenerator = ({ onNext, handwritingSamples = [] }: NoteGeneratorProps) => {
  const { user } = useAuth();
  const [formData, setFormData] = useState({
    recipientName: "",
    occasion: "",
    specificGift: "",
    personalMessage: "",
    recipientAddress: "",
    relationship: ""
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedNote, setGeneratedNote] = useState("");
  const [samples, setSamples] = useState<(string | HTMLCanvasElement)[]>(handwritingSamples);

  // Load samples from database if not provided
  useEffect(() => {
    const loadSamples = async () => {
      if (handwritingSamples.length > 0) {
        setSamples(handwritingSamples);
        return;
      }

      if (!user) return;

      console.log('🔍 Loading samples from database for user:', user.id);
      const { data, error } = await supabase
        .from('user_style_models')
        .select('sample_images, training_status, embedding_storage_url')
        .eq('user_id', user.id)
        .eq('training_status', 'completed')
        .order('created_at', { ascending: false })
        .limit(1)
        .maybeSingle();

      if (data && data.sample_images && Array.isArray(data.sample_images) && data.embedding_storage_url) {
        console.log('✅ Loaded samples from database with trained model:', data.sample_images.length);
        setSamples(data.sample_images as string[]);
      } else if (data && data.sample_images && Array.isArray(data.sample_images)) {
        console.log('⚠️ Found samples but no trained model - will need retraining');
        setSamples(data.sample_images as string[]);
      } else {
        console.log('❌ No samples found in database');
      }
    };

    loadSamples();
  }, [user, handwritingSamples]);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const generateNote = async () => {
    setIsGenerating(true);
    
    // Simulate AI generation
    setTimeout(() => {
      const note = `Dear ${formData.recipientName},

Thank you so much for ${formData.specificGift || 'your thoughtful gift'}. Your ${formData.relationship === 'friend' ? 'friendship' : 'kindness'} means the world to me, and I'm truly grateful for your generosity.

${formData.personalMessage ? `${formData.personalMessage}\n\n` : ''}With love and appreciation,
[Your Name]`;
      
      setGeneratedNote(note);
      setIsGenerating(false);
    }, 2000);
  };

  const canGenerate = formData.recipientName && formData.occasion;

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-4xl p-8 shadow-elegant">
        <div className="space-y-8">
          {/* Header */}
          <div className="text-center space-y-2">
            <h1 className="text-3xl font-elegant text-ink">Create Your Thank You Note</h1>
            <p className="text-muted-foreground">
              Tell us about your recipient and we'll help craft the perfect message
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Form Section */}
            <div className="space-y-6">
              <Card className="p-6 space-y-4 bg-cream/30">
                <div className="flex items-center gap-2 mb-4">
                  <User className="w-5 h-5 text-ink" />
                  <h3 className="font-elegant text-lg text-ink">Recipient Details</h3>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="recipientName">Recipient's Name *</Label>
                    <Input
                      id="recipientName"
                      placeholder="e.g., Sarah Johnson"
                      value={formData.recipientName}
                      onChange={(e) => handleInputChange('recipientName', e.target.value)}
                    />
                  </div>

                  <div>
                    <Label htmlFor="relationship">Your Relationship</Label>
                    <Select onValueChange={(value) => handleInputChange('relationship', value)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select relationship" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="friend">Friend</SelectItem>
                        <SelectItem value="family">Family Member</SelectItem>
                        <SelectItem value="colleague">Colleague</SelectItem>
                        <SelectItem value="neighbor">Neighbor</SelectItem>
                        <SelectItem value="acquaintance">Acquaintance</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="occasion">Occasion *</Label>
                    <Select onValueChange={(value) => handleInputChange('occasion', value)}>
                      <SelectTrigger>
                        <SelectValue placeholder="What's the occasion?" />
                      </SelectTrigger>
                      <SelectContent>
                        {occasions.map((occasion) => (
                          <SelectItem key={occasion.value} value={occasion.value}>
                            {occasion.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </Card>

              <Card className="p-6 space-y-4 bg-warm-accent/20">
                <div className="flex items-center gap-2 mb-4">
                  <Gift className="w-5 h-5 text-ink" />
                  <h3 className="font-elegant text-lg text-ink">Gift Details</h3>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="specificGift">What did they give you?</Label>
                    <Input
                      id="specificGift"
                      placeholder="e.g., beautiful flowers, homemade cookies"
                      value={formData.specificGift}
                      onChange={(e) => handleInputChange('specificGift', e.target.value)}
                    />
                  </div>

                  <div>
                    <Label htmlFor="personalMessage">Personal Message (Optional)</Label>
                    <Textarea
                      id="personalMessage"
                      placeholder="Add any specific details or memories you'd like to include..."
                      value={formData.personalMessage}
                      onChange={(e) => handleInputChange('personalMessage', e.target.value)}
                      rows={3}
                    />
                  </div>
                </div>
              </Card>

              <Card className="p-6 space-y-4">
                <div className="flex items-center gap-2 mb-4">
                  <MapPin className="w-5 h-5 text-ink" />
                  <h3 className="font-elegant text-lg text-ink">Mailing Address</h3>
                </div>

                <div>
                  <Label htmlFor="recipientAddress">Recipient's Mailing Address</Label>
                  <Textarea
                    id="recipientAddress"
                    placeholder="Full mailing address including postal code"
                    value={formData.recipientAddress}
                    onChange={(e) => handleInputChange('recipientAddress', e.target.value)}
                    rows={3}
                  />
                </div>
              </Card>

              <Button 
                variant="elegant" 
                size="lg" 
                onClick={generateNote}
                disabled={!canGenerate || isGenerating}
                className="w-full"
              >
                {isGenerating ? (
                  <>
                    <Sparkles className="w-4 h-4 animate-spin" />
                    Generating Your Note...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Generate Thank You Note
                  </>
                )}
              </Button>
            </div>

            {/* Preview Section */}
            <div className="space-y-6">
              {generatedNote ? (
                <HandwritingPreview 
                  text={generatedNote}
                  samples={samples}
                />
              ) : (
                <Card className="p-6 h-full bg-paper shadow-soft">
                  <h3 className="font-elegant text-lg text-ink mb-4">Preview</h3>
                  <div className="flex items-center justify-center h-80 text-muted-foreground">
                    <div className="text-center space-y-2">
                      <Sparkles className="w-12 h-12 mx-auto opacity-50" />
                      <p>Your handwritten note will appear here</p>
                    </div>
                  </div>
                </Card>
              )}
              
              {generatedNote && (
                <Button 
                  variant="elegant" 
                  size="lg" 
                  onClick={onNext}
                  className="w-full"
                >
                  Continue to Final Preview
                  <ChevronRight className="w-4 h-4" />
                </Button>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};