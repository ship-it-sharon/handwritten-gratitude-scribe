import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { RotateCcw, Check, ChevronRight, Upload, Camera, PenTool, Image, Smartphone, Loader2, Brain, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { MobileUploadSidecar } from "./MobileUploadSidecar";

interface HandwritingCaptureProps {
  onNext: (samples: (string | HTMLCanvasElement)[]) => void;
}

const sampleTexts = [
  "The quick brown fox jumps over the lazy dog",
  "Thank you so much for your kindness",
  "With love and appreciation",
  "Sincerely yours",
  "Best wishes and warm regards"
];

export const HandwritingCapture = ({ onNext }: HandwritingCaptureProps) => {
  const [currentSample, setCurrentSample] = useState(0);
  const [captureMethod, setCaptureMethod] = useState<'draw' | 'upload' | 'mobile'>('draw');
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [mobileImages, setMobileImages] = useState<Map<number, string>>(new Map());
  const [completedSamples, setCompletedSamples] = useState<Set<number>>(new Set());
  const [isValidating, setIsValidating] = useState(false);
  const [sessionId] = useState(() => `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const progress = (completedSamples.size / sampleTexts.length) * 100;
  const currentSampleCompleted = completedSamples.has(currentSample);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    
    const context = canvas.getContext('2d');
    if (!context) return;
    
    context.scale(2, 2);
    context.lineCap = 'round';
    context.strokeStyle = 'hsl(215 85% 25%)';
    context.lineWidth = 2;
    
    contextRef.current = context;
  }, [currentSample]);

  // Reset states when switching methods or samples
  useEffect(() => {
    setHasDrawn(false);
    setUploadedImage(null);
    clearCanvas();
  }, [captureMethod, currentSample]);

  const startDrawing = (event: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !contextRef.current) return;

    setIsDrawing(true);
    setHasDrawn(true);

    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if ('touches' in event) {
      clientX = event.touches[0].clientX;
      clientY = event.touches[0].clientY;
    } else {
      clientX = event.clientX;
      clientY = event.clientY;
    }

    const x = clientX - rect.left;
    const y = clientY - rect.top;

    contextRef.current.beginPath();
    contextRef.current.moveTo(x, y);
  };

  const draw = (event: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !contextRef.current) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if ('touches' in event) {
      clientX = event.touches[0].clientX;
      clientY = event.touches[0].clientY;
    } else {
      clientX = event.clientX;
      clientY = event.clientY;
    }

    const x = clientX - rect.left;
    const y = clientY - rect.top;

    contextRef.current.lineTo(x, y);
    contextRef.current.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !contextRef.current) return;
    
    contextRef.current.clearRect(0, 0, canvas.width, canvas.height);
    setHasDrawn(false);
  };

  const validateHandwriting = async (imageData: string) => {
    setIsValidating(true);
    try {
      const response = await fetch(`https://lkqjlibxmsnjqaifipes.supabase.co/functions/v1/validate-handwriting`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxrcWpsaWJ4bXNuanFhaWZpcGVzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyOTgyNzQsImV4cCI6MjA2Nzg3NDI3NH0.mpltb2Pc2H2vQNAuYQntJv462kFvyHG6yxe5Yt-pdto`
        },
        body: JSON.stringify({
          imageData,
          expectedText: sampleTexts[currentSample]
        })
      });

      const validation = await response.json();
      
      if (!validation.isValid) {
        let errorMessage = "This doesn't appear to be a valid handwriting sample. ";
        
        if (!validation.isHandwriting) {
          errorMessage += "Please write the text by hand rather than typing or printing it. ";
        }
        
        if (!validation.textMatches) {
          errorMessage += `The text doesn't match exactly. Expected: "${sampleTexts[currentSample]}"`;
          if (validation.extractedText) {
            errorMessage += ` but found: "${validation.extractedText}"`;
          }
        }
        
        toast.error(errorMessage);
        return false;
      }
      
      return true;
    } catch (error) {
      console.error('Validation error:', error);
      toast.error("Unable to validate the image. Please try again.");
      return false;
    } finally {
      setIsValidating(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast.error("Please select an image file");
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast.error("Image size should be less than 10MB");
      return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
      const result = e.target?.result as string;
      
      // Validate the handwriting
      const isValid = await validateHandwriting(result);
      if (isValid) {
        setUploadedImage(result);
        toast.success("Handwriting sample validated successfully!");
      }
    };
    reader.readAsDataURL(file);
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const removeUploadedImage = () => {
    setUploadedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleMobileImageReceived = async (imageUrl: string) => {
    // Validate the mobile image
    const isValid = await validateHandwriting(imageUrl);
    if (isValid) {
      const newMobileImages = new Map(mobileImages);
      newMobileImages.set(currentSample, imageUrl);
      setMobileImages(newMobileImages);
      toast.success("Photo received and validated!");
    }
  };

  const completeSample = () => {
    const newCompleted = new Set(completedSamples);
    newCompleted.add(currentSample);
    setCompletedSamples(newCompleted);
    
    toast.success(`Sample ${currentSample + 1} completed!`);
    
    if (currentSample < sampleTexts.length - 1) {
      setCurrentSample(currentSample + 1);
    }
  };

  const nextSample = () => {
    if (currentSample < sampleTexts.length - 1) {
      setCurrentSample(currentSample + 1);
    } else {
      finishCapture();
    }
  };

  const finishCapture = () => {
    // Collect all samples (canvas drawings, uploaded images, mobile images)
    const allSamples: (string | HTMLCanvasElement)[] = [];
    
    // Add canvas if we have drawn content
    if (canvasRef.current && hasDrawn) {
      allSamples.push(canvasRef.current);
    }
    
    // Add uploaded image
    if (uploadedImage) {
      allSamples.push(uploadedImage);
    }
    
    // Add mobile images
    mobileImages.forEach((imageUrl) => {
      allSamples.push(imageUrl);
    });
    
    onNext(allSamples);
  };

  const canCompleteSample = () => {
    switch (captureMethod) {
      case 'draw':
        return hasDrawn;
      case 'upload':
        return uploadedImage !== null;
      case 'mobile':
        return mobileImages.has(currentSample);
      default:
        return false;
    }
  };

  const canProceed = completedSamples.size === sampleTexts.length;

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-5xl p-8 shadow-elegant">
        <div className="space-y-6">
          {/* Header */}
          <div className="text-center space-y-4">
            <h1 className="text-3xl font-elegant text-ink">Capture Your Handwriting</h1>
            <p className="text-muted-foreground">
              Create samples by drawing on screen or uploading photos of your handwriting
            </p>
            <Progress value={progress} className="w-full max-w-md mx-auto" />
            <p className="text-sm text-muted-foreground">
              Sample {currentSample + 1} of {sampleTexts.length} • {completedSamples.size} completed
            </p>
          </div>

          {/* Sample Text */}
          <div className="text-center">
            <Card className="p-6 bg-warm-accent/20 border-dashed border-2 border-warm-accent">
              <p className="text-lg font-elegant text-ink">
                "{sampleTexts[currentSample]}"
              </p>
              {currentSampleCompleted && (
                <div className="mt-2 flex items-center justify-center gap-2 text-green-600">
                  <Check className="w-4 h-4" />
                  <span className="text-sm">Completed</span>
                </div>
              )}
            </Card>
          </div>

          {/* Method Selection */}
          <div className="flex justify-center">
            <Tabs value={captureMethod} onValueChange={(value) => setCaptureMethod(value as 'draw' | 'upload' | 'mobile')}>
              <TabsList className="grid w-full grid-cols-3 max-w-lg">
                <TabsTrigger value="draw" className="flex items-center gap-2">
                  <PenTool className="w-4 h-4" />
                  Draw
                </TabsTrigger>
                <TabsTrigger value="upload" className="flex items-center gap-2">
                  <Camera className="w-4 h-4" />
                  Upload
                </TabsTrigger>
                <TabsTrigger value="mobile" className="flex items-center gap-2">
                  <Smartphone className="w-4 h-4" />
                  Phone
                </TabsTrigger>
              </TabsList>

              <TabsContent value="draw" className="mt-6">
                <div className="space-y-4">
                  <Card className="p-4 bg-paper">
                    <canvas
                      ref={canvasRef}
                      width={800}
                      height={200}
                      className="w-full h-48 border border-border rounded cursor-crosshair bg-white"
                      onMouseDown={startDrawing}
                      onMouseMove={draw}
                      onMouseUp={stopDrawing}
                      onMouseLeave={stopDrawing}
                      onTouchStart={startDrawing}
                      onTouchMove={draw}
                      onTouchEnd={stopDrawing}
                    />
                  </Card>
                  
                  <div className="flex justify-center gap-4">
                    <Button variant="outline" onClick={clearCanvas} disabled={!hasDrawn}>
                      <RotateCcw className="w-4 h-4" />
                      Clear
                    </Button>
                  </div>

                  <div className="text-center text-sm text-muted-foreground space-y-1">
                    <p>✍️ Write naturally in your own handwriting style</p>
                    <p>📱 Use your finger on mobile or a stylus for best results</p>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="upload" className="mt-6">
                <div className="space-y-4">
                  <Card className="p-8 bg-paper">
                    {isValidating ? (
                      <div className="text-center space-y-6 py-8">
                        <div className="relative">
                          <Loader2 className="w-16 h-16 mx-auto text-primary animate-spin" />
                          <Brain className="w-6 h-6 absolute top-5 left-1/2 transform -translate-x-1/2 text-primary-light animate-pulse" />
                        </div>
                        <div className="space-y-2">
                          <h3 className="font-elegant text-xl text-ink">✨ Analyzing your handwriting...</h3>
                          <div className="space-y-1 text-muted-foreground">
                            <p className="flex items-center justify-center gap-2">
                              <Sparkles className="w-4 h-4 animate-pulse" />
                              Reading your beautiful penmanship
                            </p>
                            <p>🔍 Checking if text matches perfectly</p>
                            <p>🎨 Making sure it's handwritten (not typed!)</p>
                          </div>
                        </div>
                        <div className="text-xs text-muted-foreground italic">
                          This usually takes just a few seconds...
                        </div>
                      </div>
                    ) : !uploadedImage ? (
                      <div 
                        onClick={isValidating ? undefined : triggerFileUpload}
                        className={`border-2 border-dashed border-border rounded-lg p-8 text-center transition-colors ${
                          isValidating 
                            ? 'cursor-not-allowed opacity-50' 
                            : 'cursor-pointer hover:border-primary'
                        }`}
                      >
                        <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                        <h3 className="font-medium text-ink mb-2">Upload Handwriting Sample</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                          Take a photo of the sample text written in your handwriting
                        </p>
                        <Button variant="outline" disabled={isValidating}>
                          <Image className="w-4 h-4" />
                          Choose Image
                        </Button>
                      </div>
                    ) : (
                      <div className="text-center space-y-4">
                        <img 
                          src={uploadedImage} 
                          alt="Uploaded handwriting sample"
                          className="max-h-48 mx-auto rounded-lg border border-border"
                        />
                        <Button variant="outline" onClick={removeUploadedImage} disabled={isValidating}>
                          <RotateCcw className="w-4 h-4" />
                          Replace Image
                        </Button>
                      </div>
                    )}
                  </Card>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                    disabled={isValidating}
                  />

                  <div className="text-center text-sm text-muted-foreground space-y-1">
                    <p>📸 Take a clear photo of your handwritten sample</p>
                    <p>💡 Use good lighting and avoid shadows for best results</p>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="mobile" className="mt-6">
                <div className="space-y-4">
                  {mobileImages.has(currentSample) ? (
                    <Card className="p-6 space-y-4">
                      <h3 className="font-elegant text-lg text-ink text-center">Mobile Photo Received</h3>
                      <div className="text-center">
                        <img 
                          src={mobileImages.get(currentSample)} 
                          alt="Mobile handwriting sample"
                          className="max-h-48 mx-auto rounded-lg border border-border"
                        />
                      </div>
                      <Button 
                        variant="outline" 
                        onClick={() => {
                          const newMobileImages = new Map(mobileImages);
                          newMobileImages.delete(currentSample);
                          setMobileImages(newMobileImages);
                        }}
                        className="w-full"
                      >
                        Take New Photo
                      </Button>
                    </Card>
                  ) : (
                    <MobileUploadSidecar
                      sessionId={sessionId}
                      sampleText={sampleTexts[currentSample]}
                      onImageReceived={handleMobileImageReceived}
                      completed={false}
                    />
                  )}
                  {isValidating && (
                    <Card className="p-6 bg-gradient-primary/5 border-primary/20">
                      <div className="text-center space-y-4">
                        <div className="relative">
                          <Loader2 className="w-12 h-12 mx-auto text-primary animate-spin" />
                          <Brain className="w-5 h-5 absolute top-3.5 left-1/2 transform -translate-x-1/2 text-primary-light animate-pulse" />
                        </div>
                        <div className="space-y-1">
                          <h4 className="font-elegant text-ink">📱 Analyzing your photo...</h4>
                          <p className="text-sm text-muted-foreground">Making sure your handwriting looks perfect!</p>
                        </div>
                      </div>
                    </Card>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-center gap-4">
            {!currentSampleCompleted && (
              <Button 
                variant="elegant" 
                onClick={completeSample}
                disabled={!canCompleteSample()}
                size="lg"
              >
                <Check className="w-4 h-4" />
                Complete Sample
              </Button>
            )}

            {currentSampleCompleted && !canProceed && (
              <Button 
                variant="warm" 
                onClick={nextSample}
                size="lg"
              >
                Next Sample
                <ChevronRight className="w-4 h-4" />
              </Button>
            )}

            {canProceed && (
              <Button 
                variant="elegant" 
                onClick={finishCapture}
                size="lg"
              >
                Complete Setup
                <Check className="w-4 h-4" />
              </Button>
            )}
          </div>

          {/* Sample Navigation */}
          {sampleTexts.length > 1 && (
            <div className="flex justify-center space-x-2">
              {sampleTexts.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentSample(index)}
                  className={`w-3 h-3 rounded-full transition-colors ${
                    index === currentSample 
                      ? 'bg-primary' 
                      : completedSamples.has(index) 
                        ? 'bg-green-500' 
                        : 'bg-muted'
                  }`}
                />
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};