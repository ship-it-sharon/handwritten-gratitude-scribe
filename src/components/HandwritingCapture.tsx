import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { RotateCcw, Check, ChevronRight } from "lucide-react";

interface HandwritingCaptureProps {
  onNext: () => void;
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
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);

  const progress = ((currentSample + (hasDrawn ? 1 : 0)) / sampleTexts.length) * 100;

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

  const nextSample = () => {
    if (currentSample < sampleTexts.length - 1) {
      setCurrentSample(currentSample + 1);
      setHasDrawn(false);
      clearCanvas();
    } else {
      onNext();
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-4xl p-8 shadow-elegant">
        <div className="space-y-6">
          {/* Header */}
          <div className="text-center space-y-4">
            <h1 className="text-3xl font-elegant text-ink">Capture Your Handwriting</h1>
            <p className="text-muted-foreground">
              Write the sample text below in your natural handwriting style
            </p>
            <Progress value={progress} className="w-full max-w-md mx-auto" />
            <p className="text-sm text-muted-foreground">
              Sample {currentSample + 1} of {sampleTexts.length}
            </p>
          </div>

          {/* Sample Text */}
          <div className="text-center">
            <Card className="p-6 bg-warm-accent/20 border-dashed border-2 border-warm-accent">
              <p className="text-lg font-elegant text-ink">
                "{sampleTexts[currentSample]}"
              </p>
            </Card>
          </div>

          {/* Drawing Canvas */}
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
              <Button 
                variant="elegant" 
                onClick={nextSample} 
                disabled={!hasDrawn}
                size="lg"
              >
                {currentSample < sampleTexts.length - 1 ? (
                  <>
                    Next Sample
                    <ChevronRight className="w-4 h-4" />
                  </>
                ) : (
                  <>
                    Complete Setup
                    <Check className="w-4 h-4" />
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Instructions */}
          <div className="text-center text-sm text-muted-foreground space-y-2">
            <p>✍️ Write naturally in your own handwriting style</p>
            <p>📱 Use your finger on mobile or a stylus for best results</p>
          </div>
        </div>
      </Card>
    </div>
  );
};