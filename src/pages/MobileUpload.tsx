import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { Camera, Upload, Check, ArrowLeft } from "lucide-react";
import { useSearchParams, Link } from "react-router-dom";

const MobileUpload = () => {
  const [searchParams] = useSearchParams();
  const { toast } = useToast();
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sessionId = searchParams.get('session') || '';
  const sampleText = searchParams.get('sample') || '';

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid file type",
        description: "Please select an image file",
        variant: "destructive"
      });
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Image size should be less than 10MB",
        variant: "destructive"
      });
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setUploadedImage(result);
    };
    reader.readAsDataURL(file);
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const triggerCamera = () => {
    if (fileInputRef.current) {
      fileInputRef.current.setAttribute('capture', 'environment');
      fileInputRef.current.click();
    }
  };

  const submitUpload = async () => {
    if (!uploadedImage || !sessionId) return;

    console.log('Submitting upload for session:', sessionId);
    setIsUploading(true);
    
    try {
      // Store in localStorage with session ID
      const uploadData = {
        sessionId,
        imageUrl: uploadedImage,
        timestamp: Date.now(),
        sampleText
      };
      
      const storageKey = `mobile-upload-${sessionId}`;
      localStorage.setItem(storageKey, JSON.stringify(uploadData));
      
      console.log('Stored mobile upload data:', storageKey, uploadData);
      
      // Also store a backup with a simpler key for fallback
      localStorage.setItem('latest-mobile-upload', JSON.stringify(uploadData));
      
      setIsComplete(true);
      toast({
        title: "Photo uploaded!",
        description: "Your handwriting sample has been sent to your computer"
      });
    } catch (error) {
      console.error('Error submitting upload:', error);
      toast({
        title: "Upload failed",
        description: "Please try again",
        variant: "destructive"
      });
    } finally {
      setIsUploading(false);
    }
  };

  const removeImage = () => {
    setUploadedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  if (!sessionId || !sampleText) {
    return (
      <div className="min-h-screen bg-gradient-warm flex items-center justify-center p-4">
        <Card className="p-8 text-center max-w-md">
          <h1 className="text-2xl font-elegant text-ink mb-4">Invalid Session</h1>
          <p className="text-muted-foreground mb-6">
            This upload link is not valid. Please scan the QR code again from your computer.
          </p>
          <Link to="/">
            <Button variant="elegant">Go to Main App</Button>
          </Link>
        </Card>
      </div>
    );
  }

  if (isComplete) {
    return (
      <div className="min-h-screen bg-gradient-warm flex items-center justify-center p-4">
        <Card className="p-8 text-center max-w-md space-y-6">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
            <Check className="w-8 h-8 text-green-600" />
          </div>
          <div className="space-y-2">
            <h1 className="text-2xl font-elegant text-ink">Upload Complete!</h1>
            <p className="text-muted-foreground">
              Your handwriting sample has been sent to your computer. You can close this page and continue on your computer.
            </p>
          </div>
          <Link to="/">
            <Button variant="elegant" className="w-full">
              Return to Main App
            </Button>
          </Link>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-warm">
      <div className="container max-w-2xl mx-auto p-4 space-y-6">
        {/* Header */}
        <Card className="p-6 text-center bg-gradient-subtle">
          <h1 className="text-2xl font-elegant text-ink mb-2">Mobile Upload</h1>
          <p className="text-muted-foreground text-sm">
            Write the sample text below and take a photo
          </p>
        </Card>

        {/* Sample Text */}
        <Card className="p-6 bg-warm-accent/20 border-dashed border-2 border-warm-accent">
          <h3 className="font-elegant text-lg text-ink mb-2">Write this text:</h3>
          <p className="text-lg font-elegant text-ink italic">
            "{sampleText}"
          </p>
        </Card>

        {/* Upload Interface */}
        <Card className="p-6 space-y-6">
          {!uploadedImage ? (
            <div className="space-y-4">
              <h3 className="font-elegant text-lg text-ink text-center">Take or Upload Photo</h3>
              
              <div className="grid grid-cols-1 gap-4">
                <Button
                  variant="elegant"
                  size="lg"
                  onClick={triggerCamera}
                  className="h-16 text-lg"
                >
                  <Camera className="w-6 h-6" />
                  Take Photo with Camera
                </Button>
                
                <Button
                  variant="warm"
                  size="lg"
                  onClick={triggerFileUpload}
                  className="h-16 text-lg"
                >
                  <Upload className="w-6 h-6" />
                  Choose from Gallery
                </Button>
              </div>

              <div className="bg-muted/30 p-4 rounded-lg text-sm text-muted-foreground space-y-2">
                <p>📝 <strong>Tips for best results:</strong></p>
                <ul className="space-y-1 ml-4 list-disc">
                  <li>Write clearly in your natural handwriting</li>
                  <li>Use good lighting and avoid shadows</li>
                  <li>Make sure the text fills most of the frame</li>
                  <li>Keep the paper flat and avoid glare</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <h3 className="font-elegant text-lg text-ink text-center">Review Your Photo</h3>
              
              <div className="text-center">
                <img 
                  src={uploadedImage} 
                  alt="Uploaded handwriting sample"
                  className="max-w-full max-h-80 mx-auto rounded-lg border border-border shadow-soft"
                />
              </div>
              
              <div className="flex gap-3">
                <Button 
                  variant="outline" 
                  onClick={removeImage}
                  className="flex-1"
                >
                  Retake Photo
                </Button>
                <Button 
                  variant="elegant" 
                  onClick={submitUpload}
                  disabled={isUploading}
                  className="flex-1"
                >
                  {isUploading ? "Uploading..." : "Send to Computer"}
                  <Check className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </Card>

        {/* Back Link */}
        <div className="text-center">
          <Link to="/" className="text-sm text-muted-foreground hover:text-ink transition-colors">
            <ArrowLeft className="w-4 h-4 inline mr-1" />
            Back to main app
          </Link>
        </div>
      </div>
    </div>
  );
};

export default MobileUpload;