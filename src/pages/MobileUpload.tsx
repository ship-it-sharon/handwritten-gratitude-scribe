import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { Camera, Upload, Check, ArrowLeft, RotateCcw } from "lucide-react";
import { useSearchParams, Link } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";

const MobileUpload = () => {
  const [searchParams] = useSearchParams();
  const { toast } = useToast();
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [validationResult, setValidationResult] = useState<{
    isValid: boolean;
    extractedText?: string;
    expectedText: string;
    isHandwriting?: boolean;
    textMatches?: boolean;
    originalImageData?: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sessionId = searchParams.get('session') || '';
  const sampleText = searchParams.get('sample') || '';

  const validateHandwriting = async (imageData: string) => {
    setIsValidating(true);
    setValidationResult(null);
    try {
      const response = await fetch(`https://lkqjlibxmsnjqaifipes.supabase.co/functions/v1/validate-handwriting`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxrcWpsaWJ4bXNuanFhaWZpcGVzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIyOTgyNzQsImV4cCI6MjA2Nzg3NDI3NH0.mpltb2Pc2H2vQNAuYQntJv462kFvyHG6yxe5Yt-pdto`
        },
        body: JSON.stringify({
          imageData,
          expectedText: sampleText
        })
      });

      const validation = await response.json();
      
      // Store validation result for potential override
      setValidationResult({
        isValid: validation.isValid,
        extractedText: validation.extractedText,
        expectedText: sampleText,
        isHandwriting: validation.isHandwriting,
        textMatches: validation.textMatches,
        originalImageData: imageData
      });
      
      if (!validation.isValid) {
        // Don't show error immediately if it's just a text mismatch (allow override)
        if (validation.isHandwriting && !validation.textMatches && validation.extractedText) {
          // This is a potential override case - return false but don't show error
          return false;
        } else {
          // Show error for other validation failures
          let errorMessage = "This doesn't appear to be a valid handwriting sample. ";
          
          if (!validation.isHandwriting) {
            errorMessage += "Please write the text by hand rather than typing or printing it. ";
          }
          
          toast({
            title: "Validation Failed",
            description: errorMessage,
            variant: "destructive"
          });
          return false;
        }
      }
      
      return true;
    } catch (error) {
      console.error('Validation error:', error);
      toast({
        title: "Validation Error",
        description: "Unable to validate the image. Please try again.",
        variant: "destructive"
      });
      setValidationResult(null);
      return false;
    } finally {
      setIsValidating(false);
    }
  };

  const acceptValidationOverride = async () => {
    if (validationResult?.originalImageData) {
      setUploadedImage(validationResult.originalImageData);
      setValidationResult(null);
      toast({
        title: "Photo accepted!",
        description: "Your handwriting sample has been accepted"
      });
      
      // Automatically proceed to upload after accepting override
      await submitUploadWithImage(validationResult.originalImageData);
    }
  };

  const rejectValidationOverride = () => {
    setValidationResult(null);
    // Reset file input to allow new upload
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
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
    reader.onload = async (e) => {
      const result = e.target?.result as string;
      
      // Validate the handwriting
      const isValid = await validateHandwriting(result);
      if (isValid) {
        setUploadedImage(result);
        toast({
          title: "Photo validated!",
          description: "Your handwriting sample looks good"
        });
        
        // Automatically proceed to upload after successful validation
        await submitUploadWithImage(result);
        // Don't show the upload button UI since we've already uploaded
        return;
      } else {
        // Only reset if this is not an override case
        if (!validationResult || !validationResult.isHandwriting || validationResult.textMatches || !validationResult.extractedText) {
          // Reset file input to allow selecting the same file again after validation failure
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
        }
      }
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

  const submitUploadWithImage = async (imageData: string) => {
    if (!sessionId) {
      console.error('Missing session ID');
      return;
    }

    console.log('=== MOBILE UPLOAD DEBUG ===');
    console.log('Session ID:', sessionId);
    console.log('Sample Text:', sampleText);
    console.log('Image size:', imageData.length, 'characters');
    
    setIsUploading(true);
    
    try {
      // Store in Supabase database using UPSERT to handle duplicate session IDs
      const { data, error } = await supabase
        .from('mobile_uploads')
        .upsert({
          session_id: sessionId,
          image_data: imageData,
          sample_text: sampleText
        }, {
          onConflict: 'session_id'
        })
        .select()
        .single();

      if (error) {
        console.error('Supabase upload error:', error);
        throw new Error(`Failed to upload: ${error.message}`);
      }

      console.log('✅ Successfully uploaded to Supabase:', data?.id);
      console.log('=== MOBILE UPLOAD SUCCESS ===');
      
      setIsComplete(true);
      toast({
        title: "Photo uploaded!",
        description: "Your handwriting sample has been sent to your computer"
      });
    } catch (error) {
      console.error('=== MOBILE UPLOAD ERROR ===');
      console.error('Error details:', error);
      console.error('Error type:', typeof error);
      console.error('Error message:', error instanceof Error ? error.message : 'Unknown error');
      
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Please try again",
        variant: "destructive"
      });
    } finally {
      setIsUploading(false);
    }
  };

  const submitUpload = async () => {
    if (!uploadedImage) {
      console.error('No uploaded image available');
      return;
    }
    
    await submitUploadWithImage(uploadedImage);
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
                  disabled={isValidating}
                  className="h-16 text-lg"
                >
                  <Camera className="w-6 h-6" />
                  Take Photo with Camera
                </Button>
                
                <Button
                  variant="warm"
                  size="lg"
                  onClick={triggerFileUpload}
                  disabled={isValidating}
                  className="h-16 text-lg"
                >
                  <Upload className="w-6 h-6" />
                  Choose from Gallery
                </Button>
              </div>

              {isValidating && (
                <div className="text-center text-muted-foreground">
                  🔍 Validating your handwriting sample...
                </div>
              )}

              {/* Text Mismatch Override UI */}
              {validationResult && validationResult.isHandwriting && !validationResult.textMatches && validationResult.extractedText && (
                <Card className="p-4 bg-amber-50 border-amber-200">
                  <div className="space-y-3">
                    <div className="flex items-start gap-3">
                      <div className="text-amber-600 mt-1">⚠️</div>
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-amber-800">Text doesn't match exactly</p>
                        <div className="text-sm text-amber-700 space-y-1">
                          <p><strong>Expected:</strong> "{validationResult.expectedText}"</p>
                          <p><strong>Found:</strong> "{validationResult.extractedText}"</p>
                        </div>
                        <p className="text-sm text-amber-700">
                          Would you like to use this handwriting sample anyway? Minor differences are common.
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={acceptValidationOverride}
                        className="flex-1"
                      >
                        Accept Sample
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={rejectValidationOverride}
                        className="flex-1"
                      >
                        Try Again
                      </Button>
                    </div>
                  </div>
                </Card>
              )}

              <div className="bg-muted/30 p-4 rounded-lg text-sm text-muted-foreground space-y-2">
                <p>📝 <strong>Tips for best results:</strong></p>
                <ul className="space-y-1 ml-4 list-disc">
                  <li>Write clearly in your natural handwriting</li>
                  <li>Use good lighting and avoid shadows</li>
                  <li>Make sure the text fills most of the frame</li>
                  <li>Keep the paper flat and avoid glare</li>
                  <li>Write the exact text shown above</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <h3 className="font-elegant text-lg text-ink text-center">✅ Photo Validated</h3>
              
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
                  <RotateCcw className="w-4 h-4" />
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