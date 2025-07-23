import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { RotateCcw, Check, ChevronRight, Upload, Camera, PenTool, Image, Smartphone, Loader2, Brain, Sparkles, RefreshCw, ArrowRight } from "lucide-react";
import { toast } from "sonner";
import { MobileUploadSidecar } from "./MobileUploadSidecar";
import { analyzeHandwritingSamples, generateHandwritingStyle } from "@/lib/handwriting";
import { supabase } from "@/integrations/supabase/client";

interface HandwritingCaptureProps {
  onNext: (samples: (string | HTMLCanvasElement)[]) => void;
  user?: any; // Add user prop to load existing samples
}

const sampleTexts = [
  "The quick brown fox jumps over the lazy dog.",
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890",
  "Thank you so much! How are you? I'm fine, thanks.",
  "With love & appreciation - it's wonderful to see you.",
  "Best wishes: sincerely yours (always & forever)"
];

export const HandwritingCapture = ({ onNext, user }: HandwritingCaptureProps) => {
  const [currentSample, setCurrentSample] = useState(0);
  const [captureMethod, setCaptureMethod] = useState<'draw' | 'upload' | 'mobile'>('draw');
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedImages, setUploadedImages] = useState<Map<number, string>>(new Map());
  const [mobileImages, setMobileImages] = useState<Map<number, string>>(new Map());
  const [completedSamples, setCompletedSamples] = useState<Set<number>>(new Set());
  const [isValidating, setIsValidating] = useState(false);
  const [sessionId] = useState(() => `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
  const [showPreview, setShowPreview] = useState(false);
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false);
  const [previewSvg, setPreviewSvg] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<{
    isValid: boolean;
    extractedText?: string;
    expectedText: string;
    isHandwriting?: boolean;
    textMatches?: boolean;
    originalImageData?: string; // Store the original image data
  } | null>(null);
  
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

  // Load existing samples on component mount
  useEffect(() => {
    const loadExistingSamples = async () => {
      try {
        // Load user's saved samples if authenticated
        if (user) {
          console.log('🔍 Loading existing user samples for user:', user.id);
          const { data, error } = await supabase
            .from('user_style_models')
            .select('sample_images')
            .eq('user_id', user.id)
            .order('created_at', { ascending: false })
            .limit(1)
            .maybeSingle();

          console.log('📊 Database query result:', { data, error, userId: user.id });

          if (data?.sample_images && !error) {
            console.log('✅ Found saved samples:', data.sample_images);
            const newUploadedImages = new Map<number, string>();
            const newCompleted = new Set<number>();
            
            // Handle both array and object formats for saved samples
            if (Array.isArray(data.sample_images)) {
              // Legacy array format
              data.sample_images.forEach((imageUrl: any, index: number) => {
                if (index < sampleTexts.length && typeof imageUrl === 'string') {
                  newUploadedImages.set(index, imageUrl);
                  newCompleted.add(index);
                }
              });
            } else if (typeof data.sample_images === 'object' && data.sample_images !== null) {
              // New object format
              Object.entries(data.sample_images).forEach(([indexStr, imageUrl]) => {
                const index = parseInt(indexStr);
                if (index < sampleTexts.length && typeof imageUrl === 'string') {
                  newUploadedImages.set(index, imageUrl);
                  newCompleted.add(index);
                }
              });
            }
            
            setUploadedImages(newUploadedImages);
            setCompletedSamples(newCompleted);
            toast.success(`Loaded ${newCompleted.size} existing handwriting samples!`);
            return;
          } else if (error) {
            console.log('⚠️ No saved samples found:', error.message);
          } else {
            console.log('⚠️ No saved samples in database');
          }
        }

        // Load mobile uploads as fallback
        console.log('🔍 Checking for existing mobile uploads...');
        const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
        const { data, error } = await supabase
          .from('mobile_uploads')
          .select('*')
          .gte('created_at', oneHourAgo);

        if (error) {
          console.error('Error loading mobile uploads:', error);
          return;
        }

        if (data && data.length > 0) {
          console.log('📱 Found existing mobile uploads:', data.length);
          const newMobileImages = new Map<number, string>();
          const newCompleted = new Set<number>();
          
          data.forEach((upload) => {
            const sampleIndex = sampleTexts.findIndex(text => text === upload.sample_text);
            if (sampleIndex !== -1 && upload.image_data) {
              newMobileImages.set(sampleIndex, upload.image_data);
              newCompleted.add(sampleIndex);
            }
          });
          
          if (newMobileImages.size > 0) {
            setMobileImages(newMobileImages);
            setCompletedSamples(newCompleted);
            toast.success(`Found ${newMobileImages.size} handwriting samples from your phone!`);
          }
        }
      } catch (error) {
        console.error('Error loading existing samples:', error);
      }
    };
    
    loadExistingSamples();
  }, [user]); // Reload when user changes

  // Reset states when switching methods or samples
  useEffect(() => {
    setHasDrawn(false);
    setUploadedImage(null);
    setValidationResult(null);
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
          expectedText: sampleTexts[currentSample]
        })
      });

      const validation = await response.json();
      
      // Store validation result for potential override
      setValidationResult({
        isValid: validation.isValid,
        extractedText: validation.extractedText,
        expectedText: sampleTexts[currentSample],
        isHandwriting: validation.isHandwriting,
        textMatches: validation.textMatches,
        originalImageData: imageData // Store the original image data
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
          
          toast.error(errorMessage);
          return false;
        }
      }
      
      return true;
    } catch (error) {
      console.error('Validation error:', error);
      toast.error("Unable to validate the image. Please try again.");
      setValidationResult(null);
      return false;
    } finally {
      setIsValidating(false);
    }
  };

  const acceptValidationOverride = async () => {
    if (validationResult?.originalImageData) {
      console.log('🔄 User accepted validation override - saving sample anyway');
      console.log('🔄 Current sample index:', currentSample);
      console.log('🔄 Image data length:', validationResult.originalImageData.length);
      
      // Use the stored image data directly
      setUploadedImage(validationResult.originalImageData);
      setValidationResult(null);
      
      // CRITICAL: Save to database immediately when user accepts override
      await handleImageCapture(validationResult.originalImageData);
      
      // Complete the sample immediately
      const newCompleted = new Set(completedSamples);
      newCompleted.add(currentSample);
      setCompletedSamples(newCompleted);
      
      console.log('✅ Sample accepted and saved with override!');
      toast.success("Handwriting sample accepted and saved!");
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
    console.log('📁 handleFileUpload called');
    const file = event.target.files?.[0];
    if (!file) {
      console.log('❌ No file selected');
      return;
    }

    console.log('📁 File selected:', { name: file.name, size: file.size, type: file.type });

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast.error("Please select an image file");
      // Reset file input to allow selecting the same file again
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast.error("Image size should be less than 10MB");
      // Reset file input to allow selecting the same file again
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      return;
    }

    console.log('📖 Starting FileReader...');
    const reader = new FileReader();
    reader.onload = async (e) => {
      console.log('📖 FileReader onload triggered');
      const result = e.target?.result as string;
      console.log('📖 File read complete, data length:', result?.length);
      
      // Validate the handwriting
      console.log('🔍 Starting handwriting validation...');
      const isValid = await validateHandwriting(result);
      console.log('🔍 Validation result:', isValid);
      
      if (isValid) {
        console.log('✅ Validation passed, setting uploaded image and calling handleImageCapture...');
        setUploadedImage(result);
        
        // Complete the sample immediately since validation passed
        const newCompleted = new Set(completedSamples);
        newCompleted.add(currentSample);
        setCompletedSamples(newCompleted);
        
        // CRITICAL: Also call handleImageCapture to save to database
        await handleImageCapture(result);
        toast.success("Handwriting sample validated and saved successfully!");
      } else {
        console.log('❌ Validation failed, checking for override case...');
        console.log('🔍 Validation result:', validationResult);
        // Only reset if this is not an override case (validation result shows override UI)
        if (!validationResult || !validationResult.isHandwriting || validationResult.textMatches || !validationResult.extractedText) {
          console.log('🔄 Resetting file input due to non-override validation failure');
          // Reset file input to allow selecting the same file again after validation failure
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
        } else {
          console.log('🔍 This appears to be an override case - keeping file input');
        }
        // If validationResult shows override UI, don't reset - let user choose
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
    console.log('🎯 handleMobileImageReceived called!');
    console.log('📸 Image URL length:', imageUrl.length);
    console.log('📊 Current sample:', currentSample);
    console.log('🔄 Mobile images before:', Array.from(mobileImages.keys()));
    
    // Mobile images are already validated on the mobile side, so accept directly
    const newMobileImages = new Map(mobileImages);
    newMobileImages.set(currentSample, imageUrl);
    setMobileImages(newMobileImages);
    
    console.log('🔄 Mobile images after:', Array.from(newMobileImages.keys()));
    
    // Complete the sample automatically
    const newCompleted = new Set(completedSamples);
    newCompleted.add(currentSample);
    setCompletedSamples(newCompleted);
    
    console.log('✅ Completed samples:', Array.from(newCompleted));
    
    toast.success("Photo received from mobile!");
    
    // Auto-advance to next sample if not the last one
    if (currentSample < sampleTexts.length - 1) {
      console.log('➡️ Auto-advancing to next sample');
      setCurrentSample(currentSample + 1);
    } else {
      console.log('🏁 This was the last sample');
    }
  };

  const handleImageCapture = async (imageData: string) => {
    console.log('🎯 handleImageCapture called with image data length:', imageData.length);
    console.log('🎯 Current sample index:', currentSample);
    console.log('🎯 User authenticated:', !!user);
    
    // Update the uploaded images map
    const newUploadedImages = new Map(uploadedImages);
    newUploadedImages.set(currentSample, imageData);
    setUploadedImages(newUploadedImages);
    
    // Save this individual sample to database immediately if user is authenticated
    if (user) {
      console.log('🚀 Saving individual sample to database...', { 
        userId: user.id, 
        currentSample,
        sampleIndex: currentSample
      });
      
      try {
        // Check if user already has a style model
        const { data: existingModel, error: queryError } = await supabase
          .from('user_style_models')
          .select('*')
          .eq('user_id', user.id)
          .maybeSingle();

        if (queryError) {
          console.error('❌ Error querying existing model:', queryError);
          throw queryError;
        }

        // Get current samples from database and merge with new one
        let currentSamples: { [key: number]: string } = {};
        if (existingModel?.sample_images && typeof existingModel.sample_images === 'object') {
          // Handle both array and object formats for backwards compatibility
          if (Array.isArray(existingModel.sample_images)) {
            // Convert array to object, filtering out nulls
            existingModel.sample_images.forEach((img: any, index: number) => {
              if (img && typeof img === 'string') {
                currentSamples[index] = img;
              }
            });
          } else {
            // Already an object - convert keys to numbers and ensure values are strings
            Object.entries(existingModel.sample_images).forEach(([key, value]) => {
              const numKey = parseInt(key);
              if (!isNaN(numKey) && value && typeof value === 'string') {
                currentSamples[numKey] = value;
              }
            });
          }
        }

        // Add the new sample at the correct index
        console.log('📝 Adding sample at index:', currentSample);
        console.log('📝 Current samples before adding:', Object.keys(currentSamples));
        currentSamples[currentSample] = imageData;
        console.log('📝 Current samples after adding:', Object.keys(currentSamples));
        
        let result;
        if (existingModel) {
          console.log('🔄 Updating existing model with individual sample...');
          result = await supabase
            .from('user_style_models')
            .update({
              sample_images: currentSamples,
              training_status: 'pending',
              training_started_at: null,
              training_completed_at: null
            })
            .eq('user_id', user.id);
        } else {
          console.log('➕ Creating new model with first sample...');
          result = await supabase
            .from('user_style_models')
            .insert({
              user_id: user.id,
              model_id: `user_${user.id}_${Date.now()}`,
              sample_images: currentSamples,
              training_status: 'pending'
            });
        }
        
        console.log('📤 Database operation result:', result);
        
        if (result.error) {
          console.error('❌ Database save error:', result.error);
          throw result.error;
        }
        
        console.log('✅ Sample saved to database successfully!');
        
        // Verify the save by querying back
        const { data: verifyData, error: verifyError } = await supabase
          .from('user_style_models')
          .select('*')
          .eq('user_id', user.id)
          .maybeSingle();
          
        console.log('🔍 Verification query result:', { 
          data: verifyData ? { 
            id: verifyData.id,
            sampleImages: verifyData.sample_images,
            samplesCount: verifyData.sample_images ? Object.keys(verifyData.sample_images).length : 0
          } : null, 
          error: verifyError 
        });
        
      } catch (error) {
        console.error('❌ Critical error saving sample:', error);
        toast.error('Failed to save sample to database');
      }
    } else {
      console.log('⚠️ No authenticated user, skipping database save');
    }
  };

  const completeSample = async () => {
    // Store the current uploaded image for this sample
    if (uploadedImage) {
      console.log(`💾 Completing sample ${currentSample} with uploaded image`);
      // Call the centralized save function
      await handleImageCapture(uploadedImage);
    } else if (hasDrawn && canvasRef.current) {
      console.log(`💾 Completing sample ${currentSample} with drawn image`);
      const canvasData = canvasRef.current.toDataURL();
      await handleImageCapture(canvasData);
    }
    
    const newCompleted = new Set(completedSamples);
    newCompleted.add(currentSample);
    setCompletedSamples(newCompleted);
    
    toast.success(`Sample ${currentSample + 1} completed and saved!`);
    
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

  const generatePreviewSample = async () => {
    setIsGeneratingPreview(true);
    try {
      // Check if user has a trained model first
      const { data: userData } = await supabase.auth.getUser();
      if (!userData.user) {
        toast.error("Please log in to generate handwriting");
        return;
      }

      const { data: modelData, error: modelError } = await supabase
        .from('user_style_models')
        .select('*')
        .eq('user_id', userData.user.id)
        .maybeSingle();

      if (modelError || !modelData) {
        toast.error("No trained model found. Please complete the handwriting capture process first.");
        return;
      }

      if (modelData.training_status !== 'completed') {
        toast.error("Your handwriting model is still training. Please wait for it to complete.");
        return;
      }

      // Generate preview message
      const previewMessage = "Hey there! Here is our best attempt at matching your handwriting. How does this look to you?";
      
      // Generate handwriting SVG using the trained model
      const result = await generateHandwritingStyle(previewMessage, null, [], modelData.model_id);
      
      // Handle the new response format
      if (typeof result === 'string') {
        setPreviewSvg(result);
      } else {
        // Handle training status or other object responses
        console.log('Received training status:', result);
        setPreviewSvg(''); // Clear SVG during training
      }
      setShowPreview(true);
      
      toast.success("✨ Handwriting preview generated!");
    } catch (error) {
      console.error('Error generating preview:', error);
      toast.error("Failed to generate handwriting preview. Please try again.");
    } finally {
      setIsGeneratingPreview(false);
    }
  };

  const finishCapture = () => {
    // Collect all completed samples across all 5 steps
    const allSamples: (string | HTMLCanvasElement)[] = [];
    
    // Add mobile images (these are keyed by sample index)
    mobileImages.forEach((imageUrl) => {
      allSamples.push(imageUrl);
    });
    
    // Add uploaded images from all samples
    uploadedImages.forEach((imageUrl) => {
      allSamples.push(imageUrl);
    });
    
    // Add any additional samples from current session if needed
    if (canvasRef.current && hasDrawn) {
      allSamples.push(canvasRef.current);
    }
    
    if (uploadedImage) {
      allSamples.push(uploadedImage);
    }
    
    console.log('🎯 Finishing capture with samples:', allSamples.length);
    onNext(allSamples);
  };

  const collectMoreSamples = () => {
    setShowPreview(false);
    setPreviewSvg(null);
    // Reset to continue collecting more samples
    setCurrentSample(0);
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

  // Show preview screen if all samples completed and preview requested
  if (showPreview) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <Card className="w-full max-w-4xl p-8 shadow-elegant">
          <div className="text-center space-y-8">
            <div className="space-y-4">
              <h1 className="text-3xl font-elegant text-ink">✨ Your Handwriting Preview</h1>
              <p className="text-muted-foreground">
                We've analyzed your samples and generated this preview in your handwriting style
              </p>
            </div>

            {isGeneratingPreview ? (
              <div className="space-y-6 py-12">
                <div className="relative">
                  <Loader2 className="w-20 h-20 mx-auto text-primary animate-spin" />
                  <Sparkles className="w-8 h-8 absolute top-6 left-1/2 transform -translate-x-1/2 text-primary-light animate-pulse" />
                </div>
                <div className="space-y-3">
                  <h3 className="font-elegant text-2xl text-ink">🎨 Creating your handwriting magic...</h3>
                  <div className="space-y-2 text-muted-foreground">
                    <p className="flex items-center justify-center gap-2">
                      <Brain className="w-5 h-5 animate-pulse" />
                      Analyzing your unique writing style
                    </p>
                    <p>✍️ Capturing the essence of your penmanship</p>
                    <p>🌟 Generating your personalized message</p>
                  </div>
                </div>
                <div className="text-sm text-muted-foreground italic">
                  This might take a moment to get it just right...
                </div>
              </div>
            ) : previewSvg ? (
              <div className="space-y-6">
                <Card className="p-8 bg-paper border-2 border-warm-accent/30">
                  <div 
                    className="handwriting-preview mx-auto"
                    dangerouslySetInnerHTML={{ __html: previewSvg }}
                    style={{ maxWidth: '100%', overflow: 'hidden' }}
                  />
                </Card>
                
                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                  <Button 
                    variant="outline" 
                    onClick={collectMoreSamples}
                    size="lg"
                    className="flex items-center gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Collect More Samples
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => generatePreviewSample()}
                    size="lg"
                    className="flex items-center gap-2"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Generate Another
                  </Button>
                  <Button 
                    variant="elegant" 
                    onClick={finishCapture}
                    size="lg"
                    className="flex items-center gap-2"
                  >
                    Perfect! Continue
                    <ArrowRight className="w-4 h-4" />
                  </Button>
                </div>
                
                <div className="text-center text-sm text-muted-foreground space-y-1">
                  <p>💝 Love how it looks? Continue to create your thank you note!</p>
                  <p>🔄 Want it even more accurate? Collect a few more handwriting samples!</p>
                </div>
              </div>
            ) : (
              <div className="text-center space-y-4">
                <p className="text-muted-foreground">Something went wrong generating the preview.</p>
                <Button onClick={collectMoreSamples} variant="outline">
                  Try Again
                </Button>
              </div>
            )}
          </div>
        </Card>
      </div>
    );
  }

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

          {/* Method Selection or Completed State */}
          {currentSampleCompleted && (uploadedImages.get(currentSample) || mobileImages.get(currentSample)) ? (
            <div className="space-y-6">
              <Card className="p-6 bg-green-50 border-2 border-green-200">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                    <Check className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-green-800">Sample Completed</h3>
                </div>
                
                <div className="bg-white rounded-lg p-4 border border-green-200 mb-6">
                  <img 
                    src={uploadedImages.get(currentSample) || mobileImages.get(currentSample)} 
                    alt={`Handwriting sample ${currentSample + 1}`}
                    className="max-w-full h-auto rounded shadow-sm mx-auto"
                    style={{ maxHeight: '300px' }}
                  />
                </div>
                
                <div className="flex gap-3 justify-center">
                  <Button variant="outline" onClick={() => {
                    // Remove from completed and allow re-upload
                    const newCompleted = new Set(completedSamples);
                    newCompleted.delete(currentSample);
                    setCompletedSamples(newCompleted);
                    
                    // Clear the stored image
                    const newUploaded = new Map(uploadedImages);
                    const newMobile = new Map(mobileImages);
                    newUploaded.delete(currentSample);
                    newMobile.delete(currentSample);
                    setUploadedImages(newUploaded);
                    setMobileImages(newMobile);
                    
                    toast.info("You can now re-upload this sample");
                  }}>
                    <Upload className="w-4 h-4 mr-2" />
                    Re-upload Sample
                  </Button>
                  
                  <Button onClick={nextSample} variant="elegant">
                    <ArrowRight className="w-4 h-4 mr-2" />
                    {currentSample < sampleTexts.length - 1 ? "Next Sample" : "Continue"}
                  </Button>
                </div>
              </Card>
            </div>
          ) : (
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
                    {/* Hide upload section when validation override is showing */}
                    {!(validationResult && !validationResult.isValid && validationResult.isHandwriting && !validationResult.textMatches && validationResult.extractedText) && (
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
                    )}

                    {/* Validation Override UI */}
                    {validationResult && !validationResult.isValid && validationResult.isHandwriting && !validationResult.textMatches && validationResult.extractedText && (
                      <Card className="p-6 bg-yellow-50 border-yellow-200 border-2">
                        <div className="space-y-4">
                          <div className="text-center">
                            <h3 className="font-medium text-ink mb-2">🤔 Close Match Detected</h3>
                            <p className="text-sm text-muted-foreground mb-4">
                              We detected some handwriting, but the text doesn't match exactly. Here's what we found:
                            </p>
                          </div>
                          
                          <div className="space-y-3 text-sm">
                            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 p-3 bg-white rounded-lg border">
                              <span className="font-medium text-muted-foreground">Expected:</span>
                              <span className="font-mono text-ink">"{validationResult.expectedText}"</span>
                            </div>
                            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 p-3 bg-white rounded-lg border">
                              <span className="font-medium text-muted-foreground">We found:</span>
                              <span className="font-mono text-ink">"{validationResult.extractedText}"</span>
                            </div>
                          </div>
                          
                          <div className="text-center text-sm text-muted-foreground mb-4">
                            Does this look close enough to accept? Small differences in punctuation or letter recognition are normal.
                          </div>
                          
                          <div className="flex flex-col sm:flex-row gap-3">
                            <Button 
                              variant="outline" 
                              onClick={rejectValidationOverride}
                              className="flex-1"
                            >
                              <RotateCcw className="w-4 h-4" />
                              Try Different Photo
                            </Button>
                            <Button 
                              variant="elegant" 
                              onClick={acceptValidationOverride}
                              className="flex-1"
                            >
                              <Check className="w-4 h-4" />
                              Accept This Sample
                            </Button>
                          </div>
                        </div>
                      </Card>
                    )}

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
                        sessionId={`${sessionId}-sample-${currentSample}`}
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
          )}

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
                onClick={generatePreviewSample}
                disabled={isGeneratingPreview}
                size="lg"
                className="flex items-center gap-2"
              >
                {isGeneratingPreview ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Generating Preview...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Generate Preview
                  </>
                )}
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
