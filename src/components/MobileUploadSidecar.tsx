import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Smartphone, RefreshCw, CheckCircle, Clock } from "lucide-react";
import QRCode from "qrcode";
import { supabase } from "@/integrations/supabase/client";

interface MobileUploadSidecarProps {
  sessionId: string;
  sampleText: string;
  onImageReceived: (imageUrl: string) => void;
  completed?: boolean;
}

export const MobileUploadSidecar = ({ 
  sessionId, 
  sampleText, 
  onImageReceived,
  completed = false
}: MobileUploadSidecarProps) => {
  const [qrCodeUrl, setQrCodeUrl] = useState<string>("");
  const [isPolling, setIsPolling] = useState(true);

  // Generate QR code that links to mobile upload page
  useEffect(() => {
    const generateQR = async () => {
      try {
        const uploadUrl = `${window.location.origin}/mobile-upload?session=${sessionId}&sample=${encodeURIComponent(sampleText)}`;
        const qrUrl = await QRCode.toDataURL(uploadUrl, {
          width: 200,
          margin: 2,
          color: {
            dark: '#1e3a8a', // Dark blue in hex
            light: '#FFFFFF'
          }
        });
        setQrCodeUrl(qrUrl);
      } catch (error) {
        console.error('Error generating QR code:', error);
        // Fallback: try generating without custom colors
        try {
          const uploadUrl = `${window.location.origin}/mobile-upload?session=${sessionId}&sample=${encodeURIComponent(sampleText)}`;
          const qrUrl = await QRCode.toDataURL(uploadUrl, {
            width: 200,
            margin: 2
          });
          setQrCodeUrl(qrUrl);
        } catch (fallbackError) {
          console.error('Fallback QR generation also failed:', fallbackError);
        }
      }
    };

    generateQR();
  }, [sessionId, sampleText]);

  // Poll for uploaded images from Supabase
  useEffect(() => {
    if (!isPolling || completed) return;

    const pollForImage = async () => {
      try {
        console.log('=== POLLING DEBUG ===');
        console.log('Session ID:', sessionId);
        console.log('Polling active:', isPolling);
        console.log('Completed:', completed);
        console.log('Looking for uploads...');
        
        const { data, error } = await supabase
          .from('mobile_uploads')
          .select('*')
          .eq('session_id', sessionId)
          .single();

        if (error) {
          if (error.code !== 'PGRST116') { // Not found error is expected
            console.error('Supabase polling error:', error);
          }
          console.log('No upload found yet');
          return;
        }

        if (data && data.image_data) {
          console.log('📸 Image found in Supabase! ID:', data.id);
          console.log('📱 Calling onImageReceived with image data length:', data.image_data.length);
          console.log('📱 Session ID that worked:', data.session_id);
          onImageReceived(data.image_data);
          setIsPolling(false);
          
          // Clean up the database record
          try {
            await supabase.from('mobile_uploads').delete().eq('id', data.id);
            console.log('🧹 Cleaned up database record');
          } catch (cleanupError) {
            console.error('Failed to clean up database:', cleanupError);
          }
        }
        
        console.log('=== END POLLING DEBUG ===');
      } catch (error) {
        console.error('Error polling for image:', error);
      }
    };

    const interval = setInterval(pollForImage, 2000); // Poll every 2 seconds
    
    return () => {
      clearInterval(interval);
    };
  }, [isPolling, sessionId, onImageReceived, completed]);

  const regenerateQR = () => {
    // Force regenerate QR with new timestamp to refresh session
    const newSessionId = `${sessionId}-${Date.now()}`;
    const generateQR = async () => {
      try {
        const uploadUrl = `${window.location.origin}/mobile-upload?session=${newSessionId}&sample=${encodeURIComponent(sampleText)}`;
        const qrUrl = await QRCode.toDataURL(uploadUrl, {
          width: 200,
          margin: 2,
          color: {
            dark: '#1e3a8a', // Dark blue in hex
            light: '#FFFFFF'
          }
        });
        setQrCodeUrl(qrUrl);
      } catch (error) {
        console.error('Error regenerating QR code:', error);
        // Fallback without custom colors
        try {
          const uploadUrl = `${window.location.origin}/mobile-upload?session=${newSessionId}&sample=${encodeURIComponent(sampleText)}`;
          const qrUrl = await QRCode.toDataURL(uploadUrl, {
            width: 200,
            margin: 2
          });
          setQrCodeUrl(qrUrl);
        } catch (fallbackError) {
          console.error('Fallback QR regeneration also failed:', fallbackError);
        }
      }
    };
    generateQR();
    setIsPolling(true);
  };

  return (
    <Card className="p-6 space-y-4 bg-gradient-subtle border-2 border-dashed border-primary/30">
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-2">
          <Smartphone className="w-5 h-5 text-primary" />
          <h3 className="font-elegant text-lg text-ink">Upload from Phone</h3>
        </div>

        <div className="bg-white p-4 rounded-lg inline-block shadow-soft">
          {qrCodeUrl ? (
            <img src={qrCodeUrl} alt="QR Code for mobile upload" className="w-48 h-48" />
          ) : (
            <div className="w-48 h-48 bg-muted animate-pulse rounded" />
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-center gap-2 text-sm">
            {completed ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-600" />
                <span className="text-green-600 font-medium">Photo Received!</span>
              </>
            ) : isPolling ? (
              <>
                <Clock className="w-4 h-4 text-muted-foreground animate-pulse" />
                <span className="text-muted-foreground">Waiting for photo...</span>
              </>
            ) : (
              <span className="text-muted-foreground">Ready to scan</span>
            )}
          </div>

          <p className="text-xs text-muted-foreground max-w-xs mx-auto leading-relaxed">
            Scan this QR code with your phone's camera to open the mobile upload page
          </p>
        </div>

        <Button 
          variant="outline" 
          size="sm" 
          onClick={regenerateQR}
          disabled={completed}
        >
          <RefreshCw className="w-4 h-4" />
          Refresh QR Code
        </Button>

        <div className="bg-warm-accent/10 p-3 rounded text-xs text-muted-foreground space-y-1">
          <p>📱 <strong>Step 1:</strong> Scan QR code with your phone</p>
          <p>✍️ <strong>Step 2:</strong> Write the sample text on paper</p>
          <p>📸 <strong>Step 3:</strong> Take a clear photo and upload</p>
        </div>

        {/* Debug Information */}
        <details className="bg-muted/10 p-2 rounded text-xs">
          <summary className="cursor-pointer text-muted-foreground">Debug Info</summary>
          <div className="mt-2 space-y-1 text-muted-foreground">
            <p>Session ID: {sessionId}</p>
            <p>Polling: {isPolling ? 'Active' : 'Inactive'}</p>
            <p>Completed: {completed ? 'Yes' : 'No'}</p>
            <p>Storage Key: mobile-upload-{sessionId}</p>
          </div>
        </details>
      </div>
    </Card>
  );
};