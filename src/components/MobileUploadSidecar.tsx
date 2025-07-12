import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Smartphone, RefreshCw, CheckCircle, Clock } from "lucide-react";
import QRCode from "qrcode";

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
            dark: 'hsl(215, 85%, 25%)', // ink color
            light: '#FFFFFF'
          }
        });
        setQrCodeUrl(qrUrl);
      } catch (error) {
        console.error('Error generating QR code:', error);
      }
    };

    generateQR();
  }, [sessionId, sampleText]);

  // Poll for uploaded images (in a real app, this would use WebSockets or Supabase real-time)
  useEffect(() => {
    if (!isPolling || completed) return;

    const pollForImage = async () => {
      try {
        // Simulate checking for uploaded images
        // In a real app, this would check a database or storage
        const stored = localStorage.getItem(`mobile-upload-${sessionId}`);
        if (stored) {
          const data = JSON.parse(stored);
          onImageReceived(data.imageUrl);
          setIsPolling(false);
          localStorage.removeItem(`mobile-upload-${sessionId}`);
        }
      } catch (error) {
        console.error('Error polling for image:', error);
      }
    };

    const interval = setInterval(pollForImage, 2000);
    return () => clearInterval(interval);
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
            dark: 'hsl(215, 85%, 25%)',
            light: '#FFFFFF'
          }
        });
        setQrCodeUrl(qrUrl);
      } catch (error) {
        console.error('Error regenerating QR code:', error);
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
      </div>
    </Card>
  );
};