-- Create a table for temporary mobile uploads
CREATE TABLE public.mobile_uploads (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE,
  image_data TEXT NOT NULL,
  sample_text TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (now() + interval '1 hour')
);

-- Enable Row Level Security
ALTER TABLE public.mobile_uploads ENABLE ROW LEVEL SECURITY;

-- Create policy to allow anyone to insert (for mobile uploads)
CREATE POLICY "Anyone can create mobile uploads" 
ON public.mobile_uploads 
FOR INSERT 
WITH CHECK (true);

-- Create policy to allow anyone to read (for desktop polling)
CREATE POLICY "Anyone can read mobile uploads" 
ON public.mobile_uploads 
FOR SELECT 
USING (true);

-- Create policy to allow deletion of expired uploads
CREATE POLICY "Anyone can delete expired uploads" 
ON public.mobile_uploads 
FOR DELETE 
USING (expires_at < now());

-- Create function to clean up expired uploads
CREATE OR REPLACE FUNCTION public.cleanup_expired_mobile_uploads()
RETURNS void AS $$
BEGIN
  DELETE FROM public.mobile_uploads WHERE expires_at < now();
END;
$$ LANGUAGE plpgsql;

-- Create index for better performance
CREATE INDEX idx_mobile_uploads_session_id ON public.mobile_uploads(session_id);
CREATE INDEX idx_mobile_uploads_expires_at ON public.mobile_uploads(expires_at);