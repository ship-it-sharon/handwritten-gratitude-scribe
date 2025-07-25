-- Create storage bucket for handwriting style tensors
INSERT INTO storage.buckets (id, name, public) 
VALUES ('style-tensors', 'style-tensors', true);

-- Create RLS policies for the style-tensors bucket
CREATE POLICY "Allow public read access to style tensors" 
ON storage.objects 
FOR SELECT 
USING (bucket_id = 'style-tensors');

CREATE POLICY "Allow authenticated users to upload style tensors" 
ON storage.objects 
FOR INSERT 
WITH CHECK (bucket_id = 'style-tensors' AND auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated users to update their style tensors" 
ON storage.objects 
FOR UPDATE 
USING (bucket_id = 'style-tensors' AND auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated users to delete their style tensors" 
ON storage.objects 
FOR DELETE 
USING (bucket_id = 'style-tensors' AND auth.role() = 'authenticated');