-- Add storage URL field for persistent embedding storage
ALTER TABLE user_style_models 
ADD COLUMN IF NOT EXISTS embedding_storage_url TEXT;

-- Create storage bucket for handwriting embeddings
INSERT INTO storage.buckets (id, name, public) 
VALUES ('handwriting-embeddings', 'handwriting-embeddings', false)
ON CONFLICT (id) DO NOTHING;

-- Create storage policies for embeddings
CREATE POLICY "Users can access their own embeddings" 
ON storage.objects 
FOR SELECT 
USING (bucket_id = 'handwriting-embeddings' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "System can upload embeddings" 
ON storage.objects 
FOR INSERT 
WITH CHECK (bucket_id = 'handwriting-embeddings');

CREATE POLICY "System can update embeddings" 
ON storage.objects 
FOR UPDATE 
USING (bucket_id = 'handwriting-embeddings');

CREATE POLICY "System can delete embeddings" 
ON storage.objects 
FOR DELETE 
USING (bucket_id = 'handwriting-embeddings');