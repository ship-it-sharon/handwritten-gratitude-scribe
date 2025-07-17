-- Create table to store user style models and training status
CREATE TABLE public.user_style_models (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  model_id TEXT NOT NULL UNIQUE,
  training_status TEXT NOT NULL DEFAULT 'pending' CHECK (training_status IN ('pending', 'training', 'completed', 'failed')),
  style_model_path TEXT,
  sample_images JSONB, -- Store the 5 handwriting samples used for training
  training_started_at TIMESTAMP WITH TIME ZONE,
  training_completed_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.user_style_models ENABLE ROW LEVEL SECURITY;

-- Create policies for user access
CREATE POLICY "Users can view their own style models" 
ON public.user_style_models 
FOR SELECT 
USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can create their own style models" 
ON public.user_style_models 
FOR INSERT 
WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update their own style models" 
ON public.user_style_models 
FOR UPDATE 
USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can delete their own style models" 
ON public.user_style_models 
FOR DELETE 
USING (auth.uid()::text = user_id::text);

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_user_style_models_updated_at
BEFORE UPDATE ON public.user_style_models
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Create index for better performance
CREATE INDEX idx_user_style_models_user_id ON public.user_style_models(user_id);
CREATE INDEX idx_user_style_models_model_id ON public.user_style_models(model_id);
CREATE INDEX idx_user_style_models_status ON public.user_style_models(training_status);