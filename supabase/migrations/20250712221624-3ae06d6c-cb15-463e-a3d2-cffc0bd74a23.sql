-- Add UPDATE policy for mobile uploads
CREATE POLICY "Anyone can update mobile uploads" 
ON public.mobile_uploads 
FOR UPDATE 
USING (true)
WITH CHECK (true);