-- Add sample fingerprint tracking to user_style_models table
ALTER TABLE user_style_models 
ADD COLUMN IF NOT EXISTS sample_fingerprint TEXT;

-- Add index for faster fingerprint lookups
CREATE INDEX IF NOT EXISTS idx_user_style_models_fingerprint 
ON user_style_models(user_id, sample_fingerprint);