-- Add unique constraint on user_id to ensure one style model per user
ALTER TABLE user_style_models 
ADD CONSTRAINT user_style_models_user_id_unique UNIQUE (user_id);