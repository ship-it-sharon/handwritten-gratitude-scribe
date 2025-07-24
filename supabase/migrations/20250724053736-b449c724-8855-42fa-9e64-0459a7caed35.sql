-- Clear the invalid embedding URL to force retraining with proper Supabase storage URL
UPDATE user_style_models 
SET embedding_storage_url = NULL, 
    training_status = 'failed' 
WHERE user_id = '7acd7b34-dd19-4322-9c2c-871e8542d647';