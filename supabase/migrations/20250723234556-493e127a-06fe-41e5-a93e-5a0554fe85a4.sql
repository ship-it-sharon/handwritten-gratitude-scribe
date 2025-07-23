-- Reset the training status for all users to allow retraining
-- This removes stale "completed" status that may have been set incorrectly by fallback logic

UPDATE user_style_models 
SET training_status = 'pending',
    training_started_at = NULL,
    training_completed_at = NULL,
    style_model_path = NULL
WHERE training_status = 'completed';

-- Also clean up any failed training attempts to start fresh
UPDATE user_style_models 
SET training_status = 'pending',
    training_started_at = NULL,
    training_completed_at = NULL  
WHERE training_status = 'failed';