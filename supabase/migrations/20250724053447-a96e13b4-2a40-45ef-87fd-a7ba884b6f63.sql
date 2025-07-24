-- Make handwriting-embeddings bucket public so Modal can access the URLs
UPDATE storage.buckets 
SET public = true 
WHERE name = 'handwriting-embeddings';