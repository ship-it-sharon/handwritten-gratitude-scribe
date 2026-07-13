export const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
export const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

// The skeleton must build and render before the project's env vars are
// configured in Vercel, so every Supabase entry point checks this first.
export const isSupabaseConfigured = Boolean(supabaseUrl && supabaseAnonKey);
