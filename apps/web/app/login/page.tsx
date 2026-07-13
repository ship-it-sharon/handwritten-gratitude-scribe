"use client";

import { useState } from "react";
import { isSupabaseConfigured } from "../../lib/supabase/config";
import { createClient } from "../../lib/supabase/client";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<
    "idle" | "sending" | "sent" | "error"
  >("idle");
  const [errorMessage, setErrorMessage] = useState("");

  async function signInWithGoogle() {
    const supabase = createClient();
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/auth/confirm` },
    });
    if (error) {
      setStatus("error");
      setErrorMessage(error.message);
    }
  }

  async function sendMagicLink(event: React.FormEvent) {
    event.preventDefault();
    setStatus("sending");
    const supabase = createClient();
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${window.location.origin}/auth/confirm` },
    });
    if (error) {
      setStatus("error");
      setErrorMessage(error.message);
    } else {
      setStatus("sent");
    }
  }

  return (
    <main className="page">
      <h1 className="wordmark">Posy</h1>
      <div className="card stack">
        <h2>Sign in</h2>

        {!isSupabaseConfigured ? (
          <p className="notice">
            Almost there — the app isn&rsquo;t connected to its database yet.
            (Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in
            Vercel and redeploy.)
          </p>
        ) : status === "sent" ? (
          <p className="notice">
            Check your email — we sent a sign-in link to <b>{email}</b>.
          </p>
        ) : (
          <>
            <button className="button" onClick={signInWithGoogle}>
              Continue with Google
            </button>
            <p className="muted">or get a sign-in link by email:</p>
            <form className="stack" onSubmit={sendMagicLink}>
              <input
                className="input"
                type="email"
                required
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <div>
                <button
                  className="button secondary"
                  type="submit"
                  disabled={status === "sending"}
                >
                  {status === "sending" ? "Sending…" : "Email me a link"}
                </button>
              </div>
            </form>
            {status === "error" && (
              <p className="notice">Something went wrong: {errorMessage}</p>
            )}
          </>
        )}
      </div>
    </main>
  );
}
