"use client";

import { useState } from "react";
import {
  Button,
  Callout,
  Card,
  Flex,
  Heading,
  Text,
  TextField,
} from "@radix-ui/themes";
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
      <Card size="3" mt="5">
        <Flex direction="column" gap="4">
          <Heading size="5">Sign in</Heading>

          {!isSupabaseConfigured ? (
            <Callout.Root color="ruby">
              <Callout.Text>
                Almost there — the app isn&rsquo;t connected to its database
                yet. (Set NEXT_PUBLIC_SUPABASE_URL and
                NEXT_PUBLIC_SUPABASE_ANON_KEY in Vercel and redeploy.)
              </Callout.Text>
            </Callout.Root>
          ) : status === "sent" ? (
            <Callout.Root color="grass">
              <Callout.Text>
                Check your email — we sent a sign-in link to <b>{email}</b>.
              </Callout.Text>
            </Callout.Root>
          ) : (
            <>
              <Button size="3" onClick={signInWithGoogle}>
                Continue with Google
              </Button>
              <Text size="2" color="gray">
                or get a sign-in link by email:
              </Text>
              <form onSubmit={sendMagicLink}>
                <Flex direction="column" gap="3">
                  <TextField.Root
                    size="3"
                    type="email"
                    required
                    placeholder="you@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                  <Flex>
                    <Button
                      variant="soft"
                      size="3"
                      type="submit"
                      disabled={status === "sending"}
                    >
                      {status === "sending" ? "Sending…" : "Email me a link"}
                    </Button>
                  </Flex>
                </Flex>
              </form>
              {status === "error" && (
                <Callout.Root color="red">
                  <Callout.Text>
                    Something went wrong: {errorMessage}
                  </Callout.Text>
                </Callout.Root>
              )}
            </>
          )}
        </Flex>
      </Card>
    </main>
  );
}
