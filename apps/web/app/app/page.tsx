import { redirect } from "next/navigation";
import { isSupabaseConfigured } from "../../lib/supabase/config";
import { createClient } from "../../lib/supabase/server";

export default async function AppHome() {
  if (!isSupabaseConfigured) {
    redirect("/login");
  }

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  return (
    <main className="page">
      <h1 className="wordmark">Posy</h1>
      <div className="card stack">
        <h2>Welcome{user.email ? `, ${user.email}` : ""}</h2>
        <p>
          You&rsquo;re signed in. This is where your events will live —
          &ldquo;Sarah &amp; Tom&rsquo;s Wedding,&rdquo; &ldquo;Baby shower
          for Emma&rdquo; — each one a stack of thank-you cards from start to
          mailbox.
        </p>
        <div>
          <button className="button" disabled>
            Create an event (coming in M1)
          </button>
        </div>
        <form action="/auth/signout" method="post">
          <button className="button secondary" type="submit">
            Sign out
          </button>
        </form>
      </div>
    </main>
  );
}
