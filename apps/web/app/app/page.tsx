import Link from "next/link";
import { redirect } from "next/navigation";
import { isSupabaseConfigured } from "../../lib/supabase/config";
import { createClient } from "../../lib/supabase/server";

const OCCASION_LABELS: Record<string, string> = {
  wedding: "Wedding",
  bridal_shower: "Bridal shower",
  baby_shower: "Baby shower",
  birthday: "Birthday",
  graduation: "Graduation",
  holiday: "Holiday",
  other: "Occasion",
};

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

  const { data: events } = await supabase
    .from("events")
    .select("id, title, occasion_type, event_date, event_recipients (id)")
    .order("created_at", { ascending: false });

  return (
    <main className="page">
      <h1 className="wordmark">Posy</h1>
      <p className="muted">{user.email}</p>

      <div className="card stack">
        <h2>Your events</h2>
        {!events || events.length === 0 ? (
          <p className="muted">
            Nothing here yet. An event is the occasion you&rsquo;re saying
            thank you for — a wedding, a shower, a birthday.
          </p>
        ) : (
          <ul className="recipient-list">
            {events.map((event) => (
              <li key={event.id} className="recipient-row">
                <div>
                  <Link href={`/app/events/${event.id}`}>{event.title}</Link>
                  <div className="muted">
                    {OCCASION_LABELS[event.occasion_type] ?? "Occasion"}
                    {event.event_date ? ` · ${event.event_date}` : ""}
                  </div>
                </div>
                <span className="chip">
                  {(event.event_recipients ?? []).length} recipient
                  {(event.event_recipients ?? []).length === 1 ? "" : "s"}
                </span>
              </li>
            ))}
          </ul>
        )}
        <div>
          <Link className="button" href="/app/events/new">
            Create an event
          </Link>
        </div>
      </div>

      <div className="card stack">
        <form action="/auth/signout" method="post">
          <button className="button secondary" type="submit">
            Sign out
          </button>
        </form>
      </div>
    </main>
  );
}
