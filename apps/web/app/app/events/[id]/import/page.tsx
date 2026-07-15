import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { createClient } from "../../../../../lib/supabase/server";
import { ImportCsv } from "./ImportCsv";

export default async function ImportPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { data: event } = await supabase
    .from("events")
    .select("id, title")
    .eq("id", id)
    .single();
  if (!event) notFound();

  return (
    <main className="page">
      <p>
        <Link href={`/app/events/${id}`}>&larr; Back to {event.title}</Link>
      </p>
      <h1 className="wordmark">Import your list</h1>
      <div className="card stack">
        <p className="muted">
          Upload the spreadsheet you already have — a wedding guest list, a
          shower invite list, an export from Zola or The Knot. Names are
          enough; addresses come along if they&rsquo;re there.
        </p>
        <ImportCsv eventId={id} />
      </div>
    </main>
  );
}
