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

      <div className="card stack">
        <h2>What the file should look like</h2>
        <p className="muted">
          Any CSV works as long as there&rsquo;s a header row and a name
          column — we&rsquo;ll figure out the rest and let you correct the
          mapping before anything imports. Columns we understand:
        </p>
        <ul className="muted" style={{ paddingLeft: "1.2rem" }}>
          <li><b>Name</b> — required; one person or family per row</li>
          <li><b>Street Address</b> and <b>Apt/Unit</b> — optional</li>
          <li><b>City</b>, <b>State</b>, <b>Zip</b> — optional</li>
        </ul>
        <p className="muted">
          Rows with just a name are fine — addresses can be added any time
          before mailing. Extra columns are simply skipped.
        </p>
        <div>
          <a className="button secondary" href="/posy-import-template.csv" download>
            Download a template
          </a>
        </div>
        <p className="muted">
          Using Google Sheets or Excel? File → Download (or Save As) →
          CSV.
        </p>
      </div>
    </main>
  );
}
