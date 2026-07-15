import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { createClient } from "../../../../lib/supabase/server";
import {
  addExistingRecipients,
  addHouseholdRecipient,
  addIndividualRecipient,
  removeRecipient,
} from "../actions";
import { SubmitButton } from "../../../components/SubmitButton";

const OCCASION_LABELS: Record<string, string> = {
  wedding: "Wedding",
  bridal_shower: "Bridal shower",
  baby_shower: "Baby shower",
  birthday: "Birthday",
  graduation: "Graduation",
  holiday: "Holiday",
  other: "Occasion",
};

export default async function EventPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ imported?: string; skipped?: string }>;
}) {
  const { id } = await params;
  const { imported, skipped } = await searchParams;
  const supabase = await createClient();

  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { data: event } = await supabase
    .from("events")
    .select("id, title, occasion_type, event_date")
    .eq("id", id)
    .single();

  if (!event) notFound();

  const { data: recipients } = await supabase
    .from("event_recipients")
    .select(
      `id,
       contact:contacts (id, full_name),
       household:households (id, name, household_members (contact:contacts (full_name))),
       created_at`,
    )
    .eq("event_id", id)
    .order("created_at", { ascending: true });

  const { data: addresses } = await supabase
    .from("addresses")
    .select("contact_id, household_id")
    .eq("is_current", true);

  const addressedContacts = new Set(
    (addresses ?? []).map((a) => a.contact_id).filter(Boolean),
  );
  const addressedHouseholds = new Set(
    (addresses ?? []).map((a) => a.household_id).filter(Boolean),
  );

  const rows = (recipients ?? []).map((r) => {
    const contact = Array.isArray(r.contact) ? r.contact[0] : r.contact;
    const household = Array.isArray(r.household)
      ? r.household[0]
      : r.household;
    const memberNames = household
      ? (household.household_members ?? [])
          .map((m) => {
            const c = Array.isArray(m.contact) ? m.contact[0] : m.contact;
            return c?.full_name;
          })
          .filter(Boolean)
          .join(", ")
      : "";
    const hasAddress = household
      ? addressedHouseholds.has(household.id)
      : contact
        ? addressedContacts.has(contact.id)
        : false;
    return {
      id: r.id,
      name: household ? household.name : (contact?.full_name ?? "—"),
      detail: household
        ? memberNames
          ? `Household · ${memberNames}`
          : "Household"
        : null,
      hasAddress,
      addressHref: household
        ? `/app/recipients/address?household=${household.id}&event=${event.id}`
        : contact
          ? `/app/recipients/address?contact=${contact.id}&event=${event.id}`
          : null,
    };
  });

  // Address book: everyone from any event who isn't already in this one.
  const inEventContacts = new Set(
    (recipients ?? [])
      .map((r) => {
        const c = Array.isArray(r.contact) ? r.contact[0] : r.contact;
        return c?.id;
      })
      .filter(Boolean),
  );
  const inEventHouseholds = new Set(
    (recipients ?? [])
      .map((r) => {
        const h = Array.isArray(r.household) ? r.household[0] : r.household;
        return h?.id;
      })
      .filter(Boolean),
  );

  const { data: allContacts } = await supabase
    .from("contacts")
    .select("id, full_name, household_members (household_id)")
    .order("full_name");
  const { data: allHouseholds } = await supabase
    .from("households")
    .select("id, name")
    .order("name");

  const bookHouseholds = (allHouseholds ?? []).filter(
    (h) => !inEventHouseholds.has(h.id),
  );
  // Contacts who belong to a household are represented by it; individuals
  // stand alone.
  const bookContacts = (allContacts ?? []).filter(
    (c) =>
      !inEventContacts.has(c.id) &&
      (c.household_members ?? []).length === 0,
  );
  const hasBookCandidates = bookContacts.length + bookHouseholds.length > 0;

  return (
    <main className="page">
      <p>
        <Link href="/app">&larr; All events</Link>
      </p>
      <h1 className="wordmark">{event.title}</h1>
      <p className="muted">
        {OCCASION_LABELS[event.occasion_type] ?? "Occasion"}
        {event.event_date ? ` · ${event.event_date}` : ""}
      </p>

      {imported && (
        <p className="notice">
          Imported {imported} {Number(imported) === 1 ? "person" : "people"}
          {skipped && Number(skipped) > 0
            ? ` (${skipped} rows skipped — no name found)`
            : ""}
          .
        </p>
      )}

      <div className="card stack">
        <h2>Recipients ({rows.length})</h2>
        <p className="muted">
          A name is enough to start a card — addresses can wait until
          checkout.
        </p>

        {rows.length === 0 ? (
          <p className="muted">No one yet. Add your first below.</p>
        ) : (
          <ul className="recipient-list">
            {rows.map((row) => (
              <li key={row.id} className="recipient-row">
                <div>
                  <div>{row.name}</div>
                  {row.detail && <div className="muted">{row.detail}</div>}
                </div>
                <div className="recipient-side">
                  {row.addressHref ? (
                    <Link className="chip" href={row.addressHref}>
                      {row.hasAddress ? "✓ has address" : "📮 needs address"}
                    </Link>
                  ) : (
                    <span className="chip">
                      {row.hasAddress ? "✓ has address" : "📮 needs address"}
                    </span>
                  )}
                  <form action={removeRecipient}>
                    <input type="hidden" name="recipient_id" value={row.id} />
                    <input type="hidden" name="event_id" value={event.id} />
                    <SubmitButton className="link-button" pendingLabel="removing…">
                      remove
                    </SubmitButton>
                  </form>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {hasBookCandidates && (
        <div className="card stack">
          <h2>Add from your address book</h2>
          <p className="muted">
            People and households from your other events.
          </p>
          <form className="stack" action={addExistingRecipients}>
            <input type="hidden" name="event_id" value={event.id} />
            <ul className="recipient-list">
              {bookHouseholds.map((h) => (
                <li key={h.id} className="recipient-row">
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      name="household_ids"
                      value={h.id}
                    />
                    <span>
                      {h.name} <span className="muted">· household</span>
                    </span>
                  </label>
                </li>
              ))}
              {bookContacts.map((c) => (
                <li key={c.id} className="recipient-row">
                  <label className="checkbox-row">
                    <input type="checkbox" name="contact_ids" value={c.id} />
                    <span>{c.full_name}</span>
                  </label>
                </li>
              ))}
            </ul>
            <div>
              <SubmitButton
                className="button secondary"
                pendingLabel="Adding…"
              >
                Add selected
              </SubmitButton>
            </div>
          </form>
        </div>
      )}

      <div className="card stack">
        <h2>Bring in a whole list</h2>
        <p className="muted">
          Got a spreadsheet — a guest list, a registry export? Import
          everyone at once.
        </p>
        <div>
          <Link className="button secondary" href={`/app/events/${event.id}/import`}>
            Import from CSV
          </Link>
        </div>
      </div>

      <div className="card stack">
        <h2>Add a person</h2>
        <form className="stack row-on-wide" action={addIndividualRecipient}>
          <input type="hidden" name="event_id" value={event.id} />
          <input
            className="input"
            name="full_name"
            required
            placeholder="Aunt Carol"
          />
          <SubmitButton pendingLabel="Adding…">Add</SubmitButton>
        </form>

        <h2>Add a household</h2>
        <p className="muted">
          One card for the whole family — members are remembered
          individually.
        </p>
        <form className="stack" action={addHouseholdRecipient}>
          <input type="hidden" name="event_id" value={event.id} />
          <input
            className="input"
            name="household_name"
            required
            placeholder="The Chens"
          />
          <input
            className="input"
            name="member_names"
            placeholder="Members, comma-separated: Wei Chen, Lily Chen (optional)"
          />
          <div>
            <SubmitButton className="button secondary" pendingLabel="Adding household…">
              Add household
            </SubmitButton>
          </div>
        </form>
      </div>
    </main>
  );
}
