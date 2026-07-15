import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { createClient } from "../../../../lib/supabase/server";
import { saveAddress } from "../../events/actions";
import { SubmitButton } from "../../../components/SubmitButton";

export default async function AddressPage({
  searchParams,
}: {
  searchParams: Promise<{
    contact?: string;
    household?: string;
    event?: string;
  }>;
}) {
  const { contact: contactId, household: householdId, event: eventId } =
    await searchParams;

  if (!contactId && !householdId) notFound();

  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  let recipientName = "";
  if (contactId) {
    const { data } = await supabase
      .from("contacts")
      .select("full_name")
      .eq("id", contactId)
      .single();
    if (!data) notFound();
    recipientName = data.full_name;
  } else {
    const { data } = await supabase
      .from("households")
      .select("name")
      .eq("id", householdId!)
      .single();
    if (!data) notFound();
    recipientName = data.name;
  }

  const owner = contactId
    ? { contact_id: contactId }
    : { household_id: householdId! };
  const { data: address } = await supabase
    .from("addresses")
    .select("line1, line2, city, state, postal_code")
    .match(owner)
    .eq("is_current", true)
    .maybeSingle();

  const backHref = eventId ? `/app/events/${eventId}` : "/app";

  return (
    <main className="page">
      <p>
        <Link href={backHref}>&larr; Back</Link>
      </p>
      <h1 className="wordmark">Mailing address</h1>
      <p className="muted">for {recipientName}</p>
      <div className="card stack">
        <form className="stack" action={saveAddress}>
          {contactId && (
            <input type="hidden" name="contact_id" value={contactId} />
          )}
          {householdId && (
            <input type="hidden" name="household_id" value={householdId} />
          )}
          {eventId && <input type="hidden" name="event_id" value={eventId} />}
          <label className="stack">
            <span>Street address</span>
            <input
              className="input"
              name="line1"
              required
              placeholder="123 Oak Street"
              defaultValue={address?.line1 ?? ""}
            />
          </label>
          <label className="stack">
            <span>Apt / unit (optional)</span>
            <input
              className="input"
              name="line2"
              defaultValue={address?.line2 ?? ""}
            />
          </label>
          <label className="stack">
            <span>City</span>
            <input
              className="input"
              name="city"
              required
              defaultValue={address?.city ?? ""}
            />
          </label>
          <div className="stack row-on-wide">
            <label className="stack" style={{ flex: 1 }}>
              <span>State</span>
              <input
                className="input"
                name="state"
                defaultValue={address?.state ?? ""}
              />
            </label>
            <label className="stack" style={{ flex: 1 }}>
              <span>ZIP</span>
              <input
                className="input"
                name="postal_code"
                defaultValue={address?.postal_code ?? ""}
              />
            </label>
          </div>
          <div>
            <SubmitButton pendingLabel="Saving…">Save address</SubmitButton>
          </div>
        </form>
      </div>
    </main>
  );
}
