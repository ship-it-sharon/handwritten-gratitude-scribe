import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { Card, Flex, Text } from "@radix-ui/themes";
import { createClient } from "../../../../lib/supabase/server";
import { saveAddress } from "../../events/actions";
import { SubmitButton } from "../../../components/SubmitButton";
import { AddressFields } from "./AddressFields";

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
      <Text as="p" size="2">
        <Link href={backHref}>&larr; Back</Link>
      </Text>
      <h1 className="wordmark">Mailing address</h1>
      <Text as="p" size="2" color="gray">
        for {recipientName}
      </Text>
      <Card size="3" mt="5">
        <form action={saveAddress}>
          <Flex direction="column" gap="4">
            {contactId && (
              <input type="hidden" name="contact_id" value={contactId} />
            )}
            {householdId && (
              <input type="hidden" name="household_id" value={householdId} />
            )}
            {eventId && (
              <input type="hidden" name="event_id" value={eventId} />
            )}
            <AddressFields defaults={address ?? {}} />
            <Flex>
              <SubmitButton pendingLabel="Saving…" size="3">
                Save address
              </SubmitButton>
            </Flex>
          </Flex>
        </form>
      </Card>
    </main>
  );
}
