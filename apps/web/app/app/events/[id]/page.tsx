import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import {
  Badge,
  Button,
  Callout,
  Card,
  Checkbox,
  Flex,
  Heading,
  Separator,
  Text,
  TextField,
} from "@radix-ui/themes";
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
  const bookContacts = (allContacts ?? []).filter(
    (c) =>
      !inEventContacts.has(c.id) && (c.household_members ?? []).length === 0,
  );
  const hasBookCandidates = bookContacts.length + bookHouseholds.length > 0;

  return (
    <main className="page">
      <Text as="p" size="2">
        <Link href="/app">&larr; All events</Link>
      </Text>
      <h1 className="wordmark">{event.title}</h1>
      <Text as="p" size="2" color="gray">
        {OCCASION_LABELS[event.occasion_type] ?? "Occasion"}
        {event.event_date ? ` · ${event.event_date}` : ""}
      </Text>

      {imported && (
        <Callout.Root color="grass" mt="4">
          <Callout.Text>
            Imported {imported} {Number(imported) === 1 ? "person" : "people"}
            {skipped && Number(skipped) > 0
              ? ` (${skipped} rows skipped — no name found)`
              : ""}
            .
          </Callout.Text>
        </Callout.Root>
      )}

      <Card size="3" mt="5">
        <Flex direction="column" gap="4">
          <Heading size="5">Recipients ({rows.length})</Heading>
          <Text as="p" size="2" color="gray">
            A name is enough to start a card — addresses can wait until
            checkout.
          </Text>

          {rows.length === 0 ? (
            <Text as="p" color="gray">
              No one yet. Add your first below.
            </Text>
          ) : (
            <Flex direction="column">
              {rows.map((row, i) => (
                <Flex key={row.id} direction="column">
                  {i > 0 && <Separator size="4" my="2" />}
                  <Flex justify="between" align="center" gap="3">
                    <Flex direction="column">
                      <Text weight="medium">{row.name}</Text>
                      {row.detail && (
                        <Text size="2" color="gray">
                          {row.detail}
                        </Text>
                      )}
                    </Flex>
                    <Flex align="center" gap="3">
                      {row.addressHref ? (
                        <Badge
                          asChild
                          color={row.hasAddress ? "grass" : "amber"}
                          variant="soft"
                        >
                          <Link href={row.addressHref}>
                            {row.hasAddress ? "✓ has address" : "📮 needs address"}
                          </Link>
                        </Badge>
                      ) : (
                        <Badge color="gray" variant="soft">
                          {row.hasAddress ? "✓ has address" : "📮 needs address"}
                        </Badge>
                      )}
                      <form action={removeRecipient}>
                        <input
                          type="hidden"
                          name="recipient_id"
                          value={row.id}
                        />
                        <input type="hidden" name="event_id" value={event.id} />
                        <SubmitButton
                          variant="ghost"
                          size="1"
                          pendingLabel="removing…"
                        >
                          remove
                        </SubmitButton>
                      </form>
                    </Flex>
                  </Flex>
                </Flex>
              ))}
            </Flex>
          )}
        </Flex>
      </Card>

      {hasBookCandidates && (
        <Card size="3" mt="4">
          <Flex direction="column" gap="4">
            <Heading size="5">Add from your address book</Heading>
            <Text as="p" size="2" color="gray">
              People and households from your other events.
            </Text>
            <form action={addExistingRecipients}>
              <Flex direction="column" gap="3">
                <input type="hidden" name="event_id" value={event.id} />
                {bookHouseholds.map((h) => (
                  <Text as="label" key={h.id} size="2">
                    <Flex gap="2" align="center">
                      <Checkbox name="household_ids" value={h.id} />
                      <span>
                        {h.name}{" "}
                        <Text color="gray" size="1">
                          · household
                        </Text>
                      </span>
                    </Flex>
                  </Text>
                ))}
                {bookContacts.map((c) => (
                  <Text as="label" key={c.id} size="2">
                    <Flex gap="2" align="center">
                      <Checkbox name="contact_ids" value={c.id} />
                      <span>{c.full_name}</span>
                    </Flex>
                  </Text>
                ))}
                <Flex>
                  <SubmitButton variant="soft" pendingLabel="Adding…">
                    Add selected
                  </SubmitButton>
                </Flex>
              </Flex>
            </form>
          </Flex>
        </Card>
      )}

      <Card size="3" mt="4">
        <Flex direction="column" gap="4">
          <Heading size="5">Bring in a whole list</Heading>
          <Text as="p" size="2" color="gray">
            Got a spreadsheet — a guest list, a registry export? Import
            everyone at once.
          </Text>
          <Flex>
            <Button asChild variant="soft">
              <Link href={`/app/events/${event.id}/import`}>
                Import from CSV
              </Link>
            </Button>
          </Flex>
        </Flex>
      </Card>

      <Card size="3" mt="4">
        <Flex direction="column" gap="4">
          <Heading size="5">Add a person</Heading>
          <form action={addIndividualRecipient}>
            <Flex gap="3" align="center">
              <input type="hidden" name="event_id" value={event.id} />
              <TextField.Root
                name="full_name"
                required
                placeholder="Aunt Carol"
                style={{ flex: 1 }}
              />
              <SubmitButton pendingLabel="Adding…">Add</SubmitButton>
            </Flex>
          </form>

          <Separator size="4" />

          <Heading size="5">Add a household</Heading>
          <Text as="p" size="2" color="gray">
            One card for the whole family — members are remembered
            individually.
          </Text>
          <form action={addHouseholdRecipient}>
            <Flex direction="column" gap="3">
              <input type="hidden" name="event_id" value={event.id} />
              <TextField.Root
                name="household_name"
                required
                placeholder="The Chens"
              />
              <TextField.Root
                name="member_names"
                placeholder="Members, comma-separated: Wei Chen, Lily Chen (optional)"
              />
              <Flex>
                <SubmitButton variant="soft" pendingLabel="Adding household…">
                  Add household
                </SubmitButton>
              </Flex>
            </Flex>
          </form>
        </Flex>
      </Card>
    </main>
  );
}
