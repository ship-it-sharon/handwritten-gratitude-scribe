"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { createClient } from "../../../lib/supabase/server";

async function requireUser() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");
  return { supabase, user };
}

export async function createEvent(formData: FormData) {
  const { supabase, user } = await requireUser();

  const title = String(formData.get("title") ?? "").trim();
  const occasionType = String(formData.get("occasion_type") ?? "other");
  const eventDate = String(formData.get("event_date") ?? "").trim();

  // On any failure, send the typed values back so the form re-fills
  // instead of wiping her input.
  const keepInput = new URLSearchParams({
    title,
    occasion_type: occasionType,
    event_date: eventDate,
  });

  if (!title) redirect(`/app/events/new?error=title&${keepInput}`);

  const { data, error } = await supabase
    .from("events")
    .insert({
      user_id: user.id,
      title,
      occasion_type: occasionType,
      event_date: eventDate || null,
    })
    .select("id")
    .single();

  if (error || !data) {
    keepInput.set("error", "save");
    keepInput.set("message", error?.message ?? "unknown error");
    redirect(`/app/events/new?${keepInput}`);
  }

  revalidatePath("/app");
  redirect(`/app/events/${data.id}`);
}

export async function addIndividualRecipient(formData: FormData) {
  const { supabase, user } = await requireUser();

  const eventId = String(formData.get("event_id") ?? "");
  const fullName = String(formData.get("full_name") ?? "").trim();
  if (!eventId || !fullName) return;

  const { data: contact, error: contactError } = await supabase
    .from("contacts")
    .insert({ user_id: user.id, full_name: fullName })
    .select("id")
    .single();

  if (contactError || !contact) return;

  await supabase.from("event_recipients").insert({
    user_id: user.id,
    event_id: eventId,
    contact_id: contact.id,
  });

  revalidatePath(`/app/events/${eventId}`);
}

export async function addHouseholdRecipient(formData: FormData) {
  const { supabase, user } = await requireUser();

  const eventId = String(formData.get("event_id") ?? "");
  const householdName = String(formData.get("household_name") ?? "").trim();
  const memberNames = String(formData.get("member_names") ?? "")
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);

  if (!eventId || !householdName) return;

  const { data: household, error: householdError } = await supabase
    .from("households")
    .insert({ user_id: user.id, name: householdName })
    .select("id")
    .single();

  if (householdError || !household) return;

  if (memberNames.length > 0) {
    const { data: members } = await supabase
      .from("contacts")
      .insert(
        memberNames.map((full_name) => ({ user_id: user.id, full_name })),
      )
      .select("id");

    if (members && members.length > 0) {
      await supabase.from("household_members").insert(
        members.map((member) => ({
          household_id: household.id,
          contact_id: member.id,
          user_id: user.id,
        })),
      );
    }
  }

  await supabase.from("event_recipients").insert({
    user_id: user.id,
    event_id: eventId,
    household_id: household.id,
  });

  revalidatePath(`/app/events/${eventId}`);
}

export async function addExistingRecipients(formData: FormData) {
  const { supabase, user } = await requireUser();

  const eventId = String(formData.get("event_id") ?? "");
  const contactIds = formData.getAll("contact_ids").map(String).filter(Boolean);
  const householdIds = formData
    .getAll("household_ids")
    .map(String)
    .filter(Boolean);

  if (!eventId || (contactIds.length === 0 && householdIds.length === 0)) {
    return;
  }

  // The unique (event_id, contact_id/household_id) constraints make
  // re-adding someone a no-op rather than a duplicate; contacts and
  // households conflict on different targets, hence two upserts.
  if (contactIds.length > 0) {
    await supabase.from("event_recipients").upsert(
      contactIds.map((contact_id) => ({
        user_id: user.id,
        event_id: eventId,
        contact_id,
      })),
      { onConflict: "event_id,contact_id", ignoreDuplicates: true },
    );
  }
  if (householdIds.length > 0) {
    await supabase.from("event_recipients").upsert(
      householdIds.map((household_id) => ({
        user_id: user.id,
        event_id: eventId,
        household_id,
      })),
      { onConflict: "event_id,household_id", ignoreDuplicates: true },
    );
  }

  revalidatePath(`/app/events/${eventId}`);
}

export async function saveAddress(formData: FormData) {
  const { supabase, user } = await requireUser();

  const contactId = String(formData.get("contact_id") ?? "");
  const householdId = String(formData.get("household_id") ?? "");
  const eventId = String(formData.get("event_id") ?? "");
  const line1 = String(formData.get("line1") ?? "").trim();
  const line2 = String(formData.get("line2") ?? "").trim();
  const city = String(formData.get("city") ?? "").trim();
  const state = String(formData.get("state") ?? "").trim();
  const postalCode = String(formData.get("postal_code") ?? "").trim();

  if ((!contactId && !householdId) || !line1 || !city) return;

  const owner = contactId
    ? { contact_id: contactId }
    : { household_id: householdId };

  const { data: existing } = await supabase
    .from("addresses")
    .select("id")
    .match(owner)
    .eq("is_current", true)
    .maybeSingle();

  const fields = {
    line1,
    line2: line2 || null,
    city,
    state: state || null,
    postal_code: postalCode || null,
    validation_status: "unvalidated" as const,
  };

  if (existing) {
    await supabase.from("addresses").update(fields).eq("id", existing.id);
  } else {
    await supabase
      .from("addresses")
      .insert({ user_id: user.id, ...owner, ...fields, source: "manual" });
  }

  revalidatePath(`/app/events/${eventId}`);
  redirect(eventId ? `/app/events/${eventId}` : "/app");
}

type CsvRow = {
  full_name: string;
  line1?: string;
  line2?: string;
  city?: string;
  state?: string;
  postal_code?: string;
};

export async function importCsvRecipients(
  eventId: string,
  rows: CsvRow[],
): Promise<{ imported: number; skipped: number }> {
  const { supabase, user } = await requireUser();

  let imported = 0;
  let skipped = 0;

  for (const row of rows) {
    const fullName = (row.full_name ?? "").trim();
    if (!fullName) {
      skipped++;
      continue;
    }

    const { data: contact } = await supabase
      .from("contacts")
      .insert({ user_id: user.id, full_name: fullName })
      .select("id")
      .single();

    if (!contact) {
      skipped++;
      continue;
    }

    if ((row.line1 ?? "").trim() && (row.city ?? "").trim()) {
      await supabase.from("addresses").insert({
        user_id: user.id,
        contact_id: contact.id,
        line1: row.line1!.trim(),
        line2: (row.line2 ?? "").trim() || null,
        city: row.city!.trim(),
        state: (row.state ?? "").trim() || null,
        postal_code: (row.postal_code ?? "").trim() || null,
        source: "csv",
      });
    }

    await supabase.from("event_recipients").insert({
      user_id: user.id,
      event_id: eventId,
      contact_id: contact.id,
    });

    imported++;
  }

  revalidatePath(`/app/events/${eventId}`);
  return { imported, skipped };
}

export async function removeRecipient(formData: FormData) {
  const { supabase } = await requireUser();

  const recipientId = String(formData.get("recipient_id") ?? "");
  const eventId = String(formData.get("event_id") ?? "");
  if (!recipientId) return;

  // Removes the recipient from the event only; the contact/household stays
  // in the address book (events select from the graph, they don't own it).
  await supabase.from("event_recipients").delete().eq("id", recipientId);

  revalidatePath(`/app/events/${eventId}`);
}
