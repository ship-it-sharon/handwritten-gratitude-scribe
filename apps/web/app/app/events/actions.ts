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

  if (!title) redirect("/app/events/new?error=title");

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

  if (error || !data) redirect("/app/events/new?error=save");

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
