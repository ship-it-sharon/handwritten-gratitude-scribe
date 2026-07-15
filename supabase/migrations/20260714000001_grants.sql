-- Fixup for migration 20260714000000: table grants for the signed-in role.
-- RLS governs which rows a user sees; these grants let the authenticated
-- role reach the tables at all. Scoped to `authenticated` only (not
-- `anon`) per least-privilege — no signed-out surface touches these
-- tables. Future migrations must include grants for any new table, plus
-- the default privilege below keeps this from biting again.

grant usage on schema public to authenticated;

grant select, insert, update, delete
  on public.events,
     public.contacts,
     public.households,
     public.household_members,
     public.addresses,
     public.event_recipients
  to authenticated;

-- profiles: users read/update their own row (insert stays trigger-only).
grant select, update on public.profiles to authenticated;

-- Any table created in future migrations gets the same grants
-- automatically.
alter default privileges in schema public
  grant select, insert, update, delete on tables to authenticated;
